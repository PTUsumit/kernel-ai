# agent/agent.py
import json
import time
import subprocess
from collections import deque
import onnxruntime as ort
import numpy as np
import argparse
from bcc import BPF

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="../model.onnx")
parser.add_argument("--vocab", default="../data_vocab.json")
parser.add_argument("--seq_len", type=int, default=32)
parser.add_argument("--threshold", type=float, default=0.7)
args = parser.parse_args()

# load vocab
with open(args.vocab) as f:
    vocab = json.load(f)["pid2idx"]
# reverse-safe mapping
pid2idx = {int(k):v for k,v in vocab.items()}

sess = ort.InferenceSession(args.model)

# BPF program similar to tracer but minimal: we just need next_pid events
bpf_text = r"""
#include <linux/sched.h>
struct data_t { u64 ts; u32 next_pid; char next_comm[16]; };
BPF_PERF_OUTPUT(events);
int tracepoint__sched__sched_switch(struct tracepoint__sched__sched_switch *ctx) {
    struct data_t d = {};
    d.ts = bpf_ktime_get_ns();
    d.next_pid = ctx->next_pid;
    bpf_probe_read_kernel_str(&d.next_comm, sizeof(d.next_comm), ctx->next_comm);
    events.perf_submit(ctx, &d, sizeof(d));
    return 0;
}
"""
b = BPF(text=bpf_text)
b.attach_tracepoint(tp="sched:sched_switch", fn_name="tracepoint__sched__sched_switch")

window = deque(maxlen=args.seq_len)

def get_idx(pid):
    return pid2idx.get(pid, 0)

def adjust_nice(pid):
    # increase niceness by +10 (i.e., lower priority) to reduce hogging
    try:
        cur = int(subprocess.check_output(["ps","-o","ni=","-p",str(pid)]).decode().strip())
    except Exception:
        return
    new = min(19, cur + 10)
    subprocess.run(["renice", str(new), "-p", str(pid)])
    print(f"Adjusted nice for PID {pid}: {cur} -> {new}")

def on_event(cpu, data, size):
    ev = b["events"].event(data)
    pid = int(ev.next_pid)
    window.append(get_idx(pid))
    if len(window) < args.seq_len:
        return
    arr = np.array([list(window)], dtype=np.int64)
    # ONNX expects int64 input named "input"
    ort_out = sess.run(None, {"input": arr})
    score = float(ort_out[0][0])
    # score is probability that current / next pid will be hot
    if score >= args.threshold:
        # apply safe action: raise niceness (lower priority) of the pid (non-root usually allowed if same user)
        adjust_nice(pid)
        print("Predicted HOT pid", pid, "score", score)

b["events"].open_perf_buffer(on_event)
print("Agent started. Listening to sched_switch events... (Ctrl-C to stop)")
try:
    while True:
        b.perf_buffer_poll(timeout=1000)
except KeyboardInterrupt:
    print("Agent stopped.")
