import json
from bcc import BPF

bpf_text = r"""

#include <linux/sched.h>

struct data_t {
    u64 ts;
    u32 prev_pid;
    u32 next_pid;
    char prev_comm[TASK_COMM_LEN];
    char next_comm[TASK_COMM_LEN];
};
BPF_PREF_OUTPUT(events);

int trace_sched_switch(struct tracepoint__sched__sched_switch *ctx) {
    struct data_t data = {};
    data.ts = bpf_ktime_get_ns();
    data.prev_pid = ctx->prev_pid;
    data.next_pid = ctx->next_pid;
    bpf_probe_read_str(data.prev_comm, sizeof(data.prev_comm), ctx->prev->comm);
    bpf_probe_read_str(data.next_comm, sizeof(data.next_comm), ctx->next->comm);
    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""

b=BPF(text=bpf_text)
b.attach_tracepoint(tp="sched:sched_switch", fn_name="trace_sched_switch")  

def on_event(cpu,data,size):
    ev=b["events"].event(data)
    out = {
        "ts":ev.ts,
        "prev_pid":ev.prev_pid,
        "next_pid":ev.next_pid,
        "prev_comm":ev.prev_comm.decode('utf-8', 'replace'),
        "next_comm":ev.next_comm.decode('utf-8', 'replace')
    }
    print(json.dumps(out),flush=True)

print("Tracing sched switch .. ctrl+c to stop")
b["events"].open_perf_buffer(on_event)

try:
    while True:
        b.perf_buffer_poll()
except KeyboardInterrupt:
    print("\n Stopped Tracing")        