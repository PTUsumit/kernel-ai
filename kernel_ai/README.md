# SmartOS-KernelAI (toy demo)

Steps:
1. In Ubuntu VM, install dependencies: bpfcc, python3-bpfcc, pip packages.
2. Run tracer: `sudo python3 ebpf/trace_sched.py > trace.jsonl`
3. Preprocess: `python3 models/preprocess.py --trace trace.jsonl`
4. Train: `python3 models/train_lstm.py --data_prefix data --epochs 8`
5. Export ONNX: `python3 models/export_onnx.py --data_prefix data`
6. Run agent: `sudo python3 agent/agent.py --model ../model.onnx --vocab ../data_vocab.json`
