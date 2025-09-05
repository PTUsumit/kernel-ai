# models/export_onnx.py
import torch
import torch.nn as nn
import numpy as np
import json
from train_lstm import Model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_prefix", default="data")
parser.add_argument("--seq_len", type=int, default=32)
parser.add_argument("--out", default="model.onnx")
args = parser.parse_args()

# load vocab to know vocab size
vocab_json = args.data_prefix + "_vocab.json"
with open(vocab_json) as f:
    vocab = json.load(f)
vsize = max(int(k) for k in vocab.get("idx2pid", {}).keys()) + 1

# reconstruct model architecture (must match train params)
model = Model(vsize+1, emb=64, hid=128)
model.load_state_dict(torch.load(args.data_prefix + "_model.pth", map_location="cpu"))
model.eval()

# export
dummy = torch.randint(0, vsize+1, (1, args.seq_len), dtype=torch.long)
torch.onnx.export(model, dummy, args.out, opset_version=11,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
print("Exported ONNX model to", args.out)
