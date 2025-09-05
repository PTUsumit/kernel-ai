import json
import numpy as np 
from collections import defaultdict, Counter
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--trace",default="../trace.jsonl")
parser.add_argument("--seq_len",type=int,default=32)
parser.add_argument("--pred_window",type=int,default=50)
parser.add_argument("--hor_threshold",type=int,default=2)
parser.add_argument("--out_prefix",default="data")
args=parser.parse_args()

pids=[]

with open(args.trace,'r') as f:
    for line in f:
        try:
            obj=json.loads(line)
            pids.append(int(obj['next_pid']))
        except Exception:
            continue

unique=sorted(list(set(pids)))

pid2idx = {pid: i+1 for i , pid in enumerate(unique)}
idx2pid = {v:k for k,v in pid2idx.items()}

seq= [pid2idx.get(pid,0) for pid in pids]

X=[]
y=[]

for i in range(args.seq_len,len(seq)-args.pred_window):
    inp=seq[i-args.seq_len:i]
    target=seq[i]
    future= seq[i+1:i+1+args.pred_window]
    cnt=Counter(future).get(target,0)
    label= 1 if cnt >= args.hot_threshold else 0
    X.append(inp)
    y.append(label)
X=np.array(X,dtype=np.int64)
y=np.array(y,dtype=np.int64)

np.save(args.out_prefix + "_X.npy",X)
np.save(args.out_prefix + "_y.npy",y)   

with open(args.out_prefix + "_vocab.json", "w") as f:
    json.dump({"pid2idx":pid2idx,"idx2pid":idx2pid},f)

print("created",X_shape, "examples, Saved:", args.out_prefix + "_X.npy", args.out_prefix + "_y.npy")
    