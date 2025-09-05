import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse 

parser = argparse.ArgumentParser()

parser.add_argument("--data_prefix",default="data")
parser.add_argument("--epochs",type=int,default=6)
parser.add_argument("--batch",type=int,default=64)
parser.add_argument("--emb",type=int,default=64)
parser.add_argument("--hid",type=int,default=128)

args=parser.parse_args()

X=np.load(args.data_prefix + "_X.npy")
y=np.load(args.data_prefix + "_y.npy")

vocab= int(X.max()) + 1

tensor_x=torch.tensor(X,dtype=torch.long)
tensor_y=torch.tensor(y,dtype=torch.float32)

dataset = TensorDataset(tensor_x,tensor_y)
loader = DataLoader(dataset,batch_size=args.batch,shuffle=True)

class Model(nn.Module):
    def __init__(self,vocab,emb,hid):
        super().__init__()
        self.emb=nn.Embedding(vocab,emb,padding_idx=0)
        self.lstm=nn.LSTM(emb,hid,batch_first=True)
        self.fc=nn.Linear(hid,1)

    def forward(self,x):
        e=self.emb(x)
        o,_=self.lstm(e)
        out=o[:,-1,:]
        return torch.sigmoid(self.fc(out)).squeeze(1)

model=Model(vocab,args.emb,args.hid)
opt=torch.optim.Adam(model.parameters(),lr=1e-3)
crit=nn.BCELoss()

for epoch in range(args.epochs):
    total_loss=0.0
    for xb,yb in loader:
        opt.zero_grad()
        out=model(xb)
        loss=crit(out,yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1}/{args.epochs} :Loss = {total_loss/len(dataset):.4f}")    

torch.save(model.state_dict(),args.data_prefix + "_model.pth")
print("Saved model :",args.data_prefix + "_model.pth")

