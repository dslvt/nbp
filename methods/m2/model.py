import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class FPMC(nn.Module):

    def __init__(self, config, numItems, numUsers, device):
        super(FPMC, self).__init__()

        self.dim = config.dim
        self.EUI = nn.Embedding(numUsers, self.dim).to(device)
        self.EIU = nn.Embedding(numItems+1, self.dim, padding_idx=config.padIdx).to(device)

        self.EIL = nn.Embedding(numItems+1, self.dim, padding_idx=config.padIdx).to(device)
        self.ELI = nn.Embedding(numItems+1, self.dim, padding_idx=config.padIdx).to(device)

    def forward(self, u, x, tar, neg, offset, isEval):
        #for fpmc, the sequences is broken to several pairs (prev, next)
        #we use the prev to predict the next

        eui  = self.EUI(u)

        eTarU = self.EIU(tar)
        if not isEval:
            eNegU = self.EIU(neg)

        eTarL = self.EIL(tar)
        if not isEval:
            eNegL = self.EIL(neg)

        eli = self.ELI(x)

        #MF

        #FMC
        eli = eli.mean(1) * offset
        if not isEval:
            xmfTar = torch.bmm(eui.unsqueeze(1), eTarU.permute(0,2,1)).squeeze()
            xfpTar = torch.bmm(eTarL, eli.unsqueeze(1).permute(0,2,1)).squeeze()
        else:
            xmfTar = eui.mm(eTarU.permute(1,0))
            xfpTar = eTarL.mm(eli.permute(1,0)).permute(1,0)

        #dot = mul and sum over the last dim
        scoreTar = xmfTar + xfpTar

        scoreNeg = 0

        if not isEval:
            xmfNeg = torch.bmm(eui.unsqueeze(1), eNegU.permute(0,2,1)).squeeze()
            xfpNeg = torch.bmm(eNegL, eli.unsqueeze(1).permute(0,2,1)).squeeze()
            scoreNeg = xmfNeg + xfpNeg

        return scoreTar, scoreNeg

class FREQ(nn.Module):
    def __init__(self, numItems, device):
        super(FREQ, self).__init__()
        #self.global_freq is the vector contains the learned global frequency for each item
        #self.alf is a scaler to combine the personalized frequency and learned global frequency

        self.global_freq = nn.Embedding(1, numItems).to(device)
        self.input = torch.LongTensor([0]).to(device)
        self.alf = nn.Parameter(torch.rand(1)).to(device)
        self.soft = nn.Softmax(dim=1)

    def forward(self, his):
        alf = torch.sigmoid(self.alf)
        global_freq = self.global_freq(self.input)
        global_freq = self.soft(global_freq).reshape(-1)

        res = alf * his + (1.0-alf) * global_freq
        return res
    

class FREQP(nn.Module):
    def __init__(self, numItems, device):
        super(FREQP, self).__init__()
        #self.global_freq is the vector contains the learned global frequency for each item
        #self.alf is a scaler to combine the personalized frequency and learned global frequency

        #here dim represents the embedding for the hidden layer in gates
        self.dim = config.dim

        self.global_freq = nn.Embedding(1, numItems).to(device)
        self.input = torch.LongTensor([0]).to(device)
        self.soft = nn.Softmax(dim=1)

        self.his_embds    = nn.Linear(numItems, self.dim)
        self.global_embds = nn.Linear(numItems, self.dim)
        self.gate_his     = nn.Linear(self.dim, 1)
        self.gate_global  = nn.Linear(self.dim, 1)


    def forward(self, his):
        global_freq = self.global_freq(self.input)
        global_freq_soft = self.soft(global_freq).reshape(-1)

        embds_his = self.his_embds(his)
        embds_global = self.global_embds(global_freq)

        gate = torch.sigmoid(self.gate_his(embds_his) + self.gate_global(embds_global))

        res = gate * his + (1.0-gate) * global_freq_soft
        return res
    
class FacTor(nn.Module):
    def __init__(self, numItems, dim, device):
        super(FacTor, self).__init__()

        self.outEmbs = nn.Embedding(numItems, dim).to(device)
    
    def forward(self, rowIndex, colIndex):
        rows = self.outEmbs(rowIndex)
        cols = self.outEmbs(colIndex)

        res = (rows * cols).sum(-1)

        return res
    

class SNBR(nn.Module):
    def __init__(self, config, numItems, device, weights):
        super(SNBR, self).__init__()

        self.dim     = config.dim
        self.itemEmb = nn.Embedding(numItems+1, self.dim, padding_idx=config.padIdx).to(device)
        self.out     = nn.Linear(self.dim, numItems)

        self.his_embds = nn.Linear(numItems, self.dim)
        self.gate_his  = nn.Linear(self.dim, 1)
        self.gate_trans= nn.Linear(self.dim, 1)

        if config.isPreTrain:
            with torch.no_grad():
                self.out.weight.copy_(weights)

    def forward(self, x, decay, offset, his, isEval):
        #x:    3d: batch_size * max_num_seq * max_num_bas
        #embs: 4d: batch_size * max_num_seq * max_num_bas * dim
        #decay:3d: batch_size * max_num_bas * 1
        #his:  2d: batch_size * numItems
 
        embs   = self.itemEmb(x)

        if not isEval:
            embs = F.dropout(embs)

        embs3d = decay * embs.sum(2)
        embs2d = torch.tanh(embs3d.sum(1))

        scores_trans = self.out(embs2d)
        scores_trans = F.softmax(scores_trans, dim=-1)

        scores = scores_trans

        embs_his = self.his_embds(his)
        gate = torch.sigmoid(self.gate_his(embs_his) + self.gate_trans(embs2d))

        scores = gate * scores_trans + (1-gate) * his
        return scores, gate