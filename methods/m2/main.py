import torch
import numpy as np
import random
import time
from utils import *
from model.FPMC import FPMC
from model.SNBR import SNBR
from model.FREQ import FREQ
import sys
import pdb
import torch

import argparse
import logging
import pdb
import pickle

import os
import sys
##os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from subsequences import subsequences

class transactions:

    def __init__(self, dataLoader, config):
       
        if config.isTrain:
            self.numUsers = dataLoader.numTrain
            self.numItems = dataLoader.numItemsTrain
            self.train = subsequences(dataLoader.trainList)
            self.test  = subsequences(dataLoader.validList, dataLoader.valid2train)

        else:
            self.numUsers = dataLoader.numTrainVal
            self.numItems = dataLoader.numItemsTest 
            self.train = subsequences(dataLoader.trainValList)
            self.test  = subsequences(dataLoader.testList, dataLoader.test2trainVal)

def BPR(target, neg):
   
    diff = target-neg
    diff[diff==0.0] = float('Inf') 

    loss = -torch.log(torch.sigmoid(diff)+ 1e-8)
    #the average loss for each basket
    loss = loss.sum(-1).mean(-1)

    return loss

def Bernoulli(groud, scores):
    neg_ll  = -(torch.log(scores) * groud).sum(-1).mean()

    return neg_ll



def evalByUser(model, dataLoader, config, device, isTrain):
    evalBatchSize = config.batchSize
    
    if isTrain:
        numUser  = dataLoader.numValid
        numItems = dataLoader.numItemsTrain
    else:
        numUser = dataLoader.numTest
        numItems = dataLoader.numItemsTest

    numBatch = numUser // evalBatchSize + 1
    idxList  = [i for i in range(numUser)]

    Recall = []
    NDCG = []

    for batch in range(numBatch):
        start = batch * evalBatchSize
        end   = min(batch * evalBatchSize + evalBatchSize, numUser)

        batchList = idxList[start:end]

        #target is the same with targetList in evaluation
        samples, decays, his, target, offset, lenX, _ = generateBatchSamples(dataLoader, batchList, config, isEval=1)

        samples = torch.from_numpy(samples).type(torch.LongTensor).to(device)
        decays  = torch.from_numpy(decays).type(torch.FloatTensor).to(device)
        his     = torch.from_numpy(his).type(torch.FloatTensor).to(device)
        offset  = torch.from_numpy(offset).type(torch.FloatTensor).to(device)
        lenX    = lenX.to(device)

        if config.model == 'Dream' or config.model == 'FPMC':
            allItemIdx = np.asarray([i for i in range(numItems)])
            allItemIdx = torch.from_numpy(allItemIdx).type(torch.LongTensor).to(device)
            scores, _  = model.forward(samples, lenX, allItemIdx, neg=None, isEval=1)

        with torch.no_grad():
            if config.model == 'SNBR':        
                scores, gate = model.forward(samples, decays, offset, his, isEval=1)
            elif config.model == 'FREQ' or config.model == 'FREQP':
                scores = model.forward(his)

        #get the index of top 40 items
        predIdx = torch.topk(scores, 40, largest=True)[1]
        predIdx = predIdx.cpu().data.numpy().copy()

        if config.model == 'SNBR' and config.abla=="None":
            gate = gate.cpu().data.numpy().copy()

        if batch == 0:
            predIdxArray = predIdx
            targetList  = target
            if config.model == 'SNBR' and config.abla=="None":
                gateArray = gate
        else:
            predIdxArray = np.append(predIdxArray, predIdx, axis=0)
            if config.model == 'SNBR' and config.abla=="None":
                gateArray = np.append(gateArray, gate, axis=0)
            targetList  += target
            

    for k in [5, 10, 20, 40]:
        Recall.append(calRecall(targetList, predIdxArray, k))
        NDCG.append(calNDCG(targetList, predIdxArray, k))

    ##if config.model == 'SNBR' and config.abla=="None" and not config.isTrain and config.retrival:
    ##    np.savetxt('./gates/gateBest/gate_'+config.dataset+'_'+str(config.testOrder)+'.txt', gateArray.reshape(-1,1), fmt="%.5f")

    return Recall, NDCG

def evalByBas(model, trans, config, device):
    
    numBas   = len(trans.test.training)
    numBatch = numBas // config.batchSize + 1
    idxList  = [i for i in range(numBas)]
    Recall = []
    NDCG = []

    targetBatch = np.asarray([i for i in range(trans.numItems)])
    targetBatch= torch.from_numpy(targetBatch).type(torch.LongTensor).to(device)

    for batch in range(numBatch):
        start = config.batchSize * batch
        end   = min(numBas, start+config.batchSize)
 
        batchList  = idxList[start:end]
        trainBatch, groundTruth, userBatch = generateBatchBas(trans, batchList, isEval=1)

        trainBatch, offset = padBasWise(trainBatch, config)
        userBatch  = np.asarray(userBatch)

        trainBatch = torch.from_numpy(trainBatch).type(torch.LongTensor).to(device)
        offset     = torch.from_numpy(offset).type(torch.FloatTensor).to(device)
        userBatch  = torch.from_numpy(userBatch).type(torch.LongTensor).to(device)

        scoresTar, _ = model.forward(userBatch, trainBatch, targetBatch, None, offset, isEval=1)
        
        #get the index of top 40 items
        predIdx = torch.topk(scoresTar, 40, largest=True)[1]
        predIdx = predIdx.cpu().data.numpy().copy()

        if batch == 0:
            predIdxArray = predIdx
            targetList  = groundTruth
        else:
            predIdxArray = np.append(predIdxArray, predIdx, axis=0)
            targetList  += groundTruth

    for k in [5, 10, 20, 40]:
        Recall.append(calRecall(targetList, predIdxArray, k))
        NDCG.append(calNDCG(targetList, predIdxArray, k))

    return Recall, NDCG


def trainByBas(trans, config, logger, device):
   
    numBas   = len(trans.train.training)
    numBatch = numBas // config.batchSize + 1 
    idxList  = [i for i in range(numBas)]

    if config.model == 'FPMC':
        model = FPMC(config, trans.numItems, trans.numUsers, device)

    #open to more
    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
    elif config.opt == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr, weight_decay=config.l2)

    for epoch in range(config.numIter):

        random.shuffle(idxList)
        timeEpStr = time.time()
        epochLoss = 0

        for batch in range(numBatch):
            start = config.batchSize * batch
            end   = min(numBas, start+config.batchSize)

            batchList  = idxList[start:end]
            trainBatch, targetBatch, userBatch = generateBatchBas(trans, batchList, isEval=0)
            trainBatch, offset = padBasWise(trainBatch, config)
            targetBatch, negBatch = generateNegatives(targetBatch, trans.numItems, config)
            userBatch = np.asarray(userBatch)

            trainBatch = torch.from_numpy(trainBatch).type(torch.LongTensor).to(device)
            offset     = torch.from_numpy(offset).type(torch.FloatTensor).to(device)
            targetBatch= torch.from_numpy(targetBatch).type(torch.LongTensor).to(device)
            negBatch   = torch.from_numpy(negBatch).type(torch.LongTensor).to(device)
            userBatch  = torch.from_numpy(userBatch).type(torch.LongTensor).to(device)

            scoresTar, scoresNeg = model.forward(userBatch, trainBatch, targetBatch, negBatch, offset, isEval=0)

            if config.model == 'FPMC':
                loss = BPR(scoresTar, scoresNeg)
           
            epochLoss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                zeros = torch.zeros(config.dim).to(device)
                model.EIU.weight[config.padIdx].copy_(zeros)
                model.EIL.weight[config.padIdx].copy_(zeros)
                model.ELI.weight[config.padIdx].copy_(zeros)

        epochLoss = epochLoss / float(numBatch)
        timeEpEnd   = time.time()
        logger.info("num_epoch: %d, elaspe: %.1f, loss: %.3f" % (epoch, timeEpEnd - timeEpStr, epochLoss))

        if (epoch + 1)% config.evalEpoch == 0:
            timeEvalStar = time.time()
            logger.info("start evaluation")

            #recall 5, 10, 20, 40...
            recall, ndcg = evalByBas(model, trans, config, device)
            logger.info(', '.join(str(e) for e in recall))
            logger.info(', '.join(str(e) for e in ndcg))

            timeEvalEnd = time.time()
            logger.info("Evaluation time:{}".format(timeEvalEnd - timeEvalStar))

            if not config.isTrain:
                torch.save(model, config.saveRoot+'_'+str(epoch))

    logger.info("\n")
    logger.info("\n")

def trainByUser(dataLoader, config, logger, device):

    if config.isTrain:
        numUsers = dataLoader.numTrain
        numItems = dataLoader.numItemsTrain
    else:
        numUsers = dataLoader.numTrainVal
        numItems = dataLoader.numItemsTest

    #preTrain
    outEmbsWeights = None
    numBatch = numUsers // config.batchSize + 1
    idxList  = [i for i in range(numUsers)]

    if config.model == 'Dream':
        model = Dream(config, numItems, device).to(device)
    elif config.model == 'SNBR':
        model = SNBR(config, numItems, device, outEmbsWeights).to(device)
    elif config.model == 'FREQ' or config.model == 'FREQP':
        model = FREQ(numItems, device).to(device)

    #open to more
    if config.opt == 'Adam':
        #condidate lr: 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2)
    elif config.opt == 'Adagrad':
        #condidate lr: 1e-2
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr, weight_decay=config.l2)

    for epoch in range(config.numIter):
        #permutate user list
        random.shuffle(idxList)
        timeEpStr = time.time()
        epochLoss = 0

        for batch in range(numBatch):
            start = config.batchSize * batch
            end   = min(numUsers, start + config.batchSize)

            batchList = idxList[start:end]
            samples, decays, his, target, offset, lenX, targetList = generateBatchSamples(dataLoader, batchList, config, isEval=0)

            samples = torch.from_numpy(samples).type(torch.LongTensor).to(device)
            decays  = torch.from_numpy(decays).type(torch.FloatTensor).to(device)
            his     = torch.from_numpy(his).type(torch.FloatTensor).to(device)
            target  = torch.from_numpy(target).type(torch.FloatTensor).to(device)
            offset  = torch.from_numpy(offset).type(torch.FloatTensor).to(device)
            lenX    = lenX.to(device)

            if config.model == 'Dream':
                tarArr, negArr = generateNegatives(targetList, numItems, config)
                tarLongTensor = torch.from_numpy(tarArr).type(torch.LongTensor).to(device)
                negLongTensor = torch.from_numpy(negArr).type(torch.LongTensor).to(device)
                scoresTar, scoresNeg = model.forward(samples, lenX, tarLongTensor, negLongTensor, isEval=0)
            elif config.model == 'SNBR':
                scores, _  = model.forward(samples, decays, offset, his, isEval=0)
            elif config.model == 'RNBR':
                scores  = model.forward(samples, offset, lenX, his, isEval=0)
            elif config.model == 'UNBR':
                scores  = model.forward(samples, decays, offset, lenX, his, isEval=0)
            elif config.model == 'FREQ' or config.model == 'FREQP':
                scores  = model.forward(his)

            #Dream and FPMC are trained using BPR
            if config.model == 'Dream' or config.model == 'FPMC':
                loss = BPR(scoresTar, scoresNeg)
            else:
                loss = Bernoulli(target, scores)

            epochLoss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config.model != 'FREQ' and config.model != 'FREQP':
                with torch.no_grad():
                    zeros = torch.zeros(config.dim).to(device)
                    model.itemEmb.weight[config.padIdx].copy_(zeros)

        epochLoss = epochLoss / float(numBatch)
        timeEpEnd   = time.time()
        logger.info("num_epoch: %d, elaspe: %.1f, loss: %.3f" % (epoch, timeEpEnd - timeEpStr, epochLoss))
        

        if (epoch + 1)% config.evalEpoch == 0:
            timeEvalStar = time.time()
            logger.info("start evaluation")

            #recall 5, 10, 20, 40...
            recall, ndcg = evalByUser(model, dataLoader, config, device, config.isTrain)
            logger.info(', '.join(str(e) for e in recall))
            logger.info(', '.join(str(e) for e in ndcg))

            timeEvalEnd = time.time()
            logger.info("Evaluation time:{}".format(timeEvalEnd - timeEvalStar))

            if not config.isTrain:
                torch.save(model, config.saveRoot+'_'+str(epoch))

    logger.info("\n")
    logger.info("\n")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='time_split')
    parser.add_argument('--dataset', type=str, default='TaFeng')
    
    parser.add_argument('--batchSize', type=int, default=100)
    parser.add_argument('--opt', type=str, default='Adagrad')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--l2', type=float, default=1e-3)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--decay', type=float, default=0.6)
    parser.add_argument('--testOrder', type=int, default=1)
    parser.add_argument('--numIter', type=int, default=200)
    parser.add_argument('--evalEpoch', type=int, default=5)
    parser.add_argument('--isTrain', type=int, default=1)
    parser.add_argument('--k', type=float, default=0.0)
    parser.add_argument('--preTrainBatchSize', type=int, default=4096)
    parser.add_argument('--preTrainEpochs', type=int, default=100)
    parser.add_argument('--isPreTrain', type=int, default=0)
    parser.add_argument('--abla', type=str, default="None")
    parser.add_argument('--retrival', type=int, default=0)
    parser.add_argument('--testEpoch', type=int, default=0)
    parser.add_argument('--store', type=int, default=0, help='1 for storing training data only')

    parser.add_argument('--model', type=str, default='SNBR')

    config = parser.parse_args()

    if config.mode == 'time_split':
        if config.isTrain:
            resultName = 'all_results_valid'
        else:
            resultName = 'all_results_test'
    elif config.mode == 'seq_split':
        if config.isTrain:
            resultName = 'all_results_valid_leave_one'
        else:
            resultName = 'all_results_test_leave_one'

    if config.abla != "None":
        logName = resultName+'/'+config.model+'_'+config.abla+'/'+config.dataset+'/'+config.model+'_'+config.opt+'_'+str(config.batchSize)+'_'+str(config.decay)+'_'+str(config.dim)+'_'+str(config.l2)+'_'+str(config.isPreTrain)+'_'+str(config.k)+'_'+str(config.testOrder)
    else:
        logName = resultName+'/'+config.model+'/'+config.dataset+'/'+config.model+'_'+config.opt+'_'+str(config.batchSize)+'_'+str(config.decay)+'_'+str(config.dim)+'_'+str(config.l2)+'_'+str(config.isPreTrain)+'_'+str(config.k)+'_'+str(config.testOrder)

    if config.store == 0:
        #avoid to stack results from different runs
        if os.path.exists(logName):
            os.remove(logName)

    logging.basicConfig(filename=logName, level=logging.DEBUG)
    ##logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    if config.mode == 'time_split':
        if not config.isTrain:
            config.saveRoot = 'models_test/'+config.model+'/'+config.dataset+'/'+config.model+'_'+config.opt+'_'+str(config.batchSize)+'_'+str(config.decay)+'_'+str(config.dim)+'_'+str(config.l2)+'_'+str(config.isPreTrain)+'_'+str(config.k)+'_'+str(config.testOrder)
    elif config.mode == 'seq_split':
        if not config.isTrain:
            config.saveRoot = 'models_test_leave_one/'+config.model+'/'+config.dataset+'/'+config.model+'_'+config.opt+'_'+str(config.batchSize)+'_'+str(config.decay)+'_'+str(config.dim)+'_'+str(config.l2)+'_'+str(config.isPreTrain)+'_'+str(config.k)+'_'+str(config.testOrder)

    if config.mode == 'time_split':
        from data import dataLoader
        dataset = dataLoader(config.dataset, config)
    elif config.mode == 'seq_split':
        from data_leave_one import dataLoader
        dataset = dataLoader(config.dataset, config)

    if config.store == 1:
        with open('SetsData_leave_one/train_user_list_'+config.dataset+'.pkl', 'wb') as f:
            pickle.dump(dataset.trainList, f)

        with open('SetsData_leave_one/valid_user_list_'+config.dataset+'_'+str(config.testOrder)+'.pkl', 'wb') as f:
            pickle.dump(dataset.validList, f)

        with open('SetsData_leave_one/train_valid_user_list_'+config.dataset+'.pkl', 'wb') as f:
            pickle.dump(dataset.trainValList, f)

        with open('SetsData_leave_one/test_user_list_'+config.dataset+'_'+str(config.testOrder)+'.pkl', 'wb') as f:
            pickle.dump(dataset.testList, f)
        print('store done')
        sys.exit()
        

    if config.isTrain:
        config.padIdx = dataset.numItemsTrain
    else:
        config.padIdx = dataset.numItemsTest

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    logger.info('start training')

    if config.retrival:
        model = torch.load(config.saveRoot+'_'+str(config.testEpoch))
        if config.model == 'FPMC':
            recall, ndcg = evalByBas(model, trans, config, device)
            userEmbeddings = model.EUI.cpu().weight.data.numpy().copy()
            np.save('embeddings/userEmb_'+config.dataset+'_FPMC', userEmbeddings)
            trans = transactions(dataset, config)
        else:
            recall, ndcg = evalByUser(model, dataset, config, device, config.isTrain)
            W = model.itemEmb.cpu().weight.data.numpy().copy()
            bias = model.out.cpu().bias.data.numpy().copy()
            np.savetxt('weights/'+config.model+'_'+config.dataset+'_'+str(config.testOrder)+'_bias.txt',bias,fmt='%.4f',delimiter=',')
            np.savetxt('weights/'+config.model+'_'+config.dataset+'_'+str(config.testOrder)+'_W.txt',W,fmt='%.4f',delimiter=',')

        logger.info(', '.join(str(e) for e in recall))
        logger.info(', '.join(str(e) for e in ndcg))

    else:
        if config.model == 'FPMC':
            trans = transactions(dataset, config)
            trainByBas(trans, config, logger, device)
        else:
            trainByUser(dataset, config, logger, device)

    logger.info('end')