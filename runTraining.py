import numpy as np
import torch
import torchvision
from torch import nn
import autoEnc
import pandas as pd
import argparse
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import copy
import pickle as pkl

def main(options):

    df = pd.read_csv(options.infile)
    df['pdTimestamp'] = [pd.Timestamp(ts) for ts in df['timestamp']]
    df['minutes'] = [float((pdts - df['pdTimestamp'][0]).to_numpy()/(60e9)) for pdts in df['pdTimestamp']]
    minsFail = df.loc[df['machine_status'] == 'BROKEN', 'minutes'].values
    mnsFail = np.zeros(0)
    for i in range(len(minsFail)):
        base = minsFail[i-1] if i>0 else 0.
        mnsFail = np.hstack((mnsFail, minsFail[i]*np.ones(int(minsFail[i])-int(base))))
    mnsFail = np.hstack((mnsFail, 999999.*np.ones(df.index.size-len(mnsFail))))
    df['minsNextFail'] = mnsFail
    df['minsToNextFail'] = df['minsNextFail'] - df['minutes']
    df.dropna(thresh=53, inplace=True)
    dfAll = copy.deepcopy(df)
    
    df_fails = df.loc[df['minsToNextFail']<60.,:]
    df_fails.reset_index(inplace=True)
    df.query('minsToNextFail>60', inplace=True)
    # df_check = df.loc[170001:, :]
    df.query("machine_status=='NORMAL'", inplace=True)
    
    sensNames = []
    for i in range(52):
        sensNames.append('sensor_{}'.format(str(i).zfill(2)))
    sensNames.remove('sensor_50')
    sensNames.remove('sensor_15')

    features = df.loc[:, sensNames].fillna(0.0).values
    trainData, evalData = train_test_split(features, shuffle=True, train_size=0.8)
    scaler = RobustScaler(quantile_range=(0.1,99.9)).fit(trainData)
    trainData = scaler.transform(trainData)
    evalData = scaler.transform(evalData)

    trainData = torch.utils.data.TensorDataset(torch.from_numpy(trainData))
    evlData = torch.utils.data.TensorDataset(torch.from_numpy(evalData))

    net = autoEnc.Autoencoder(epochs=100, batchSize=64, learningRate=1e-5)
    net.train(trainData, evalData)

    # testIn = scaler.transform(df.loc[1234, sensNames].fillna(0.0).values.reshape(1,-1))
    # testIn = torch.from_numpy(testIn).type(torch.FloatTensor)
    # testOut = net(testIn)
    # print('Input: {}'.format(testIn))
    # print('Output: {}'.format(testOut))
    # testLoss = net.criterion(testOut, testIn)
    # print('Test Loss: {:.4f}'.format(testLoss.data))

    testFailIn = scaler.transform(df_fails.loc[:, sensNames].fillna(0.0).values)
    testFailIn = torch.from_numpy(testFailIn).type(torch.FloatTensor)
    testFailOut = net(testFailIn)
    
    # print('Input Fail: {}'.format(testFailIn))
    # print('Output Fail: {}'.format(testFailOut))
    testFailLoss = nn.MSELoss(reduction='none')(testFailOut, testFailIn)
    # print(testFailLoss.mean(axis=-1))
    df_fails['netLoss'] = testFailLoss.mean(axis=-1).data.numpy()
    df_fails.to_hdf('./outputFails.hd5', key='df_fails', mode='w')
    # print('Test Loss Failure: {:.4f}'.format(testFailLoss.data))
    
    testCheckIn = scaler.transform(dfAll.loc[:, sensNames].fillna(0.0).values)
    testCheckIn = torch.from_numpy(testCheckIn).type(torch.FloatTensor)
    testCheckOut = net(testCheckIn)

    testCheckLoss = nn.MSELoss(reduction='none')(testCheckOut, testCheckIn)
    dfAll['netLoss'] = testCheckLoss.mean(axis=-1).data.numpy()
    dfAll.to_hdf('./allOutput.hd5', key='dfAll', mode='w')

    pkl.dump(net.trainingLossMean, open('./trainingLossMean.pkl', 'wb'))
    pkl.dump(net.validationLossMean, open('./validationLossMean.pkl', 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--cudaDevice', '-c', action='store', default='0', type=str)
    parser.add_argument(
        '--infile', '-i', action='store', type=str)
    # parser.add_argument(
    #     '--tmpDir', '-t', action='store', default='./', type=str)
    # parser.add_argument(
    #     '--saveFile', '-s', action='store', type=str)
    # parser.add_argument(
    #     '--config', action='store', type=int)
    options = parser.parse_args()
    main(options)
