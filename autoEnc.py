import numpy as np
import torch
import torchvision
from torch import nn


class Autoencoder(nn.Module):

    def __init__(self, epochs=100, batchSize=128, learningRate=1e-3, device=torch.device('cpu')):
        super(Autoencoder, self).__init__()
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(50, 26),
            nn.ReLU(True),
            nn.Linear(26,13),
            nn.ReLU(True),
            nn.Linear(13,5)
        )

        self.decoder = nn.Sequential(
            nn.Linear(5,13),
            nn.ReLU(True),
            nn.Linear(13,26),
            nn.ReLU(True),
            nn.Linear(26,50),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train(self, X, evalSet):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

        self.to(self.device)

        dataLoader = torch.utils.data.DataLoader(dataset=X, batch_size=self.batchSize, shuffle=True)
        evalLoader = torch.utils.data.DataLoader(dataset=evalSet, batch_size=self.batchSize, shuffle=True)

        self.trainingLossMean = []
        self.validationLossMean = []
        
        for epoch in range(self.epochs):
            trainingLoss = []
            validationLoss = []
            for i, data in enumerate(dataLoader):

                # data, _ = data
                data = data[0]
                data = data.type(torch.FloatTensor)
                data = data.to(self.device)

                output = self(data)

                loss = self.criterion(output, data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                trainingLoss.append(loss.data.numpy())

                if i % 100 == 0:

                    for evalD in evalLoader:

                        evalD = evalD[0]
                        evalD = evalD.type(torch.FloatTensor)
                        evalD = evalD.to(self.device)
                        valOut = self(evalD)
                        evalLoss = self.criterion(valOut, evalD)

                        validationLoss.append(evalLoss.data.numpy())

                
            self.trainingLossMean.append([epoch, np.mean(trainingLoss)])
            self.validationLossMean.append([epoch, np.mean(validationLoss)])
            
            print('epoch {}/{}, loss: {:.4f}'.format(epoch+1, self.epochs, loss.data))

        self.trainingLossMean = np.array(self.trainingLossMean)
        self.validationLossMean = np.array(self.validationLossMean)
