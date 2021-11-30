from __future__ import print_function, division
import os

import torch.optim as optim
from model.sdpn import *
from torchsummary import summary

def WBCEloss(labels, outputs, weight_p, weight_n):
    loss = 0.0
    eps = 1e-12
    if not labels.size() == outputs.size():
        print('labels and outputs must have the same size')
    else:
        N = labels.size(0)
        for label, output, w_p, w_n in zip(labels, outputs, weight_p, weight_n):
            temp = -(w_p * (label * torch.log(output + eps)) + (w_n*((1.0-label)*torch.log(1.0-output + eps))))
            loss += temp.sum()
        loss = loss/N
    return loss

class TrainAndValidation:
    def __init__(self, save, train_, val_, class_length, learning_rate, total_epoch):
        self.TOTAL_EPOCH = total_epoch
        self.TRAIN_EXECUTE = train_
        self.VAL_EXECUTE = val_
        self.G_MODEL = SDPNetGlobal(224, class_length).cuda()
        self.G_OPTIMIZER = optim.Adam(self.G_MODEL.parameters(), lr=learning_rate)
        self.CRITERION = WBCEloss
        self.SAVE = save


    def importData(self, data):
        inputs, labels, weight_p, weight_n = data['image'], data['label'], data['weight_p'], data['weight_n']
        inputs = torch.where(torch.isnan(inputs), torch.zeros_like(inputs), inputs)
        inputs = inputs.type(torch.cuda.FloatTensor)
        labels = labels.type(torch.cuda.FloatTensor)
        weight_p = weight_p.type(torch.cuda.FloatTensor)
        weight_n = weight_n.type(torch.cuda.FloatTensor)
        inputs, labels, weight_p, weight_n = inputs.cuda(), labels.cuda(), weight_p.cuda(), weight_n.cuda()

        return inputs, labels, weight_p, weight_n

    def returnGlobalModelOutput(self, inputs, labels, weight_p, weight_n):
        self.G_OPTIMIZER.zero_grad()
        outputs, poolings, heatmaps = self.G_MODEL.gOutput(inputs)
        loss = self.CRITERION(labels, outputs, weight_p, weight_n)
        return outputs, poolings, heatmaps, loss


    def updateModel(self, optimizer, loss):
        loss.backward()
        optimizer.step()

    def appendAvgLoss(self,avg_loss, loss):
        avg_loss.append(np.mean(loss))

    def appendAvgAcc(self, data_length, avg_accuracy, accuracy):
        avg_accuracy.append(accuracy / data_length)

    def saveBestModelWithLoss(self, loss_list, avg_loss, model_name='G'):
        if len(avg_loss) > 0:
            if min(avg_loss) > np.mean(loss_list):
                print(f'Validation G_loss is improved:{min(avg_loss)} to {np.mean(loss_list)}')
                path = os.path.join(self.SAVE.SAVE_FOLDER, 'g_model.pth')
                torch.save(self.G_MODEL.state_dict(), path)
            else:
                print(f'Validation accuracy is not improved(G_loss:{np.mean(loss_list)})')

    def execute(self):
        for epoch in range(self.TOTAL_EPOCH):
            print(f'{epoch + 1} / {self.TOTAL_EPOCH}')
            print('-------------------TRAIN STARTED-------------------')
            g_loss_list = []
            g_accuracy = 0
            for i, data in enumerate(self.TRAIN_EXECUTE.DATA_LOADER, 0):
                inputs, labels, weight_p, weight_n = self.importData(data)
                outputs, g_poolings, heatmaps, loss= self.returnGlobalModelOutput(inputs, labels, weight_p, weight_n)
                self.updateModel(self.G_OPTIMIZER,loss)
                self.TRAIN_EXECUTE.appendLoss(i, g_loss_list, loss ,'G')
                g_accuracy = self.TRAIN_EXECUTE.returnAcc(outputs, labels, g_accuracy)


            self.appendAvgLoss(self.TRAIN_EXECUTE.G_AVG_LOSS, g_loss_list)
            self.appendAvgAcc(self.TRAIN_EXECUTE.DATA_LENGTH, self.TRAIN_EXECUTE.G_AVG_ACC, g_accuracy)

            print('-------------------VALIDATION STARTED-------------------')
            g_loss_list = []
            g_accuracy = 0
            with torch.no_grad():
                for i, data in enumerate(self.VAL_EXECUTE.DATA_LOADER, 0):
                    inputs, labels, weight_p, weight_n = self.importData(data)

                    outputs, g_poolings, heatmaps, loss = self.returnGlobalModelOutput(inputs, labels, weight_p, weight_n)
                    self.VAL_EXECUTE.appendLoss(i, g_loss_list, loss, 'G')
                    g_accuracy = self.TRAIN_EXECUTE.returnAcc(outputs, labels, g_accuracy)

            self.saveBestModelWithLoss(g_loss_list, self.VAL_EXECUTE.G_AVG_LOSS, model_name='G')
            self.appendAvgLoss(self.VAL_EXECUTE.G_AVG_LOSS, g_loss_list)
            self.appendAvgAcc(self.VAL_EXECUTE.DATA_LENGTH, self.VAL_EXECUTE.G_AVG_ACC, g_accuracy)

            print(f'G_TRAIN LOSS:{round(self.TRAIN_EXECUTE.G_AVG_LOSS[epoch], 5)}|G_TRAIN ACCURACY:{round(self.TRAIN_EXECUTE.G_AVG_ACC[epoch], 5)}|G_VALIDATION LOSS:{round(self.VAL_EXECUTE.G_AVG_LOSS[epoch], 5)}|G_VALIDATION ACCURACY:{round(self.VAL_EXECUTE.G_AVG_ACC[epoch], 5)}')
            if epoch > 0:
                self.SAVE.LossImg()
                self.SAVE.AccImg()
        self.SAVE.DataFrame()