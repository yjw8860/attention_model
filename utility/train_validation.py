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
        self.L_MODEL = SDPNetLocal(224, class_length).cuda()
        self.F_MODEL = SDPNetFusion(224, class_length).cuda()
        self.F_MODEL.zero_grad()
        self.G_OPTIMIZER = optim.Adam(self.G_MODEL.parameters(), lr=learning_rate)
        self.L_OPTIMIZER = optim.Adam(self.L_MODEL.parameters(), lr=learning_rate)
        self.F_OPTIMIZER = optim.Adam(self.F_MODEL.parameters(), lr=learning_rate)
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

    def returnLocalModelOutput(self, inputs, labels, weight_p, weight_n):
        self.L_OPTIMIZER.zero_grad()
        outputs, poolings = self.L_MODEL.forward(inputs)
        loss = self.CRITERION(labels, outputs, weight_p, weight_n)
        return outputs, loss, poolings

    def returnFusionModelOutput(self, concated_poolings, labels, weight_p, weight_n):
        self.F_OPTIMIZER.zero_grad()
        outputs = self.F_MODEL.forward(concated_poolings)
        loss = self.CRITERION(labels, outputs, weight_p, weight_n)
        return outputs, loss

    def updateModel(self, optimizer, loss):
        loss.backward()
        optimizer.step()

    def updateFModel(self, optimizer, loss):
        loss.backward(retain_graph=True)
        optimizer.step()

    def appendAvgLoss(self,avg_loss, loss):
        avg_loss.append(np.mean(loss))

    def appendAvgAcc(self, data_length, avg_accuracy, accuracy):
        avg_accuracy.append(accuracy / data_length)

    def saveBestModelWithAccuracy(self, accuracy, avg_acc, model_name='G'):
        if len(avg_acc) > 0:
            if max(avg_acc) < accuracy / self.VAL_EXECUTE.DATA_LENGTH:
                if model_name == 'G':
                    print(f'Validation G_accuracy is improved:{max(avg_acc)} to {accuracy / self.VAL_EXECUTE.DATA_LENGTH}')
                    path = os.path.join(self.SAVE.SAVE_FOLDER, 'g_model.pth')
                    torch.save(self.G_MODEL.state_dict(), path)
                elif model_name == 'L':
                    print(f'Validation L_accuracy is improved:{max(avg_acc)} to {accuracy / self.VAL_EXECUTE.DATA_LENGTH}')
                    path = os.path.join(self.SAVE.SAVE_FOLDER, 'l_model.pth')
                    torch.save(self.L_MODEL.state_dict(), path)
                else:
                    print(f'Validation F_accuracy is improved:{max(avg_acc)} to {accuracy / self.VAL_EXECUTE.DATA_LENGTH}')
                    path = os.path.join(self.SAVE.SAVE_FOLDER, 'f_model.pth')
                    torch.save(self.F_MODEL.state_dict(), path)
            else:
                if model_name == 'G':
                    print(f'Validation G_accuracy is not improved(G_accuracy:{max(avg_acc)})')
                elif model_name == 'L':
                    print(f'Validation L_accuracy is not improved(G_accuracy:{max(avg_acc)})')
                else:
                    print(f'Validation F_accuracy is not improved(G_accuracy:{max(avg_acc)})')

    def saveBestModelWithLoss(self, loss_list, avg_loss, model_name='G'):
        if len(avg_loss) > 0:
            if min(avg_loss) > np.mean(loss_list):
                if model_name == 'G':
                    print(f'Validation G_loss is improved:{min(avg_loss)} to {np.mean(loss_list)}')
                    path = os.path.join(self.SAVE.SAVE_FOLDER, 'g_model.pth')
                    torch.save(self.G_MODEL.state_dict(), path)
                elif model_name == 'L':
                    print(f'Validation L_loss is improved:{min(avg_loss)} to {np.mean(loss_list)}')
                    path = os.path.join(self.SAVE.SAVE_FOLDER, 'l_model.pth')
                    torch.save(self.L_MODEL.state_dict(), path)
                else:
                    print(f'Validation F_loss is improved:{min(avg_loss)} to {np.mean(loss_list)}')
                    path = os.path.join(self.SAVE.SAVE_FOLDER, 'f_model.pth')
                    torch.save(self.F_MODEL.state_dict(), path)
            else:
                if model_name == 'G':
                    print(f'Validation accuracy is not improved(G_loss:{np.mean(loss_list)})')
                elif model_name == 'L':
                    print(f'Validation accuracy is not improved(L_loss:{np.mean(loss_list)})')
                else:
                    print(f'Validation accuracy is not improved(F_loss:{np.mean(loss_list)})')


    def execute(self):
        for epoch in range(self.TOTAL_EPOCH):
            print(f'{epoch + 1} / {self.TOTAL_EPOCH}')
            print('-------------------TRAIN STARTED-------------------')
            g_loss_list, l_loss_list, f_loss_list = [], [], []
            g_accuracy, l_accuracy, f_accuracy = 0, 0, 0
            for i, data in enumerate(self.TRAIN_EXECUTE.DATA_LOADER, 0):
                inputs, labels, weight_p, weight_n = self.importData(data)

                outputs, g_poolings, heatmaps, loss= self.returnGlobalModelOutput(inputs, labels, weight_p, weight_n)
                self.updateModel(self.G_OPTIMIZER,loss)
                self.TRAIN_EXECUTE.appendLoss(i, g_loss_list, loss ,'G')
                g_accuracy = self.TRAIN_EXECUTE.returnAcc(outputs, labels, g_accuracy)

                outputs, loss, l_poolings = self.returnLocalModelOutput(heatmaps, labels, weight_p, weight_n)
                self.updateModel(self.L_OPTIMIZER,loss)
                self.TRAIN_EXECUTE.appendLoss(i, l_loss_list, loss, 'L')
                l_accuracy = self.TRAIN_EXECUTE.returnAcc(outputs, labels, l_accuracy)

                g_ = g_poolings.detach().cpu().numpy()
                g_ = torch.from_numpy(g_).float().cuda()
                l_ = l_poolings.detach().cpu().numpy()
                l_ = torch.from_numpy(l_).float().cuda()
                concated_poolings = torch.cat((g_, l_), dim=1)
                outputs, loss = self.returnFusionModelOutput(concated_poolings, labels, weight_p, weight_n)
                self.updateModel(self.F_OPTIMIZER,loss)
                self.TRAIN_EXECUTE.appendLoss(i, f_loss_list, loss, 'F')
                f_accuracy = self.TRAIN_EXECUTE.returnAcc(outputs, labels, f_accuracy)

            self.appendAvgLoss(self.TRAIN_EXECUTE.G_AVG_LOSS, g_loss_list)
            self.appendAvgLoss(self.TRAIN_EXECUTE.L_AVG_LOSS, l_loss_list)
            self.appendAvgLoss(self.TRAIN_EXECUTE.F_AVG_LOSS, f_loss_list)
            self.appendAvgAcc(self.TRAIN_EXECUTE.DATA_LENGTH, self.TRAIN_EXECUTE.G_AVG_ACC, g_accuracy)
            self.appendAvgAcc(self.TRAIN_EXECUTE.DATA_LENGTH, self.TRAIN_EXECUTE.L_AVG_ACC, l_accuracy)
            self.appendAvgAcc(self.TRAIN_EXECUTE.DATA_LENGTH, self.TRAIN_EXECUTE.F_AVG_ACC, f_accuracy)

            print('-------------------VALIDATION STARTED-------------------')
            g_loss_list, l_loss_list, f_loss_list = [], [], []
            g_accuracy, l_accuracy, f_accuracy = 0, 0, 0
            with torch.no_grad():
                for i, data in enumerate(self.VAL_EXECUTE.DATA_LOADER, 0):
                    inputs, labels, weight_p, weight_n = self.importData(data)

                    outputs, g_poolings, heatmaps, loss = self.returnGlobalModelOutput(inputs, labels, weight_p, weight_n)
                    self.VAL_EXECUTE.appendLoss(i, g_loss_list, loss, 'G')
                    g_accuracy = self.TRAIN_EXECUTE.returnAcc(outputs, labels, g_accuracy)

                    outputs, loss, l_poolings = self.returnLocalModelOutput(heatmaps, labels, weight_p, weight_n)
                    self.VAL_EXECUTE.appendLoss(i, l_loss_list, loss, 'L')
                    l_accuracy = self.VAL_EXECUTE.returnAcc(outputs, labels, l_accuracy)

                    g_ = g_poolings.detach().cpu().numpy()
                    g_ = torch.from_numpy(g_).float().cuda()
                    l_ = l_poolings.detach().cpu().numpy()
                    l_ = torch.from_numpy(l_).float().cuda()
                    concated_poolings = torch.cat((g_, l_), dim=1)
                    outputs, loss = self.returnFusionModelOutput(concated_poolings, labels, weight_p, weight_n)
                    self.VAL_EXECUTE.appendLoss(i, f_loss_list, loss, 'F')
                    f_accuracy = self.VAL_EXECUTE.returnAcc(outputs, labels, f_accuracy)

            self.saveBestModelWithLoss(g_loss_list, self.VAL_EXECUTE.G_AVG_LOSS, model_name='G')
            self.saveBestModelWithLoss(l_loss_list, self.VAL_EXECUTE.L_AVG_LOSS, model_name='L')
            self.saveBestModelWithLoss(f_loss_list, self.VAL_EXECUTE.F_AVG_LOSS, model_name='F')
            self.appendAvgLoss(self.VAL_EXECUTE.G_AVG_LOSS, g_loss_list)
            self.appendAvgLoss(self.VAL_EXECUTE.L_AVG_LOSS, l_loss_list)
            self.appendAvgLoss(self.VAL_EXECUTE.F_AVG_LOSS, f_loss_list)
            self.appendAvgAcc(self.VAL_EXECUTE.DATA_LENGTH, self.VAL_EXECUTE.G_AVG_ACC, g_accuracy)
            self.appendAvgAcc(self.VAL_EXECUTE.DATA_LENGTH, self.VAL_EXECUTE.L_AVG_ACC, l_accuracy)
            self.appendAvgAcc(self.VAL_EXECUTE.DATA_LENGTH, self.VAL_EXECUTE.F_AVG_ACC, f_accuracy)

            print(f'G_TRAIN LOSS:{round(self.TRAIN_EXECUTE.G_AVG_LOSS[epoch], 5)}|G_TRAIN ACCURACY:{round(self.TRAIN_EXECUTE.G_AVG_ACC[epoch], 5)}|G_VALIDATION LOSS:{round(self.VAL_EXECUTE.G_AVG_LOSS[epoch], 5)}|G_VALIDATION ACCURACY:{round(self.VAL_EXECUTE.G_AVG_ACC[epoch], 5)}')
            print(f'L_TRAIN LOSS:{round(self.TRAIN_EXECUTE.L_AVG_LOSS[epoch], 5)}|L_TRAIN ACCURACY:{round(self.TRAIN_EXECUTE.L_AVG_ACC[epoch], 5)}|L_VALIDATION LOSS:{round(self.VAL_EXECUTE.L_AVG_LOSS[epoch], 5)}|L_VALIDATION ACCURACY:{round(self.VAL_EXECUTE.L_AVG_ACC[epoch], 5)}')
            print(f'F_TRAIN LOSS:{round(self.TRAIN_EXECUTE.F_AVG_LOSS[epoch], 5)}|F_TRAIN ACCURACY:{round(self.TRAIN_EXECUTE.F_AVG_ACC[epoch], 5)}|F_VALIDATION LOSS:{round(self.VAL_EXECUTE.F_AVG_LOSS[epoch], 5)}|F_VALIDATION ACCURACY:{round(self.VAL_EXECUTE.F_AVG_ACC[epoch], 5)}')
        self.SAVE.DataFrame()
        self.SAVE.LossImg()
        self.SAVE.AccImg()