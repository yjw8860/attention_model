import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

def json_load(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data

def json_save(json_path, data):
    f = open(json_path, 'w', encoding='utf-8')
    json.dumps(data, f, ensure_ascii=False)

class returnLossAndAcc:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.DATASET = dataset
        self.DATA_LENGTH = len(self.DATASET.df.iloc[:,0])
        self.DATA_LOADER = DataLoader(self.DATASET, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        self.G_AVG_LOSS = []
        self.G_AVG_ACC = []
        self.L_AVG_LOSS = []
        self.L_AVG_ACC = []
        self.F_AVG_LOSS = []
        self.F_AVG_ACC = []

    def appendLoss(self,idx, loss_list, current_loss, name):
        loss_list.append(current_loss.item())
        print(f'iteration:{idx}/{len(self.DATA_LOADER)}|{name}_loss:{current_loss.item()}')

    def returnAcc(self, outputs, labels, accuracy):
        outputs = torch.round(outputs)
        outputs = outputs.detach().cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        for label, output in zip(labels, outputs):
            if label == output:
                accuracy += 1.
        return accuracy

class Save:
    def __init__(self, save_name, total_epoch, train_, val_):
        self.SAVE_FOLDER = f'./{save_name}'
        os.makedirs(self.SAVE_FOLDER, exist_ok=True)
        self.TOTAL_EPOCH = total_epoch
        self.TRAIN = train_
        self.VAL = val_

    def DataFrame(self):
        path_g = os.path.join(self.SAVE_FOLDER, 'g_loss_acc.csv')
        # path_l = os.path.join(self.SAVE_FOLDER, 'l_loss_acc.csv')
        # path_f = os.path.join(self.SAVE_FOLDER, 'f_loss_acc.csv')
        df_g = pd.DataFrame({'epoch': list(range(self.TOTAL_EPOCH)), 'train_loss': self.TRAIN.G_AVG_LOSS, 'train_acc': self.TRAIN.G_AVG_ACC,
                   'val_loss': self.VAL.G_AVG_LOSS, 'val_acc': self.VAL.G_AVG_LOSS},
                  columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        # df_l = pd.DataFrame({'epoch': list(range(self.TOTAL_EPOCH)), 'train_loss': self.TRAIN.L_AVG_LOSS, 'train_acc': self.TRAIN.L_AVG_ACC,
        #            'val_loss': self.VAL.L_AVG_LOSS, 'val_acc': self.VAL.L_AVG_LOSS},
        #           columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        # df_f = pd.DataFrame({'epoch': list(range(self.TOTAL_EPOCH)), 'train_loss': self.TRAIN.F_AVG_LOSS, 'train_acc': self.TRAIN.F_AVG_ACC,
        #            'val_loss': self.VAL.F_AVG_LOSS, 'val_acc': self.VAL.F_AVG_LOSS},
        #           columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        df_g.to_csv(path_g, index=False, encoding='euc-kr')
        # df_l.to_csv(path_l, index=False, encoding='euc-kr')
        # df_f.to_csv(path_f, index=False, encoding='euc-kr')

    def LossImg(self):
        path_g = os.path.join(self.SAVE_FOLDER, 'g_loss.png')
        # path_l = os.path.join(self.SAVE_FOLDER, 'l_loss.png')
        # path_f = os.path.join(self.SAVE_FOLDER, 'f_loss.png')
        plt.plot(list(range(len(self.TRAIN.G_AVG_LOSS))), self.TRAIN.G_AVG_LOSS, 'b', label='Training Loss')
        plt.plot(list(range(len(self.VAL.G_AVG_LOSS))), self.VAL.G_AVG_LOSS, 'r', label='Validation Loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(path_g)
        plt.cla()

        # plt.plot(list(range(self.TOTAL_EPOCH)), self.TRAIN.L_AVG_LOSS, 'b', label='Training Loss')
        # plt.plot(list(range(self.TOTAL_EPOCH)), self.VAL.L_AVG_LOSS, 'r', label='Validation Loss')
        # plt.title('Training and validation loss')
        # plt.legend()
        # plt.savefig(path_l)
        # plt.cla()
        #
        # plt.plot(list(range(self.TOTAL_EPOCH)), self.TRAIN.F_AVG_LOSS, 'b', label='Training Loss')
        # plt.plot(list(range(self.TOTAL_EPOCH)), self.VAL.F_AVG_LOSS, 'r', label='Validation Loss')
        # plt.title('Training and validation loss')
        # plt.legend()
        # plt.savefig(path_f)
        # plt.cla()

    def AccImg(self):
        path_g = os.path.join(self.SAVE_FOLDER, 'g_accuracy.png')
        # path_l = os.path.join(self.SAVE_FOLDER, 'l_accuracy.png')
        # path_f = os.path.join(self.SAVE_FOLDER, 'f_accuracy.png')
        plt.plot(list(range(len(self.TRAIN.G_AVG_ACC))), self.TRAIN.G_AVG_ACC, 'b', label='Training Accuracy')
        plt.plot(list(range(len(self.VAL.G_AVG_ACC))), self.VAL.G_AVG_ACC, 'r', label='Validation Accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(path_g)
        plt.cla()
        #
        # plt.plot(list(range(self.TOTAL_EPOCH)), self.TRAIN.L_AVG_ACC, 'b', label='Training Accuracy')
        # plt.plot(list(range(self.TOTAL_EPOCH)), self.VAL.L_AVG_ACC, 'r', label='Validation Accuracy')
        # plt.title('Training and validation accuracy')
        # plt.legend()
        # plt.savefig(path_l)
        # plt.cla()
        #
        # plt.plot(list(range(self.TOTAL_EPOCH)), self.TRAIN.F_AVG_ACC, 'b', label='Training Accuracy')
        # plt.plot(list(range(self.TOTAL_EPOCH)), self.VAL.F_AVG_ACC, 'r', label='Validation Accuracy')
        # plt.title('Training and validation accuracy')
        # plt.legend()
        # plt.savefig(path_f)
        # plt.cla()