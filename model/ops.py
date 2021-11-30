import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

BATCH_SIZE = 24
EPOCHS = 100
LR = 0.0001
NUM_WORKER = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class depthwise_separable_conv(nn.Module):
    def __init__(self, f_size, n_class):
        super(depthwise_separable_conv, self).__init__()
        self.f_size = f_size
        self.n_class = n_class
        self.standard = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm_0 = nn.BatchNorm2d(64)
        self.depthwise_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.batchnorm_1 = nn.BatchNorm2d(64)
        self.pointwise_1 = nn.Conv2d(64, 128, kernel_size=1)
        self.depthwise_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128)
        self.batchnorm_2 = nn.BatchNorm2d(128)
        self.pointwise_2 = nn.Conv2d(128, 256, kernel_size=1)
        self.depthwise_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256)
        self.batchnorm_3 = nn.BatchNorm2d(256)
        self.pointwise_3 = nn.Conv2d(256, 512, kernel_size=1)
        self.depthwise_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512)
        self.batchnorm_4 = nn.BatchNorm2d(512)
        self.averagepool = nn.AvgPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512*f_size*self.f_size, self.n_class)

    def block_0(self,x):
        return self.batchnorm_0(F.relu(self.standard(x)))

    def block_1(self, x):
        for i in range(4):
            for i in range(2):
                x = self.batchnorm_1(F.relu(self.depthwise_1(x)))
        return x

    def bridge_1(self, x):
        x = self.batchnorm_1(F.relu(self.depthwise_1(x)))
        x = self.batchnorm_2(F.relu(self.pointwise_1(x)))
        return x

    def block_2(self, x):
        for i in range(5):
            for i in range(2):
                x = self.batchnorm_2(F.relu(self.depthwise_2(x)))
        return x

    def bridge_2(self, x):
        x = self.batchnorm_2(F.relu(self.depthwise_2(x)))
        x = self.batchnorm_3(F.relu(self.pointwise_2(x)))
        return x

    def block_3(self, x):
        for i in range(6):
            for i in range(2):
                x = self.batchnorm_3(F.relu(self.depthwise_3(x)))
        return x

    def bridge_3(self, x):
        x = self.batchnorm_3(F.relu(self.depthwise_3(x)))
        x = self.batchnorm_4(F.relu(self.pointwise_3(x)))
        return x

    def block_4(self, x):
        for i in range(5):
            for i in range(2):
                x = self.batchnorm_4(F.relu(self.depthwise_4(x)))
        return x

    def forward(self, x):
        b_0 = self.block_0(x)
        out = self.block_1(b_0)
        out = torch.add(b_0, out)
        b_1 = self.bridge_1(out)
        out = self.block_2(b_1)
        out = torch.add(b_1, out)
        b_3 = self.bridge_2(out)
        out = self.block_3(b_3)
        out = torch.add(b_3, out)
        b_4 = self.bridge_3(out)
        out = self.block_4(b_4)
        out = torch.add(b_4, out)
        out = self.averagepool(out)
        out = out.view(-1, 512*self.f_size*self.f_size)
        out = self.dropout(out)
        out = self.fc(out)
        return out


SDPN = depthwise_separable_conv(16, 10)
SDPN.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(SDPN.parameters(), lr=LR)

train_avg_loss = []
test_avg_loss = []
train_accuracy_list = []
test_accuracy_list = []

for epoch in range(EPOCHS):   # 데이터셋을 수차례 반복합니다.
    train_loss_list = []
    test_loss_list = []
    total = 0
    correct = 0
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader)):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = SDPN(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        train_loss_list.append(loss.item())
        running_loss += loss.item()
        if i % 50 == 49:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
    train_accuracy = 100 * correct / total
    train_accuracy_list.append(train_accuracy)
    train_avg_loss.append(round(np.mean(train_loss_list), 3))

    print(f'START VALIDATION EPOCH:{epoch}')
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = SDPN(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss_list.append(criterion(outputs, labels).item())
        test_accuracy = 100 * correct / total
        test_accuracy_list.append(test_accuracy)
        test_avg_loss.append(round(np.mean(test_loss_list), 3))
    print(f'EPOCH:{epoch}/{EPOCHS}'
          f'|TRAIN LOSS(AVG):{round(np.mean(train_loss_list), 3)}'
          f'|TRAIN ACCURACY:{train_accuracy}'
          f'|VALIDATION LOSS(AVG):{round(np.mean(test_loss_list), 3)}'
          f'|VALIDATION ACCURACY:{test_accuracy}')
print('Finished Training')
torch.save(SDPN, 'redness.pth')

df = pd.DataFrame({'epoch': list(range(EPOCHS)),
                   'train_loss': train_avg_loss,
                   'train_accuracy': train_accuracy_list,
                   'test_loss':test_avg_loss,
                   'test_accuracy':test_accuracy_list},
                  columns=['epoch', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy'])
df_save_path = './redness_loss.csv'
df.to_csv(df_save_path, index=False, encoding='euc-kr')