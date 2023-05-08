import json
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import math
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, precision_recall_fscore_support, \
    roc_curve, balanced_accuracy_score
from sklearn.utils import shuffle
from torch.utils.data.sampler import  WeightedRandomSampler
from torch import argmax
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import argparse
from model.lenet import *
from model.resnet import *
from utils_image import *


# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

onehot_encoder = OneHotEncoder(sparse=False)

# Define super-parameter
parser = argparse.ArgumentParser(description='Attack_PI')
parser.add_argument('--dataset1', type=str, default='utkface', help="dataset1")
parser.add_argument('--dataset2', type=str, default='utkface', help="dataset2")
parser.add_argument('--model', type=str, default='lenet', help="model")
parser.add_argument('--level', default=1, type=int, help='level')
parser.add_argument('--acti', type=str, default='leakyrelu', help="acti")
parser.add_argument('--attack_label', type=int, default='0')
parser.add_argument('--attributes', type=str, default="race_gender", help="For attrinf, two attributes should be in format x_y e.g. race_gender")
parser.add_argument('--lr', default=1e-4, type=float, help='lr')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--attack_batch_size', default=128, type=int, help='attack_batch_size')
parser.add_argument('--epochs', default=50, type=int, help='epochs')
parser.add_argument('--num_cutlayer', default=1000, type=int, help='num_cutlayer')
parser.add_argument('--noise_scale', default=0, type=float, help='noise_scale')
parser.add_argument('--number_client', default=2, type=int, help='number_client')
parser.add_argument('--num_shadow', default=2, type=int, help='num_shadow')
parser.add_argument('--level1', default=1, type=int, help='level1')
parser.add_argument('--level2', default=1, type=int, help='level2')

args = parser.parse_args()

dataset1 = args.dataset1
dataset2 = args.dataset2
model = args.model
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
acti = args.acti
attributes = args.attributes
attack_label = args.attack_label
num_cutlayer = args.num_cutlayer
noise_scale = args.noise_scale
level1 =args.level1
level2 =args.level2
number_client = args.number_client
num_shadow = args.num_shadow
attack_batch_size = args.attack_batch_size


out_dim = 2


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


class LinearNet_mul(nn.Module):
    def __init__(self, in_dim=64, n_hidden_1=500, n_hidden_2=128, out_dim=2):
        super(LinearNet_mul, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),
                                    #nn.Dropout(0.5), # drop 50% of the neuron to avoid over-fitting
                                    nn.LeakyReLU()
                                    )
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    #nn.Dropout(0.5),  # drop 50% of the neuron to avoid over-fitting
                                    nn.LeakyReLU()
                                    )
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer3_out



def train(dataloader, model):
    global train_acc_sum 
    global train_loss_sum

    size = train_size
    model.train()
    correct = 0
    for batch, data in enumerate(dataloader):
        X = data[:, :data_dim - 1].to(device)
        X = X.to(torch.float32)
        y = ( (data[:, -1]+1)%2).to(device).long()

        pred = model(X)
        loss1 = loss_func(pred, y)
        # l2 = l2_regularization(model, 10)
        loss = loss1
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        if batch == len(train_iter)-1:
            correct_train = correct / ((batch + 1) * len(X))
            loss, current = loss.item(), (batch +1) * len(X)
            f2.write('loss:' + str(loss) +  '\n' +' Accuracy' + str(100 * correct_train) + '\n')

            train_acc_sum.append(100 * correct_train)
            train_loss_sum.append(loss)


def test(dataloader, model):
    global test_acc_sum 
    global test_loss_sum

    n_classes = out_dim
    size = test_size
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    TP, FP, TN, FN = 0, 0, 0, 0
    ypred = []
    ytrue = []
    y_pred = []
    y_true = []

    for data in dataloader:
        X = data[:, :data_dim - 1].to(device)
        X = X.to(torch.float32)
        y = data[:, -1].to(device).long()
        pred = model(X)
        test_loss += loss_func(pred, y).item()
        correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

        ypred.extend(np.array(pred.argmax(1).cpu(), dtype=int))
        ytrue.extend(np.array(y.cpu(), dtype=int))

 
        y_one_hot = torch.randint(1, (attack_batch_size, n_classes)).to(device).scatter_(1, y.view(-1, 1), 1)
        y_true.extend(y_one_hot.tolist())
        y_pred.extend(pred.softmax(dim=-1).tolist())

    cm = confusion_matrix(ytrue, ypred, labels=range(n_classes))
    cm = cm.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    print(f'TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}')

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    test_acc_sum.append(100 * correct)
    test_loss_sum.append(test_loss)


    acc = (TP + TN) / (TP + FP + TN + FN)
    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    F_meature = 2.0 * precision * recall / (precision + recall)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (FP + TN)

    acc1=accuracy_score(ytrue, ypred)
    acc1_w=balanced_accuracy_score(ytrue, ypred)

    f1=f1_score(ytrue,ypred,average='macro')
    precision1=precision_score(ytrue,ypred,average='macro')
    recall1=recall_score(ytrue,ypred,average='macro')

    f1_w=f1_score(ytrue,ypred,average='weighted')
    precision1_w=precision_score(ytrue,ypred,average='weighted')
    recall1_w=recall_score(ytrue,ypred,average='weighted')

    # auc
    auc_score = roc_auc_score(y_true, y_pred, multi_class='ovr')

    return acc, precision, recall, F_meature, TPR, FPR, TNR, TP, FP, TN, FN, auc_score, acc1, acc1_w, f1, f1_w, precision1, precision1_w, recall1,recall1_w
   


# Test_Client Side Program
def shadow_client(dataloader, client_model_1, client_model_2):
    client_model_1.eval()
    client_model_2.eval()
    correct = 0
    size = len(dataloader.dataset)

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(X)
        Y_1 = target[0].to(device)
        Y_2 = target[1].view(-1, 1).to(device)

        # client1--train and update
        fx1 = client_model_1(X_1)
        fx2 = client_model_2(X_2)

        # for property inference
        n1 = torch.cat([fx1, Y_2], dim=1)
        n1 = n1.cpu().detach().numpy()
        writer_shadow_1_disjoint.writerows(n1)

if dataset2 == 'utkface':
    attributes = 'race_gender'

if dataset2 == 'celeba':
    attributes = "attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr"


# Define rocords
save_path = f'Results_PI/{dataset1}-{dataset2}/{model}/level{level1}/client{number_client}/n{noise_scale}/shadow{num_shadow}/c{num_cutlayer}/client{number_client}/n{noise_scale}/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

filename = f'Results_PI/{dataset1}-{dataset2}/{model}/level{level1}/client{number_client}/n{noise_scale}/shadow{num_shadow}/c{num_cutlayer}/client{number_client}/n{noise_scale}/c{num_cutlayer}_{attack_label}.txt'
acc1 = ['epochs', 'acc', 'precision', 'recall', 'f1score', 'TPR', 'FPR', 'TNR', 'TP', 'FP', 'TN', 'FN', 'AUC']
f2 = open(filename, 'w')

# record shadow data
### Load data
root_path = '.'
data_path = os.path.join(root_path, '../data').replace('\\', '/')
train_data_1, test_data_1, shadow_data_1, num_classes1_1, num_classes2_1, channel_1, hideen_1 = load_data(args.dataset1, args.attack_label, args.attributes, data_path,
                                                                                   attack_batch_size)
train_data_2, test_data_2, shadow_data_2, num_classes1_2, num_classes2_2, channel_2, hideen_2 = load_data(args.dataset2, args.attack_label, args.attributes, data_path,
                                                                               attack_batch_size)  

save_path_train = f'Results/{dataset1}/{model}/level{level1}/client{number_client}/n{noise_scale}/'
csvFile_1 = open(os.path.join(save_path_train, f'VFL_client1_c{num_cutlayer}_shadow_{dataset2}.csv'), 'w+')
writer_shadow_1_disjoint = csv.writer(csvFile_1)

### Define model
if model == 'resnet':
    client_model_1 = ResNet18(level=level1, hideen1=hideen_1, num_classes=num_cutlayer).to(device)
    client_model_2 = ResNet18(level=level1, hideen1=hideen_1, num_classes=num_cutlayer).to(device)
    server_model = Server_ResNet(hideen2=num_cutlayer * 2, hideen3=256, hideen4=128, hideen5=64,
                                 num_classes=num_classes1_1).to(device)
    
    save_path1 = f'Results/{dataset1}/{model}/level{level1}/client{number_client}/n{noise_scale}/client1_c{num_cutlayer}.pth'
    save_path2 = f'Results/{dataset1}/{model}/level{level1}/client{number_client}/n{noise_scale}/client2_c{num_cutlayer}.pth'
    save_path3 = f'Results/{dataset1}/{model}/level{level1}/client{number_client}/n{noise_scale}/server_c{num_cutlayer}.pth'
    client_model_1 = torch.load(save_path1)
    client_model_2 = torch.load(save_path2)
    server_model = torch.load(save_path3)

# record shadow data
shadow_client(shadow_data_2, client_model_1, client_model_2)


# Define data
data_train = pd.read_csv(f'Results/{dataset1}/{model}/level{level1}/client{number_client}/n{noise_scale}/VFL_client1_c{num_cutlayer}_shadow_{dataset2}.csv', sep=',', header=None)
data_train =np.array(data_train)
train_set=torch.tensor(data_train[:num_shadow, ], dtype=float)


data_test = pd.read_csv(f'Results/{dataset1}/{model}/level{level1}/client{number_client}/n{noise_scale}/VFL_client1_c{num_cutlayer}.csv', sep=',', header=None)
data_test = np.array(data_test)
test_set=torch.tensor(data_test,dtype=float)



# Pre-processing data
data_dim = len(data_train[-1, :])


# Sample data
train_iter = Data.DataLoader(
                    dataset=train_set, 
                    batch_size=attack_batch_size,  
                    drop_last=True,)

test_iter = Data.DataLoader(
                    dataset=test_set, 
                    batch_size=attack_batch_size,  
                    drop_last=True,)

train_size = len(train_iter)*attack_batch_size
test_size = len(test_iter)*attack_batch_size


# Define model
Attack_model = LinearNet_mul(in_dim=num_cutlayer, n_hidden_1=500, n_hidden_2=128, out_dim=out_dim).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Attack_model.parameters(), lr=lr)


train_acc_sum = []
train_loss_sum = []
test_acc_sum = []
test_loss_sum = []

for t in range(epochs):
    print(f"Epoch {t + 1}")
    train(train_iter, Attack_model)

    acc, precision, recall, fscore,    TPR, FPR, TNR, TP, FP, TN, FN, auc_score, acc1, acc1_w, f1, f1_w, precision1, precision1_w, recall1, recall1_w = test(test_iter, Attack_model)
    metrics_score = {'Epochs': t,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
            'TPR': TPR,
            'FPR': FPR,
            'TNR': TNR,
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
            'AUC': auc_score,
            'ACC':acc1,
            'ACC_w':acc1_w,
            'F1':f1,
            'F1_w':f1_w,
            'Precision1':precision1,
            'Precision1_w':precision1_w,
            'recall1':recall1_w,
            'recall1_w':recall1_w, }
    for key in metrics_score:
        f2.write('\n')
        f2.writelines('"' + str(key) + '": ' + str(metrics_score[key]))
    f2.write('\n')
    f2.write('\n')
    print(metrics_score)
f2.close()
print("Done!")


