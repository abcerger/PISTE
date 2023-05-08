import json
import pandas as pd
import torch
import torch.nn as nn
import numpy as  np
import math
import time
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, precision_recall_fscore_support, \
    roc_curve, balanced_accuracy_score
from sklearn.utils import shuffle
from torch import argmax
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import argparse
import os

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc





# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("using {} device.".format(device))
onehot_encoder = OneHotEncoder(sparse=False)
# Define super-parameter
parser = argparse.ArgumentParser(description='Attack_VFL1')
parser.add_argument('--dataset', type=str, default='census', help="dataset") # [bank_marketing, credit]
parser.add_argument('--acti', type=str, default='leakyrelu_2', help="activate")  # [leakrelu, non]
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--client', default='VFL_client1', type=str, help='client')
parser.add_argument('--epochs_attack', default=200, type=int, help='epochs')
parser.add_argument('--epochs', default=50, type=int, help='epochs')
parser.add_argument('--end_shadow', default=1, type=int, help='end_shadow')
parser.add_argument('--num_data', default=7936, type=int, help='num_data')
parser.add_argument('--attack_label', default='y5', type=str, help='num_data')
parser.add_argument('--out_dim', default=2, type=int, help='out_dim')
parser.add_argument('--lr', default=1e-4, type=float, help='lr')  # [1e-4, ]
parser.add_argument('--attackepoch', default=30, type=int, help='attackepoch')
parser.add_argument('--num_cutlayer', default=64, type=int, help='num_cutlayer')
parser.add_argument('--new_shadow', default=100, type=int, help='new_shadow')
parser.add_argument('--noise', default=10, type=float, help='noise')
args = parser.parse_args()

dataset = args.dataset
acti = args.acti
batch_size = args.batch_size
epochs_attack =args.epochs_attack
epochs=args.epochs
end_shadow = args.end_shadow
num_data = args.num_data
attack_label = args.attack_label
cutlayer=args.num_cutlayer
out_dim = args.out_dim
lr = args.lr
client_c=args.client
attackepoch=args.attackepoch
newshadow=args.new_shadow

minibatch=num_data/batch_size
minibatch=int(minibatch)
noise=args.noise

# Define record path
save_path1 = f'Results_attack/{dataset}/{cutlayer}/{newshadow}/{noise}'
if not os.path.exists(save_path1):
  os.makedirs(save_path1)

# Define data
if dataset=='bank_marketing':
  traindata = pd.read_csv(f'Results_noise/{dataset}/{acti}/num_client2/n{noise}/{client_c}_{acti}_{dataset}_c{cutlayer}_shadow.csv', sep=',',header=None)
  traindata = np.array(traindata)
  testdata = pd.read_csv(f'Results_noise/{dataset}/{acti}/num_client2/n{noise}/{client_c}_{acti}_{dataset}_c{cutlayer}.csv',sep=',', header=None)
  testdata = np.array(testdata)
  if client_c=='VFL_client1':
    if attack_label == 'y1':   # "job_admin":10
      traindata=np.delete(traindata,[cutlayer,cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+6,cutlayer+7,cutlayer+8,cutlayer+9,cutlayer+10], 1)
      testdata=np.delete(testdata,[cutlayer,cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+6,cutlayer+7,cutlayer+8,cutlayer+9,cutlayer+10], 1)
    elif attack_label == 'y2':   # "loan":2
      traindata=np.delete(traindata,[cutlayer,cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+5,cutlayer+7,cutlayer+8,cutlayer+9,cutlayer+10], 1)
      testdata=np.delete(testdata,[cutlayer,cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+5,cutlayer+7,cutlayer+8,cutlayer+9,cutlayer+10], 1)
    elif attack_label == 'y3':   # "education":7
      traindata=np.delete(traindata,[cutlayer,cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+5,cutlayer+6,cutlayer+8,cutlayer+9,cutlayer+10], 1)
      testdata=np.delete(testdata,[cutlayer,cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+5,cutlayer+6,cutlayer+8,cutlayer+9,cutlayer+10], 1)
    elif attack_label == 'y4':   # "housing":2
      traindata=np.delete(traindata,[cutlayer,cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+5,cutlayer+6,cutlayer+7,cutlayer+9,cutlayer+10], 1)
      testdata=np.delete(testdata,[cutlayer,cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+5,cutlayer+6,cutlayer+7,cutlayer+9,cutlayer+10], 1)


elif dataset=='credit':
  traindata = pd.read_csv(f'Results_noise/{dataset}/{acti}/num_client2/n{noise}/{client_c}_{acti}_{dataset}_c{cutlayer}_shadow.csv', sep=',',header=None)
  traindata = np.array(traindata)
  testdata = pd.read_csv(f'Results_noise/{dataset}/{acti}/num_client2/n{noise}/{client_c}_{acti}_{dataset}_c{cutlayer}.csv',sep=',', header=None)
  testdata = np.array(testdata)
  if client_c=='VFL_client1':
      if attack_label == 'y1':   # "sex":2
        traindata=np.delete(traindata,[cutlayer,cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+5,cutlayer+6,cutlayer+8,cutlayer+9,cutlayer+10,cutlayer+11,cutlayer+12],1)
        testdata=np.delete(testdata,[cutlayer,cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+5,cutlayer+6,cutlayer+8,cutlayer+9,cutlayer+10,cutlayer+11,cutlayer+12],1)
      elif attack_label == 'y2':   # "education":6
        traindata=np.delete(traindata,[cutlayer,cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+5,cutlayer+6,cutlayer+7,cutlayer+9,cutlayer+10,cutlayer+11,cutlayer+12],1)
        testdata=np.delete(testdata,[cutlayer,cutlayer,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+5,cutlayer+6,cutlayer+7,cutlayer+9,cutlayer+10,cutlayer+11,cutlayer+12],1)
      elif attack_label == 'y3':   # "marriage":3
        traindata=np.delete(traindata,[cutlayer,cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+5,cutlayer+6,cutlayer+7,cutlayer+8,cutlayer+10,cutlayer+11,cutlayer+12],1)
        testdata=np.delete(testdata,[cutlayer,cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+5,cutlayer+6,cutlayer+7,cutlayer+8,cutlayer+10,cutlayer+11,cutlayer+12],1)


elif dataset=='census':
  traindata = pd.read_csv(f'Results_noise/{dataset}/{acti}/num_client2/n{noise}/{client_c}_{acti}_{dataset}_c{cutlayer}_shadow.csv', sep=',',header=None)
  traindata = np.array(traindata)
  testdata = pd.read_csv(f'Results_noise/{dataset}/{acti}/num_client2/n{noise}/{client_c}_{acti}_{dataset}_c{cutlayer}.csv',sep=',', header=None)
  testdata = np.array(testdata)
  if client_c=='VFL_client1':
    dit = { 'y1':cutlayer+6,#'class of worker',#9
    'y2':cutlayer+7,#'enrolled in edu inst last wk',  3
    'y3':cutlayer+8,#'marital status',#2
    'y4':cutlayer+9,#'major industry code'24
    'y5':cutlayer+10,#'major occupation code'#15
    'y6':cutlayer+11,#'race', 5
    'y7':cutlayer+12,#'hispanic origin',#10
    'y8':cutlayer+13,#'sex'2
    'y9':cutlayer+14,#'member of a labor union',#3
    'y10':cutlayer+15,#'reason for unemployment',6
    'y11':cutlayer+16,#'full or part time employment stat',#8
    'y12':cutlayer+17,#'tax filer status',6
    'y13':cutlayer+18,#'ndustry code',52
    }
    attlist=['age','capital gains','capital losses','divdends from stocks','num persons worked for employer','wage per hour', 'tax filer status', 'veterans benefits', 'marital status','sex', 'full or part time employment stat', 'enrolled in edu inst last wk', 'race', 'own business or self employed',
                       'member of a labor union', 'citizenship', 'live in this house 1 year ago', 'fill inc questionnaire for veterans admin', 'year', ]
    attlistnum=[cutlayer,cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+5,cutlayer+6,cutlayer+7,cutlayer+8,cutlayer+9,cutlayer+10,cutlayer+11,cutlayer+12,cutlayer+13,cutlayer+14,cutlayer+15,cutlayer+16,cutlayer+17,cutlayer+18,cutlayer+19]
    attlistnum.remove(dit[attack_label])
    traindata=np.delete(traindata,attlistnum,1)
    testdata=np.delete(testdata,attlistnum,1)
  elif client_c=='VFL_client2':
    dit = { 
    'y1':cutlayer,#'region of previous residence'}#6
    'y2':cutlayer+1,#'detailed household summary in household',8     
    'y3':cutlayer+2,#'migration code-change in msa',#10
    'y4':cutlayer+3,#'migration code-change in reg', #9
    'y5':cutlayer+4,#'migration code-move within reg',#10
    'y6':cutlayer+5,#'live in this house 1 year ago', 3  
    'y7':cutlayer+6,#'migration prev res in sunbelt',4
    'y8':cutlayer+7,#'family members under 18',5
    'y9':cutlayer+8,#'citizenship',#5
    'y10':cutlayer+9,#'own business or self employed',3
    'y11':cutlayer+10,#'fill inc questionnaire for veterans admin',#3
    'y12':cutlayer+11,#'veterans benefits',3
    'y13':cutlayer+12,#'year'}#2
    'y14':cutlayer+13,#'detailed household and family stat'#38
    'y15':cutlayer+14,#'country of birth father'#43
    'y16':cutlayer+15,#'country of birth mother'#43
    'y17':cutlayer+16,#'country of birth self'#43
    }
    attlist=['region of previous residence','detailed household and family stat','detailed household summary in household','migration code-change in msa',
    'migration code-change in reg','migration code-move within reg','live in this house 1 year ago','migration prev res in sunbelt','family members under 18','citizenship',
    'own business or self employed','fill inc questionnaire for veterans admin','veterans benefits','year','detailed household and family stat', 'country of birth father', 'country of birth mother','country of birth self']
    attlistnum=[cutlayer,cutlayer+1,cutlayer+2,cutlayer+3,cutlayer+4,cutlayer+5,cutlayer+6,cutlayer+7,cutlayer+8,cutlayer+9,cutlayer+10,cutlayer+11,cutlayer+12,cutlayer+13,cutlayer+14,cutlayer+15,cutlayer+16,cutlayer+17]
    traindata = np.delete(traindata, attlistnum, 1)
    testdata = np.delete(testdata, attlistnum, 1)
  
elif dataset=='cancer':
    data = pd.read_csv(f'Results/{client_c}_{acti}_{dataset}_c{cutlayer}.csv', sep=',',header=None)
    data=np.array(data,float)
    if client_c=='VFL_client1':
      if attack_label == 'y1':   # "Limit"
        data=np.delete(data,[cutlayer+1,cutlayer+2],1)
        #data[:,-1]=data[:,-1]*10
        #data[:,-1]=data[:,-1]-1
      if attack_label == 'y2':   # "Limit"
        data=np.delete(data,[cutlayer,cutlayer+2],1)
        #data[:,-1]=data[:,-1]*10
        #data[:,-1]=data[:,-1]-1
      if attack_label == 'y3':   # "Limit"
        data=np.delete(data,[cutlayer,cutlayer+1],1)
        #data[:,-1]=data[:,-1]*10
        #data[:,-1]=data[:,-1]-1
    elif client_c=='VFL_client2':
        if attack_label == 'y1':  # "Limit"
            data = np.delete(data, [cutlayer + 1, cutlayer + 2], 1)
            # data[:,-1]=data[:,-1]*10
            # data[:,-1]=data[:,-1]-1
        if attack_label == 'y2':  # "Limit"
            data = np.delete(data, [cutlayer, cutlayer + 2], 1)
            # data[:,-1]=data[:,-1]*10
            # data[:,-1]=data[:,-1]-1
        if attack_label == 'y3':  # "Limit"
            data = np.delete(data, [cutlayer, cutlayer + 1], 1)
            # data[:,-1]=data[:,-1]*10
            # data[:,-1]=data[:,-1]-1
    elif client_c=='VFL_client3':
        if attack_label == 'y1':  # "Limit"
            data = np.delete(data, [cutlayer + 1, cutlayer + 2], 1)
            # data[:,-1]=data[:,-1]*10
            # data[:,-1]=data[:,-1]-1
        if attack_label == 'y2':  # "Limit"
            data = np.delete(data, [cutlayer, cutlayer + 2], 1)
            # data[:,-1]=data[:,-1]*10
            # data[:,-1]=data[:,-1]-1
        if attack_label == 'y3':  # "Limit"
            data = np.delete(data, [cutlayer, cutlayer + 1], 1)
            # data[:,-1]=data[:,-1]*10
            # data[:,-1]=data[:,-1]-1


traindata=np.array(traindata)
testdata=np.array(testdata)
traindata=traindata[:newshadow,:]
data_dim=len(traindata[-1,:])
for i in range(data_dim-1):
  traindata[:,i] = (traindata[:,i]-traindata[:,i].mean())/traindata[:,i].std()
  testdata[:,i] = (testdata[:,i]-testdata[:,i].mean())/testdata[:,i].std()
print(traindata.shape)
if(dataset=='census'):
  traindata=traindata
else:
  traindata[:,-1]=traindata[:,-1]-1
  testdata[:,-1]=testdata[:,-1]-1
sensordata_num,sensor_num = traindata.shape
acc1=['epochs', 'acc','precision','recall','f1score','TPR','FPR', 'TNR', 'TP','FP','TN','FN','AUC']


# Define model
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
train_set=traindata
test_set=testdata
print(out_dim)
model = LinearNet_mul(in_dim=cutlayer, n_hidden_1=500, n_hidden_2=128, out_dim=out_dim).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train(dataloader, model):
    size = len(train_set)
    model.train()
    correct = 0
    for batch, data in enumerate(dataloader):
        X=data[:,:data_dim-1].to(device)
        X=X.to(torch.float32)
        y=data[:,-1].to(device).long()
        # y=torch.tensor(y,dtype=int)
        # Compute prediction error
        pred = model(X)
        loss = loss_func(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        if batch % 10 == 0 and batch != 0:
            correct_train = correct / ((batch+1) * batch_size)
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%")

def test(dataloader, model):
    n_classes=out_dim
    size = len(test_set)
    num_batches = len(dataloader)
    print('size:', size)
    print('num_batches:', num_batches)
    model.eval()
    test_loss, correct = 0, 0
    TP,FP,TN,FN=0,0,0,0
    ypred = []
    ytrue = []
    y_pred = []
    y_true = []
   #aucypred=np.zeros((size,out_dim))##
    with torch.no_grad():
      for data in dataloader:
        X = data[:, :data_dim-1].to(device)
        X = X.to(torch.float32)
        y = data[:, -1].to(device).long()
        pred= model(X)
        test_loss += loss_func(pred, y).item()
        correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

        # for TP FP~~
        ypred.extend(np.array(pred.argmax(1).cpu(),dtype=int))
        ytrue.extend(np.array(y.cpu(),dtype=int))
        
        # for auc
        y_one_hot = torch.randint(1,(batch_size, n_classes)).to(device).scatter_(1,y.view(-1, 1),1)
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
    correct /=size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    acc = (TP+TN)/(TP + FP + TN + FN)
    precision = 1.0*TP/(TP + FP)
    recall = 1.0*TP/(TP + FN)
    F_meature = 2.0*precision*recall/(precision + recall)
    acc1=accuracy_score(ytrue,ypred)
    # acc2=accuracy_score(ytrue, ypred, normalize=True, sample_weight=None)
    acc2=balanced_accuracy_score(ytrue,ypred)
    f1=f1_score(ytrue,ypred,average='macro')
    precision1=precision_score(ytrue,ypred,average='macro')
    recall1=recall_score(ytrue,ypred,average='macro')
    f2=f1_score(ytrue,ypred,average='weighted')
    precision2=precision_score(ytrue,ypred,average='weighted')
    recall2=recall_score(ytrue,ypred,average='weighted')
    TPR=TP/(TP+FN)
    FPR=FP/(FP+TN)
    TNR = TN/(FP+TN)

    

    #auc_score=0
    auc_score=roc_auc_score(y_true, y_pred, multi_class='ovr')
    return acc,precision,recall,F_meature,TPR, FPR,TNR, TP,FP,TN,FN,auc_score,acc1,precision1,f1,recall1,precision2,f2,recall2,acc2


train_iter = Data.DataLoader(
    dataset=train_set,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    drop_last=True,
)

test_iter = Data.DataLoader(
    dataset=test_set,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    drop_last=True,
)

filename=f'Results_attack/{dataset}/{cutlayer}/{newshadow}/{noise}/{dataset}_{acti}_c{cutlayer}_{client_c}_{attack_label}_{attackepoch}_{newshadow}.txt'
f2 = open(filename, 'w')
f2.write('Attack_'+ str(dataset)+'_epochs_attack'+ str(epochs_attack)+'_Adam_lr_'+ str(lr)+ '_end_shadow_' + str(end_shadow)+'_'+ str(attack_label) + str(attackepoch)+'\n')

f2.write(str(acc1)+'\n')
for t in range(epochs_attack):
    print(f"Epoch {t+1}")
    train(train_iter, model)
    acc,precision,recall,fscore,TPR,FPR,TNR,TP,FP,TN,FN,auc,meacc,meprecission,mef1,merecall,precision2,f12,merecall2,acc2=test(test_iter, model)
    acc1 ={'Epochs':t,
            'acc':acc, 
            'precision':precision, 
            'recall':recall,
            'fscore':fscore, 
            'TPR':TPR, 
            'FPR':FPR, 
            'TNR':TNR, 
            'TP':TP,
            'FP':FP, 
            'TN':TN, 
            'FN':FN, 
            'AUC':auc,
            'MEACC':meacc,
            'MEPrecision':meprecission,
            'MEF1':mef1,
            'MErecall':merecall,
            'MEPrecision2':precision2,
            'MEF12':f12,
            'MErecall2':merecall2,
            'ACC2':acc2
            }
    for key in acc1:
        f2.write('\n')
        f2.writelines('"' + str(key) + '": ' + str(acc1[key]))        
    f2.write('\n')
    f2.write('\n')
    f2.write('\n')
    print(acc1)
f2.close()
print("Done!")