import random
import time
from datetime import datetime
from torch.utils.data.sampler import  WeightedRandomSampler
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from utils_tabular import *
from torch import nn
from sys import argv
import os
import argparse
from collections import defaultdict
import csv
import copy
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif


# Define A random_seed
def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
set_random_seed(1234)


# Define parameter
def parse_args():
    parser = argparse.ArgumentParser(description='VFL1')
    parser.add_argument('--dataset', type=str, default='credit', help="dataset") # [bank_marketing, credit, census]
    parser.add_argument('--acti', type=str, default='leakyrelu_2', help="acti")  
    parser.add_argument('--number_client', default=2, type=int, help='number_client')
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('--num_cutlayer', default=200, type=int, help='num_cutlayer')  
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='lr')  
    parser.add_argument('--pruning', default=0, type=float, help='pruning')
    parser.add_argument('--mode', type=str, default='complex3')
    return parser.parse_args(argv[1:])

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


def l2_regularization(model, l2_alpha):
    reg_loss = None
    for param in model.parameters():
        if reg_loss is None:
            reg_loss = l2_alpha * torch.sum(param**2)
        else:
            reg_loss = reg_loss + l2_alpha * param.norm(2)**2
    return reg_loss


# Train_Server Side Program
def train_server(client1_fx, client2_fx, Y, t,  batch,correct):
    server_model.train()
    correct = correct

    global train_acc 
    global train_loss

    # Data of (output_cutlayer, y) for server
    client1_fx = client1_fx.to(device)
    client2_fx = client2_fx.to(device)
    Y = Y.to(device)

    # train and update
    optimizer_server.zero_grad()
    fx_server = server_model(client1_fx, client2_fx)

    loss = criterion(fx_server, Y) 
    # loss = criterion(fx_server, Y) + l2_regularization(server_model, 0.0005) +l2_regularization(client_model_1, 0.0005) +l2_regularization(client_model_2, 0.0005) 
    loss.backward()

    optimizer_server.step()
  
    correct += (fx_server.argmax(1) == Y).type(torch.float).sum().item()
    correct_train = correct / size_train
    loss, current = loss.item(), (batch+1) * len(client1_fx.grad.clone().detach())

    if batch == len(train_iter) - 1:
        print(f"train-loss: {loss:>7f}  [{current:>5d}/{size_train:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%",
          file=filename)
        train_acc.append(100 * correct_train)
        train_loss.append(loss)

    # record for attack
    if t == epochs-1 and batch == len(train_iter)-1:
      save_path3 = f'Results_pruning/{dataset}/{acti}/num_client{number_client}/p{pruning}/server_c{num_cutlayer}_{acti}_{batch_size}_epoch{t}_b{batch}.pth'
      torch.save(server_model, save_path3)
      
    return client1_fx.grad, client2_fx.grad, correct


# Train_Client Side Program
def train_client(dataloader, client_model_1, client_model_2,  t):
    client_model_1.train()
    client_model_2.train()

    correct = 0
    for batch, (X, Y) in enumerate(dataloader):
      X, Y = X.to(device), Y.to(device)
      if dataset == 'bank_marketing':
          X1 = X[:, :10]
          X2 = X[:, 10:]

      if dataset == 'credit':
          X1 = X[:, :12]
          X2 = X[:, 12:]

      if dataset == 'census':
          X1 = X[:, :19]
          X2 = X[:, 19:]

      # client--train and update
      fx1 = client_model_1(X1, 1)
      fx2 = client_model_2(X2, 2)

      pruning_fx1 = torch.ones_like(fx1)
      _, zis = torch.sort(fx1, descending=False)
      zi = zis[:, :int(pruning*(fx1.shape[-1]))]
      for i in range(len(zi)):
          pruning_fx1[i, zi[i]] =0
      client1_fx = (fx1 * pruning_fx1).clone().detach().requires_grad_(True)


      pruning_fx2 = torch.ones_like(fx2)
      _, zis = torch.sort(fx2, descending=False)
      zi = zis[:, :int(pruning*(fx2.shape[-1]))]
      for i in range(len(zi)):
          pruning_fx2[i, zi[i]] =0
      client2_fx = (fx2 * pruning_fx2).clone().detach().requires_grad_(True)
      
      # Sending activations to server and receiving gradients from server
      g_fx1, g_fx2, correct = train_server(client1_fx, client2_fx, Y, t, batch, correct)  

      # backward prop
      optimizer_client1.zero_grad()
      optimizer_client2.zero_grad()
      (fx1* pruning_fx1).backward(g_fx1) 
      (fx2* pruning_fx2).backward(g_fx2) 

      optimizer_client1.step()
      optimizer_client2.step()
      # record for training

      # record for attack
      if t == epochs-1:
        # record for attack
        YR = Y.reshape(batch_size,1)
        n1 = torch.cat([(fx1* pruning_fx1), X1, YR], dim=1)
        n2 = torch.cat([(fx2* pruning_fx2), X2, YR], dim=1)
        n1 = n1.cpu().detach().numpy()
        n2 = n2.cpu().detach().numpy()
        writer_1.writerows(n1)
        writer_2.writerows(n2)


        if batch == len(train_iter)-1:
          save_path1 = f'Results_pruning/{dataset}/{acti}/num_client{number_client}/p{pruning}/client1_c{num_cutlayer}_{acti}_epoch{t}_b{batch}.pth'
          save_path2 = f'Results_pruning/{dataset}/{acti}/num_client{number_client}/p{pruning}/client2_c{num_cutlayer}_{acti}_epoch{t}_b{batch}.pth'
          torch.save(client_model_1, save_path1)
          torch.save(client_model_2, save_path2)
 

def test_server(client1_fx, client2_fx, Y, batch, correct):
    server_model.eval()
    correct = correct

    # Data of (output_cutlayer, y) for server
    client1_fx = client1_fx.to(device)
    client2_fx = client2_fx.to(device)
    Y = Y.to(device)

    # eval
    fx_server = server_model(client1_fx, client2_fx)
    loss = criterion(fx_server, Y)
    correct += (fx_server.argmax(1) == Y).type(torch.float).sum().item()

    correct_train = correct / size_test
    loss, current = loss.item(), (batch+1) * len(client1_fx)
    if batch == len(test_iter)-1:
      print(f"ttest-loss: {loss:>7f}  [{current:>5d}/{size_test:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%" , file=filename)
      test_acc.append(100 * correct_train)
      test_loss.append(loss)

    return correct


# Test_Client Side Program
def test_client(dataloader, client_model_1, client_model_2, t):
    client_model_1.eval()
    client_model_2.eval()

    correct = 0
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        if dataset == 'bank_marketing':
          X1 = X[:, :10]
          X2 = X[:, 10:]

        if dataset == 'credit':
          X1 = X[:, :12]
          X2 = X[:, 12:]

        if dataset == 'census':
          X1 = X[:, :19]
          X2 = X[:, 19:]

        fx1 = client_model_1(X1, 1)        
        fx2 = client_model_2(X2, 2)

        pruning_fx1 = torch.ones_like(fx1)
        _, zis = torch.sort(fx1, descending=False)
        zi = zis[:, :int(pruning*(fx1.shape[-1]))]
        for i in range(len(zi)):
            pruning_fx1[i, zi[i]] =0
        client1_fx = (fx1 * pruning_fx1).clone().detach().requires_grad_(True)


        pruning_fx2 = torch.ones_like(fx2)
        _, zis = torch.sort(fx2, descending=False)
        zi = zis[:, :int(pruning*(fx2.shape[-1]))]
        for i in range(len(zi)):
            pruning_fx2[i, zi[i]] =0
        client2_fx = (fx2 * pruning_fx2).clone().detach().requires_grad_(True)

 
        # Sending activations to server and receiving gradients from server
        correct = test_server(client1_fx, client2_fx, Y, batch, correct)

    correct /= size_test
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}% \n", file=filename)


# Test_Client Side Program
def shadow_client(dataloader, client_model_1, client_model_2, t):
    client_model_1.eval()
    client_model_2.eval()

    correct = 0
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        if dataset == 'bank_marketing':
          X1 = X[:, :10]
          X2 = X[:, 10:]

        if dataset == 'credit':
          X1 = X[:, :12]
          X2 = X[:, 12:]

        if dataset == 'census':
          X1 = X[:, :19]
          X2 = X[:, 19:]

        fx1 = client_model_1(X1, 1)        
        fx2 = client_model_2(X2, 2)
        pruning_fx1 = torch.ones_like(fx1)
        _, zis = torch.sort(fx1, descending=False)
        zi = zis[:, :int(pruning*(fx1.shape[-1]))]
        for i in range(len(zi)):
            pruning_fx1[i, zi[i]] =0
        client1_fx = (fx1 * pruning_fx1).clone().detach()


        pruning_fx2 = torch.ones_like(fx2)
        _, zis = torch.sort(fx2, descending=False)
        zi = zis[:, :int(pruning*(fx2.shape[-1]))]
        for i in range(len(zi)):
            pruning_fx2[i, zi[i]] =0
        client2_fx = (fx2 * pruning_fx2).clone().detach()

        # record for attack
        YR = Y.reshape(batch_size,1)
        n1 = torch.cat([(fx1 * pruning_fx1), X1, YR], dim=1)
        n2 = torch.cat([(fx2 * pruning_fx2), X2, YR], dim=1)
        n1 = n1.cpu().detach().numpy()
        n2 = n2.cpu().detach().numpy()
        writer_shadow_1.writerows(n1)
        writer_shadow_2.writerows(n2)




if __name__ == '__main__':
    print('Start training')
    args = parse_args()
    batch_size = args.batch_size
    num_cutlayer = args.num_cutlayer
    epochs = args.epochs
    lr = args.lr
    dataset=args.dataset
    acti = args.acti
    number_client = args.number_client
    pruning = args.pruning


    time_start_load_everything = time.time()


    # Define record path
    save_path = f'Results_pruning/{dataset}/{acti}/num_client{number_client}/p{pruning}/'
    if not os.path.exists(save_path):
      os.makedirs(save_path)

    filename = open(f'Results_pruning/{dataset}/{acti}/num_client{number_client}/p{pruning}/c{num_cutlayer}_b{batch_size}.txt', 'w+')

    # Define records path
    writer_1, writer_2 = records_path(save_path, acti, dataset,  num_cutlayer)

    writer_shadow_1, writer_shadow_2 = records_shadow_path(save_path, acti, dataset,  num_cutlayer)


    # Define data
    train_iter, test_iter, shadow_iter,  size_train, size_test, size_shadow = load_data(dataset, batch_size)

        
    # Define model
    if dataset == 'bank_marketing':
      from model.Linear_NN_bm import *
    if dataset == 'credit':
      from model.Linear_NN_credit import *
    if dataset == 'census':
      from model.Linear_NN_census import *
    

    if dataset == 'bank_marketing':
        if acti == 'leakyrelu_1':
          client_model_1 = Client_LeakyreluNet_1_bm(in_dim=10, n_hidden_1=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_1_bm(in_dim=10, n_hidden_1=num_cutlayer, client=2).to(device)

        if acti == 'leakyrelu_2':
          client_model_1 = Client_LeakyreluNet_2_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_2_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=2).to(device)

        if acti == 'leakyrelu_3':
          client_model_1 = Client_LeakyreluNet_3_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_3_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=2).to(device)

        if acti == 'leakyrelu_4':
          client_model_1 = Client_LeakyreluNet_4_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_4_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=2).to(device)
        if acti == 'leakyrelu_5':
          client_model_1 = Client_LeakyreluNet_5_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer,n_hidden_5=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_5_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer,n_hidden_5=num_cutlayer, client=2).to(device)
    
    if dataset == 'credit':
        if acti == 'leakyrelu_2':
          client_model_1 = Client_LeakyreluNet_2_credit(in_dim=12, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_2_credit(in_dim=11, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=2).to(device)

        if acti == 'leakyrelu_3':
          client_model_1 = Client_LeakyreluNet_3_credit(in_dim=12, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_3_credit(in_dim=11, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=2).to(device)

        if acti == 'leakyrelu_4':
          client_model_1 = Client_LeakyreluNet_4_credit(in_dim=12, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_4_credit(in_dim=11, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=2).to(device)
        if acti == 'leakyrelu_5':
          client_model_1 = Client_LeakyreluNet_5_credit(in_dim=12, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, n_hidden_5=num_cutlayer,client=1).to(device)
          client_model_2 = Client_LeakyreluNet_5_credit(in_dim=11, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, n_hidden_5=num_cutlayer,client=2).to(device)

    if dataset == 'census':
        if acti == 'leakyrelu_2':
          client_model_1 = Client_LeakyreluNet_2_census(in_dim=19, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_2_census(in_dim=21, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=2).to(device)
          
        if acti == 'leakyrelu_3':
          client_model_1 = Client_LeakyreluNet_3_census(in_dim=19, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_3_census(in_dim=21, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=2).to(device)

        if acti == 'leakyrelu_4':
          client_model_1 = Client_LeakyreluNet_4_census(in_dim=19, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_4_census(in_dim=21, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=2).to(device)

        if acti == 'leakyrelu_5':
          client_model_1 = Client_LeakyreluNet_5_census(in_dim=19, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, n_hidden_5=num_cutlayer,client=1).to(device)
          client_model_2 = Client_LeakyreluNet_5_census(in_dim=21, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, n_hidden_5=num_cutlayer,client=2).to(device)


    server_model = Server_LeakyreluNet_4(n_hidden_2=num_cutlayer*2, n_hidden_3=num_cutlayer, n_hidden_4=128, n_hidden_5=64, out_dim=2).to(device)
    optimizer_client1 = torch.optim.Adam(client_model_1.parameters(), lr=lr)
    optimizer_client2 = torch.optim.Adam(client_model_2.parameters(), lr=lr)
    optimizer_server = torch.optim.Adam(server_model.parameters(), lr=lr)

  
    # Define criterion
    criterion = nn.CrossEntropyLoss()     

    # start training
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------", file=filename)
        train_client(train_iter, client_model_1, client_model_2, t)
        test_client(test_iter, client_model_1, client_model_2, t)

    print("Done!", file=filename)

    shadow_client(shadow_iter, client_model_1, client_model_2, t)




    fig=plt.figure(figsize=(6, 4.8), dpi=500)
    sns.set_style('darkgrid')
    x_axis_data = np.arange(epochs)
    plt.plot(x_axis_data, train_acc,  marker='D', markersize=1, alpha =0.8, linewidth=1, label=f'train_acc_{pruning}')  
    plt.plot(x_axis_data, test_acc,  marker='^', markersize=1, alpha =0.8, linewidth=1, label=f'test_acc_{pruning}')
    plt.legend(frameon=False, fontsize=10)  
    plt.xticks(fontsize=10) 
    plt.yticks(fontsize=10)

    plt.ylabel('Accuracy', fontsize=10, labelpad=3)  
    plt.xlabel('Epoch', fontsize=10)  
    plt.savefig(f'Results_pruning/{dataset}/{acti}/num_client{number_client}/p{pruning}/c{num_cutlayer}_b{batch_size}_acc.png', dpi = 500, bbox_inches='tight',pad_inches=0)



    fig=plt.figure(figsize=(6, 4.8), dpi=500)
    sns.set_style('darkgrid')
    x_axis_data = np.arange(epochs)
    plt.plot(x_axis_data, train_loss,  marker='D', markersize=1, alpha =0.8, linewidth=1, label=f'train_loss_{pruning}')  
    plt.plot(x_axis_data, test_loss,  marker='^', markersize=1, alpha =0.8, linewidth=1, label=f'test_loss_{pruning}')
    plt.legend(frameon=False, fontsize=10)  
    plt.xticks(fontsize=10) 
    plt.yticks(fontsize=10)

    plt.ylabel('loss', fontsize=10, labelpad=3)  
    plt.xlabel('Epoch', fontsize=10, y=-0.27)  
    plt.savefig(f'Results_pruning/{dataset}/{acti}/num_client{number_client}/p{pruning}/c{num_cutlayer}_b{batch_size}_loss.png', dpi = 500, bbox_inches='tight',pad_inches=0)






