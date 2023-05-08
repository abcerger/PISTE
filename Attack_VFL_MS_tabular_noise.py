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
    parser.add_argument('--dataset', type=str, default='bank_marketing', help="dataset")
    parser.add_argument('--acti1', type=str, default='leakyrelu_2', help="acti1")
    parser.add_argument('--acti2', type=str, default='leakyrelu_2', help="acti2")
    parser.add_argument('--number_client', default=2, type=int, help='number_client')
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--num_cutlayer', default=200, type=int, help='num_cutlayer')
    parser.add_argument('--num_shadow', default=10, type=int, help='num_cutlayer')
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='lr')  
    parser.add_argument('--noise_scale', default=0, type=float, help='noise_scale')
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


# Train_Client Side Program
def train_client(train_iter, client_model_1, client_model_1_fake, t):
    client_model_1.eval()
    client_model_1_fake.train()

    correct = 0
    for batch, (X, Y) in enumerate(train_iter):
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
      noise1 = np.random.normal(0, noise_scale, fx1.size())
      noise1 = torch.from_numpy(noise1).float().to(device)


      fx1_fake = client_model_1_fake(X1, 1)

      loss = ((fx1_fake - (fx1+noise1)) ** 2).sum()

      optimizer_client1_fake.zero_grad()
      loss.backward()
      optimizer_client1_fake.step()

      if  batch == num_shadow :
          print('train_current_loss:', loss.item(), file=filename)
          train_loss.append(loss.item())
          break

   

def test_true_server(client1_fx, client2_fx, Y, batch, correct):
    server_model.eval()
    correct = correct

    # Data of (output_cutlayer, y) for server
    client1_fx = client1_fx.to(device)
    client2_fx = client2_fx.to(device)
    Y = Y.to(device)

    # eval
    fx_server = server_model(client1_fx, client2_fx)
    test_true.extend(fx_server.argmax(1).tolist())
    correct += (fx_server.argmax(1) == Y).type(torch.float).sum().item()

    correct_train = correct / size_test
    current = (batch+1) * len(client1_fx)
    if batch == len(test_iter)-1:
      print(f"ttest-loss:  [{current:>5d}/{size_test:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%" , file=filename)

    return correct

def test_fake_server(client1_fx, client2_fx, Y, batch, correct):
    server_model.eval()
    correct = correct

    # Data of (output_cutlayer, y) for server
    client1_fx = client1_fx.to(device)
    client2_fx = client2_fx.to(device)
    Y = Y.to(device)

    # eval
    fx_server = server_model(client1_fx, client2_fx)
    test_fake.extend(fx_server.argmax(1).tolist())
    correct += (fx_server.argmax(1) == Y).type(torch.float).sum().item()

    correct_train = correct / size_test
    current = (batch+1) * len(client1_fx)
    if batch == len(test_iter)-1:
      print(f"ttest-loss:  [{current:>5d}/{size_test:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%" , file=filename)
      test_acc.append(100 * correct_train)

    return correct


# Test_Client Side Program
def test_client(test_iter, client_model_1, client_model_1_fake, server_model, t):
    client_model_1.eval()
    client_model_1_fake.eval()
    server_model.eval()
    global test_true


    correct_true = 0
    correct_fake = 0
    for batch, (X, Y) in enumerate(test_iter):
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
        fx1_fake = client_model_1_fake(X1, 1)        
        fx2 = client_model_2(X2, 2)

        client1_fx_fake = (fx1_fake).clone().detach().requires_grad_(True)

        noise1 = np.random.normal(0, noise_scale, fx1.size())
        noise1 = torch.from_numpy(noise1).float().to(device)
        noise2 = np.random.normal(0, noise_scale, fx1.size())
        noise2 = torch.from_numpy(noise2).float().to(device)
        client1_fx = (fx1+noise1).clone().detach().requires_grad_(True)
        client2_fx = (fx2+noise2).clone().detach().requires_grad_(True)

        
        # Sending activations to server and receiving gradients from server
        correct_true = test_true_server(client1_fx, client2_fx, Y, batch, correct_true)
        correct_fake = test_fake_server(client1_fx_fake, client2_fx, Y, batch, correct_fake)

    correct_fake /= size_test
    correct_true /= size_test
    print(f"Test Error_true: \n Accuracy: {(100 * correct_true):>0.1f}% \n", file=filename)
    print(f"Test Error_fake: \n Accuracy: {(100 * correct_fake):>0.1f}% \n", file=filename)




if __name__ == '__main__':
    print('Start training')
    args = parse_args()
    batch_size = args.batch_size
    num_cutlayer = args.num_cutlayer
    epochs = args.epochs
    lr = args.lr
    dataset=args.dataset
    acti1 = args.acti1
    acti2 = args.acti2
    num_shadow = args.num_shadow
    number_client = args.number_client
    noise_scale = args.noise_scale


    time_start_load_everything = time.time()


    # Define record path
    save_path = f'Results_MS/{dataset}/{acti1}-{acti2}/num_client{number_client}/n{noise_scale}/s{num_shadow}/'
    if not os.path.exists(save_path):
      os.makedirs(save_path)

    filename = open(f'Results_MS/{dataset}/{acti1}-{acti2}/num_client{number_client}/n{noise_scale}/s{num_shadow}/c{num_cutlayer}_b{batch_size}_shadow{num_shadow}_acti1_{acti1}acti2{acti2}.txt', 'w+')


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
        if acti1 == 'leakyrelu_1':
          client_model_1 = Client_LeakyreluNet_1_bm(in_dim=10, n_hidden_1=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_1_bm(in_dim=10, n_hidden_1=num_cutlayer, client=2).to(device)

        if acti1 == 'leakyrelu_2':
          client_model_1 = Client_LeakyreluNet_2_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_2_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=2).to(device)

        if acti1 == 'leakyrelu_3':
          client_model_1 = Client_LeakyreluNet_3_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_3_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=2).to(device)

        if acti1 == 'leakyrelu_4':
          client_model_1 = Client_LeakyreluNet_4_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_4_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=2).to(device)

        if acti1 == 'leakyrelu_5':
          client_model_1 = Client_LeakyreluNet_5_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer,n_hidden_5=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_5_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer,n_hidden_5=num_cutlayer, client=2).to(device)

        if acti2 == 'leakyrelu_1':
          client_model_1_fake = Client_LeakyreluNet_1_bm(in_dim=10, n_hidden_1=num_cutlayer, client=1).to(device)
          client_model_2_fake = Client_LeakyreluNet_1_bm(in_dim=10, n_hidden_1=num_cutlayer, client=2).to(device)

        if acti2 == 'leakyrelu_2':
          client_model_1_fake = Client_LeakyreluNet_2_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=1).to(device)
          client_model_2_fake = Client_LeakyreluNet_2_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=2).to(device)

        if acti2 == 'leakyrelu_3':
          client_model_1_fake = Client_LeakyreluNet_3_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=1).to(device)
          client_model_2_fake = Client_LeakyreluNet_3_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=2).to(device)

        if acti2 == 'leakyrelu_4':
          client_model_1_fake = Client_LeakyreluNet_4_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=1).to(device)
          client_model_2_fake = Client_LeakyreluNet_4_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=2).to(device)

        if acti2 == 'leakyrelu_5':
          client_model_1_fake = Client_LeakyreluNet_5_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer,n_hidden_5=num_cutlayer, client=1).to(device)
          client_model_2_fake = Client_LeakyreluNet_5_bm(in_dim=10, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer,n_hidden_5=num_cutlayer, client=2).to(device)



    
    if dataset == 'credit':
        if acti1 == 'leakyrelu_2':
          client_model_1 = Client_LeakyreluNet_2_credit(in_dim=12, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_2_credit(in_dim=11, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=2).to(device)

        if acti1 == 'leakyrelu_3':
          client_model_1 = Client_LeakyreluNet_3_credit(in_dim=12, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_3_credit(in_dim=11, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=2).to(device)

        if acti1 == 'leakyrelu_4':
          client_model_1 = Client_LeakyreluNet_4_credit(in_dim=12, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_4_credit(in_dim=11, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=2).to(device)
        if acti1 == 'leakyrelu_5':
          client_model_1 = Client_LeakyreluNet_5_credit(in_dim=12, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, n_hidden_5=num_cutlayer,client=1).to(device)
          client_model_2 = Client_LeakyreluNet_5_credit(in_dim=11, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, n_hidden_5=num_cutlayer,client=2).to(device)


        if acti2 == 'leakyrelu_2':
          client_model_1_fake = Client_LeakyreluNet_2_credit(in_dim=12, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=1).to(device)
          client_model_2_fake = Client_LeakyreluNet_2_credit(in_dim=11, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=2).to(device)

        if acti2 == 'leakyrelu_3':
          client_model_1_fake = Client_LeakyreluNet_3_credit(in_dim=12, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=1).to(device)
          client_model_2_fake = Client_LeakyreluNet_3_credit(in_dim=11, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=2).to(device)

        if acti2 == 'leakyrelu_4':
          client_model_1_fake = Client_LeakyreluNet_4_credit(in_dim=12, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=1).to(device)
          client_model_2_fake = Client_LeakyreluNet_4_credit(in_dim=11, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=2).to(device)
        if acti2 == 'leakyrelu_5':
          client_model_1_fake = Client_LeakyreluNet_5_credit(in_dim=12, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, n_hidden_5=num_cutlayer,client=1).to(device)
          client_model_2_fake = Client_LeakyreluNet_5_credit(in_dim=11, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, n_hidden_5=num_cutlayer,client=2).to(device)

    if dataset == 'census':
        if acti1 == 'leakyrelu_2':
          client_model_1 = Client_LeakyreluNet_2_census(in_dim=19, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_2_census(in_dim=21, n_hidden_1=128,  n_hidden_2=num_cutlayer, client=2).to(device)
          
        if acti1 == 'leakyrelu_3':
          client_model_1 = Client_LeakyreluNet_3_census(in_dim=19, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_3_census(in_dim=21, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, client=2).to(device)

        if acti1 == 'leakyrelu_4':
          client_model_1 = Client_LeakyreluNet_4_census(in_dim=19, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=1).to(device)
          client_model_2 = Client_LeakyreluNet_4_census(in_dim=21, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, client=2).to(device)

        if acti1 == 'leakyrelu_5':
          client_model_1 = Client_LeakyreluNet_5_census(in_dim=19, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, n_hidden_5=num_cutlayer,client=1).to(device)
          client_model_2 = Client_LeakyreluNet_5_census(in_dim=21, n_hidden_1=128,  n_hidden_2=128, n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer, n_hidden_5=num_cutlayer,client=2).to(device)

        if acti2 == 'leakyrelu_2':
            client_model_1_fake = Client_LeakyreluNet_2_census(in_dim=19, n_hidden_1=128, n_hidden_2=num_cutlayer,client=1).to(device)
            client_model_2_fake = Client_LeakyreluNet_2_census(in_dim=21, n_hidden_1=128, n_hidden_2=num_cutlayer,client=2).to(device)

        if acti2 == 'leakyrelu_3':
            client_model_1_fake = Client_LeakyreluNet_3_census(in_dim=19, n_hidden_1=128, n_hidden_2=128,n_hidden_3=num_cutlayer, client=1).to(device)
            client_model_2_fake = Client_LeakyreluNet_3_census(in_dim=21, n_hidden_1=128, n_hidden_2=128,n_hidden_3=num_cutlayer, client=2).to(device)

        if acti2 == 'leakyrelu_4':
            client_model_1_fake = Client_LeakyreluNet_4_census(in_dim=19, n_hidden_1=128, n_hidden_2=128,n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer,client=1).to(device)
            client_model_2_fake = Client_LeakyreluNet_4_census(in_dim=21, n_hidden_1=128, n_hidden_2=128,n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer,client=2).to(device)

        if acti2 == 'leakyrelu_5':
            client_model_1_fake = Client_LeakyreluNet_5_census(in_dim=19, n_hidden_1=128, n_hidden_2=128,n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer,n_hidden_5=num_cutlayer, client=1).to(device)
            client_model_2_fake = Client_LeakyreluNet_5_census(in_dim=21, n_hidden_1=128, n_hidden_2=128,n_hidden_3=num_cutlayer, n_hidden_4=num_cutlayer,n_hidden_5=num_cutlayer, client=2).to(device)



    server_model = Server_LeakyreluNet_4(n_hidden_2=num_cutlayer*2, n_hidden_3=num_cutlayer, n_hidden_4=128, n_hidden_5=64, out_dim=2).to(device)
    save_path1 = f'Results_noise/{dataset}/{acti1}/num_client2/n{noise_scale}/client1_c{num_cutlayer}_{acti1}_epoch{epochs-1}_b{len(train_iter)-1}.pth'
    save_path2 = f'Results_noise/{dataset}/{acti1}/num_client2/n{noise_scale}/client2_c{num_cutlayer}_{acti1}_epoch{epochs-1}_b{len(train_iter)-1}.pth'
    save_path3 = f'Results_noise/{dataset}/{acti1}/num_client2/n{noise_scale}/server_c{num_cutlayer}_{acti1}_{batch_size}_epoch{epochs-1}_b{len(train_iter)-1}.pth'

    client_model_1 = torch.load(save_path1)
    client_model_2 = torch.load(save_path2)
    server_model = torch.load(save_path3)

    optimizer_client1_fake = torch.optim.Adam(client_model_1_fake.parameters(), lr=lr)
    optimizer_client2_fake = torch.optim.Adam(client_model_2_fake.parameters(), lr=lr)

  
    # Define criterion
    criterion = nn.CrossEntropyLoss()     

    # start training
    train_loss = []
    test_acc = []

    test_true = []
    test_fake = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------", file=filename)
        train_client(shadow_iter, client_model_1, client_model_1_fake, t)
        test_client(test_iter, client_model_1, client_model_1_fake, server_model, t)

    print("Done!", file=filename)

    save_path = f'Results_MS/{dataset}/{acti1}-{acti2}/num_client{number_client}/n{noise_scale}/s{num_shadow}/client1_fake_c{num_cutlayer}_shadow{num_shadow}.pth'
    torch.save(client_model_1_fake, save_path)


    test_aggrement = 0
    for i in range(len(test_true)):
        if test_true[i]==test_fake[i]:
            test_aggrement +=1

    print('len(test_true)', len(test_true),  file=filename)
    print('test_aggrement', test_aggrement,  file=filename)
    print('test_aggrement', test_aggrement/ len(test_true),  file=filename)












