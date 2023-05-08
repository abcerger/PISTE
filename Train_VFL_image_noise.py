import random
import time
from model.lenet import *
from model.resnet import *
from torch import nn
from sys import argv
import os
import argparse
import copy
import seaborn as sns
from utils_image import *


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
    parser = argparse.ArgumentParser(description='VFL')
    parser.add_argument('--dataset', type=str, default='utkface', help="dataset")
    parser.add_argument('--model', type=str, default='lenet', help="model")
    parser.add_argument('--level', default=1, type=int, help='number_client')
    parser.add_argument('--acti', type=str, default='leakyrelu', help="acti")
    parser.add_argument('--attack_label', type=int, default='0')
    parser.add_argument('--attributes', type=str, default="race_gender", help="For attrinf, two attributes should be in format x_y e.g. race_gender")
    parser.add_argument('--lr', default=1e-4, type=float, help='lr')
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    parser.add_argument('--num_cutlayer', default=1000, type=int, help='num_cutlayer')
    parser.add_argument('--noise_scale', default=0, type=float, help='noise_scale')
    parser.add_argument('--number_client', default=2, type=int, help='number_client')
    
    return parser.parse_args(argv[1:])


# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Train_Server Side Program
def train_server(client1_fx, client2_fx, Y_1, t, batch_id, correct, size):
    server_model.train()
    correct = correct
    global train_acc 
    global train_loss

    # train and update
    optimizer_server.zero_grad()
    fx_server = server_model(client1_fx, client2_fx)
    loss = criterion(fx_server, Y_1)

    # backward
    loss.backward()
    dfx1_client = client1_fx.grad.clone().detach().to(device)
    dfx2_client = client2_fx.grad.clone().detach().to(device)
    optimizer_server.step()
    correct += (fx_server.argmax(1) == Y_1).type(torch.float).sum().item()
    
    if t == epochs-1 and batch_id == len(train_data)-1:
        save_path3 = f'Results/{dataset}/{model}/level{level}/client{number_client}/n{noise_scale}/server_c{num_cutlayer}.pth'
        torch.save(server_model, save_path3)

    if batch_id == len(train_data)-1:
        correct_train = correct / size
        loss, current = loss.item(), (batch_id + 1) *batch_size
        print(f"train-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%",
            file=filename)
        train_acc.append(100 * correct_train)
        train_loss.append(loss)

    return dfx1_client, dfx2_client, correct


# Train_Client Side Program
def train_client(dataloader, client_model_1, client_model_2, t):
    client_model_1.train()
    client_model_2.train()
    correct = 0
    size = len(dataloader)*batch_size

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(X)
        Y_1 = target[0].to(device)
        Y_2 = target[1].view(-1, 1).to(device)


        # client1--train and update
        fx1 = client_model_1(X_1)
        fx2 = client_model_2(X_2)

        noise1 = np.random.normal(0, noise_scale, fx1.size())
        noise1 = torch.from_numpy(noise1).float().to(device)

        noise2 = np.random.normal(0, noise_scale, fx1.size())
        noise2 = torch.from_numpy(noise2).float().to(device)

        client1_fx = (fx1+noise1).clone().detach().requires_grad_(True)
        client2_fx = (fx2+noise2).clone().detach().requires_grad_(True)


        # Sending activations to server and receiving gradients from server
        g_fx1, g_fx2, correct = train_server(client1_fx, client2_fx, Y_1, t, batch_id, correct, size)

        # backward prop
        optimizer_client1.zero_grad()
        optimizer_client2.zero_grad()
        (fx1+noise1).backward(g_fx1)
        (fx2+noise2).backward(g_fx2)

        optimizer_client1.step()
        optimizer_client2.step()

        # record for attack
        if t == epochs-1:            
            # for property inference
            n1 = torch.cat([ (fx1+noise1), Y_2], dim=1)
            n1 = n1.cpu().detach().numpy()
            writer_1.writerows(n1)

            if batch_id == len(train_data)-1:
                save_path1 = f'Results/{dataset}/{model}/level{level}/client{number_client}/n{noise_scale}/client1_c{num_cutlayer}.pth'
                save_path2 = f'Results/{dataset}/{model}/level{level}/client{number_client}/n{noise_scale}/client2_c{num_cutlayer}.pth'
                torch.save(client_model_1, save_path1)
                torch.save(client_model_2, save_path2)

                X1_s = copy.deepcopy(X_1.detach().cpu().numpy())
                client1_fx_s = copy.deepcopy(client1_fx.detach().cpu().numpy())
                np.save(f'Results/{dataset}/{model}/level{level}/client{number_client}/n{noise_scale}/X1_c{num_cutlayer}.npy', X1_s)
                np.save(f'Results/{dataset}/{model}/level{level}/client{number_client}/n{noise_scale}/fx1_c{num_cutlayer}.npy', client1_fx_s)


# Test_Server Side Program
def test_server(client1_fx, client2_fx, y, batch_id, correct, size):
    server_model.train()
    correct = correct

    # train and update
    optimizer_server.zero_grad()
    fx_server = server_model(client1_fx, client2_fx)
    loss = criterion(fx_server, y)

    correct += (fx_server.argmax(1) == y).type(torch.float).sum().item()
    correct_train = correct / size
    loss, current = loss.item(), (batch_id + 1) * len(y)
    if batch_id == len(test_data) - 1:
        print(f"ttest-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%",
              file=filename)
        test_acc.append(100 * correct_train)
        test_loss.append(loss)
    return correct


# Test_Client Side Program
def test_client(dataloader, client_model_1, client_model_2, t):
    client_model_1.eval()
    client_model_2.eval()
    correct = 0
    size = len(dataloader.dataset)

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(X)

        Y_1 = target[0].to(device)

        # client1--train and update
        fx1 = client_model_1(X_1)
        fx2 = client_model_2(X_2)
        noise1 = np.random.normal(0, noise_scale, fx1.size())
        noise1 = torch.from_numpy(noise1).float().to(device)

        noise2 = np.random.normal(0, noise_scale, fx1.size())
        noise2 = torch.from_numpy(noise2).float().to(device)

        client1_fx = (fx1+noise1).clone().detach().requires_grad_(True)
        client2_fx = (fx2+noise2).clone().detach().requires_grad_(True)

        # Sending activations to server and receiving gradients from server
        correct = test_server(client1_fx, client2_fx, Y_1, batch_id, correct, size)

    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}% \n", file=filename)

# Test_Client Side Program
def shadow_client(dataloader, client_model_1, client_model_2, t):
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

        noise1 = np.random.normal(0, noise_scale, fx1.size())
        noise1 = torch.from_numpy(noise1).float().to(device)

        # for property inference
        n1 = torch.cat([(fx1+noise1), Y_2], dim=1)
        n1 = n1.cpu().detach().numpy()
        writer_shadow_1.writerows(n1)


if __name__ == '__main__':
    print('Start training')
    args = parse_args()
    dataset = args.dataset
    model = args.model
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    acti = args.acti
    attributes = args.attributes
    attack_label = args.attack_label
    num_cutlayer = args.num_cutlayer
    noise_scale = args.noise_scale
    level =args.level
    number_client = args.number_client
    time_start_load_everything = time.time()

    if dataset == 'utkface':
        attributes = 'race_gender'

    if dataset == 'celeba':
        attributes = "attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr"


    # Define record path
    save_path = f'Results/{dataset}/{model}/level{level}/client{number_client}/n{noise_scale}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = open(f'Results/{dataset}/{model}/level{level}/client{number_client}/n{noise_scale}/c{num_cutlayer}_{acti}_b{batch_size}.txt', 'w+')
    
    writer_1, writer_2 = records_path(save_path,  num_cutlayer)
    writer_shadow_1, writer_shadow_2 = records_shadow_path(save_path,  num_cutlayer)

    ### Load data
    root_path = '.'
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    train_data, test_data, shadow_data, num_classes1, num_classes2, channel, hideen = load_data(args.dataset, args.attack_label,
                                                                                   args.attributes, data_path,
                                                                                   batch_size)

    # Define model
    if model == 'resnet':
        if acti == 'leakyrelu':
            # Define client-side model
            client_model_1 = ResNet18(level=level, hideen1=hideen, num_classes=num_cutlayer).to(device)
            client_model_2 = ResNet18(level=level, hideen1=hideen, num_classes=num_cutlayer).to(device)
            optimizer_client1 = torch.optim.Adam(client_model_1.parameters(), lr=lr)
            optimizer_client2 = torch.optim.Adam(client_model_2.parameters(), lr=lr)

            # Define server-side model
            server_model = Server_ResNet(hideen2=num_cutlayer * 2, hideen3=256, hideen4=128, hideen5=64,
                                         num_classes=num_classes1).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer_server = torch.optim.Adam(server_model.parameters(), lr=lr)


    # Define criterion
    criterion = nn.CrossEntropyLoss()
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    # start training
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------", file=filename)
        train_client(train_data, client_model_1, client_model_2, t)
        test_client(test_data, client_model_1, client_model_2, t)
    print("Done!", file=filename)

    shadow_client(shadow_data, client_model_1, client_model_2, t)

    












