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
    parser.add_argument('--dataset1', type=str, default='utkface', help="dataset1")
    parser.add_argument('--dataset2', type=str, default='utkface', help="dataset2")
    parser.add_argument('--model', type=str, default='lenet', help="model")
    parser.add_argument('--acti', type=str, default='leakyrelu', help="acti")
    parser.add_argument('--attack_label', type=int, default='0')
    parser.add_argument('--attributes', type=str, default="race_gender", help="For attrinf, two attributes should be in format x_y e.g. race_gender")
    parser.add_argument('--lr', default=1e-4, type=float, help='lr')
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--attack_batch_size', default=100, type=int, help='attack_batch_size')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--num_cutlayer', default=1000, type=int, help='num_cutlayer')
    parser.add_argument('--noise_scale', default=0, type=float, help='noise_scale')
    parser.add_argument('--number_client', default=2, type=int, help='number_client')
    parser.add_argument('--num_shadow', default=3, type=int, help='num_shadow')
    parser.add_argument('--level1', default=1, type=int, help='level1')
    parser.add_argument('--level2', default=1, type=int, help='level2')
    return parser.parse_args(argv[1:])


# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Train_Client Side Program
def train_client(dataloader, client_model_fake_1, client_model_1, client_model_2, t):
    client_model_1.eval()
    client_model_2.eval()
    client_model_fake_1.train()

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(X)

        # client1--train and update
        fx1_fake = client_model_fake_1(X_1) 


        fx1 = client_model_1(X_1)
        noise1 = np.random.normal(0, noise_scale, fx1.size())
        noise1 = torch.from_numpy(noise1).float().to(device)
        client1_fx = (fx1+noise1).clone().detach()

        loss = ((fx1_fake - client1_fx) ** 2).sum()

        optimizer_client_fake1.zero_grad()
        loss.backward()
        optimizer_client_fake1.step()

        if  batch_id == num_shadow :
            print('train_current_loss:', loss.item(), file=filename)
            train_loss.append(loss.item())
            break


# Test_Server Side Program
def test_true_server(client1_fx, client2_fx, y, batch_id, correct, size):
    server_model.eval()
    correct = correct

    # train and update
    fx_server = server_model(client1_fx, client2_fx)
    test_true.extend(fx_server.argmax(1).tolist())
    loss = criterion(fx_server, y)

    correct += (fx_server.argmax(1) == y).type(torch.float).sum().item()
    correct_train = correct / size
    loss, current = loss.item(), (batch_id + 1) * len(y)
    if batch_id == len(test_data_1) - 1:
        print(f"ttest-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%",
              file=filename)
    return correct

# Test_Server Side Program
def test_fake_server(client1_fx, client2_fx, y, batch_id, correct, size):
    server_model.eval()
    correct = correct

    # train and update
    fx_server = server_model(client1_fx, client2_fx)
    test_fake.extend(fx_server.argmax(1).tolist())
    loss = criterion(fx_server, y)

    correct += (fx_server.argmax(1) == y).type(torch.float).sum().item()
    correct_train = correct / size
    loss, current = loss.item(), (batch_id + 1) * len(y)
    if batch_id == len(test_data_1) - 1:
        print(f"ttest-loss: {loss:>7f}  [{current:>5d}/{size:>5d}]  Accuracy: {(100 * correct_train):>0.1f}%",
              file=filename)
        test_acc.append(100 * correct_train)
    return correct


# Test_Client Side Program
def test_client(dataloader, client_model_fake_1, client_model_1, client_model_2, t):
    client_model_1.eval()
    client_model_2.eval()
    client_model_fake_1.eval()
    correct_true = 0
    correct_fake = 0
    size = len(dataloader.dataset)
    global test_true

    for batch_id, batch in enumerate(dataloader):
        X, target = batch
        X_1, X_2 = split_data(X)
        Y_1 = target[0].to(device)

        # client1--train and update
        fx1 = client_model_1(X_1)
        fx1_fake = client_model_fake_1(X_1)
        fx2 = client_model_2(X_2)

        noise1 = np.random.normal(0, noise_scale, fx1.size())
        noise1 = torch.from_numpy(noise1).float().to(device)
        noise2 = np.random.normal(0, noise_scale, fx1.size())
        noise2 = torch.from_numpy(noise2).float().to(device)
        client1_fx = (fx1+noise1).clone().detach().requires_grad_(True)
        client2_fx = (fx2+noise2).clone().detach().requires_grad_(True)


        client1_fx = fx1.clone().detach().requires_grad_(True)
        client1_fx_fake = fx1_fake.clone().detach().requires_grad_(True)
        client2_fx = fx2.clone().detach().requires_grad_(True)

        # Sending activations to server and receiving gradients from server
        correct_true = test_true_server(client1_fx, client2_fx, Y_1, batch_id, correct_true, size)
        correct_fake = test_fake_server(client1_fx_fake, client2_fx, Y_1, batch_id, correct_fake, size)

    correct_fake /= size
    correct_true /= size
    print(f"Test Error_true: \n Accuracy: {(100 * correct_true):>0.1f}% \n", file=filename)
    print(f"Test Error_fake: \n Accuracy: {(100 * correct_fake):>0.1f}% \n", file=filename)


if __name__ == '__main__':
    print('Start training')
    args = parse_args()
    dataset1 = args.dataset1
    dataset2 = args.dataset2
    model = args.model
    batch_size = args.batch_size
    attack_batch_size = args.attack_batch_size
    epochs = args.epochs
    lr = args.lr
    acti = args.acti
    attributes = args.attributes
    attack_label = args.attack_label
    num_cutlayer = args.num_cutlayer
    noise_scale = args.noise_scale
    number_client = args.number_client
    level1 = args.level1
    level2 = args.level2
    num_shadow =args.num_shadow
    time_start_load_everything = time.time()

    if dataset2 == 'utkface':
        attributes = 'race_gender'

    if dataset2 == 'celeba':
        attributes = "attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr"

    # Define record path
    save_path = f'Results_MS/{dataset1}-{dataset2}/{model}/level{level1}-level{level2}/num_shadow{num_shadow}/c{num_cutlayer}/client{number_client}/n{noise_scale}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = open(f'Results_MS/{dataset1}-{dataset2}/{model}/level{level1}-level{level2}/num_shadow{num_shadow}/c{num_cutlayer}/client{number_client}/n{noise_scale}/c{num_cutlayer}_{acti}.txt', 'w+')
    

    ### Load data
    root_path = '.'
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    train_data_2, test_data_2, shadow_data_2, num_classes1_2, num_classes2_2, channel_2, hideen_2 = load_data(args.dataset2, args.attack_label,
                                                                                   args.attributes, data_path,
                                                                                   attack_batch_size)
    
    train_data_1, test_data_1, shadow_data_1, num_classes1_1, num_classes2_1, channel_1, hideen = load_data(args.dataset1, args.attack_label,
                                                                                   args.attributes, data_path,
                                                                                   attack_batch_size)


    # Define model
    if model == 'resnet':
        client_model_1 = ResNet18(level=level1, hideen1=hideen, num_classes=num_cutlayer).to(device)
        client_model_2 = ResNet18(level=level1, hideen1=hideen, num_classes=num_cutlayer).to(device)
        # Define server-side model
        server_model = Server_ResNet(hideen2=num_cutlayer * 2, hideen3=256, hideen4=128, hideen5=64,
                                     num_classes=num_classes1_1).to(device)
        
        save_path1 = f'Results/{dataset1}/{model}/level{level1}/client{number_client}/n{noise_scale}/client1_c{num_cutlayer}.pth'
        save_path2 = f'Results/{dataset1}/{model}/level{level1}/client{number_client}/n{noise_scale}/client2_c{num_cutlayer}.pth'
        save_path3 = f'Results/{dataset1}/{model}/level{level1}/client{number_client}/n{noise_scale}/server_c{num_cutlayer}.pth'
        client_model_1 = torch.load(save_path1)
        client_model_2 = torch.load(save_path2)
        server_model = torch.load(save_path3)
        
        client_model_fake_1 = ResNet18(level=level2, hideen1=hideen, num_classes=num_cutlayer).to(device)
        optimizer_client_fake1 = torch.optim.Adam(client_model_fake_1.parameters(), lr=lr)


    # Define criterion
    criterion = nn.CrossEntropyLoss()

    test_true = []
    test_fake = []
    train_loss = []
    test_acc = []

    # start training
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------", file=filename)
        train_client(shadow_data_2, client_model_fake_1, client_model_1, client_model_2, t)
        test_client(test_data_1, client_model_fake_1, client_model_1, client_model_2, t)
    print("Done!", file=filename)

    save_path = f'Results_MS/{dataset1}-{dataset2}/{model}/level{level1}-level{level2}/num_shadow{num_shadow}/c{num_cutlayer}/client{number_client}/n{noise_scale}/client1_fake_c{num_cutlayer}.pth'
    torch.save(client_model_fake_1, save_path)


    test_aggrement = 0
    for i in range(len(test_true)):
        if test_true[i]==test_fake[i]:
            test_aggrement +=1

    print('len(test_true)', len(test_true),  file=filename)
    print('test_aggrement', test_aggrement,  file=filename)
    print('test_aggrement', test_aggrement/ len(test_true),  file=filename)













