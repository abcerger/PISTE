from torch import nn
import numpy as np
import torch
import math

# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


class Client_LeakyreluNet_2_census(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, n_hidden_3=64, n_hidden_4=64, client=1):
        super(Client_LeakyreluNet_2_census, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(54, n_hidden_1),
                                    nn.Tanh())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(451, n_hidden_1),
                                    nn.Tanh())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.Tanh())


    def forward(self, x, client):
        if client ==1:
            z = (x[:,6:]).long()
            c = [6, 3, 2, 2, 8, 3, 5, 3, 3, 5, 3, 3, 2]

            for i in range(13):
                a = torch.unsqueeze(z[:,i], dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :6], b], 1)
                else:
                    x = torch.cat([x, b], 1)

        if client ==2:
            z = (x[:,1:]).long()
            c = [47, 17, 15, 52, 24, 38, 8, 9, 5, 43, 43, 10, 43,       10, 9, 10, 51, 6, 4, 6]
            for i in range(20):

                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :1], b], 1)
                else:
                    x = torch.cat([x, b], 1)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer2_out


class Client_LeakyreluNet_3_census(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, n_hidden_3=64, n_hidden_4=64, client=1):
        super(Client_LeakyreluNet_3_census, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(54, n_hidden_1),
                                    nn.Tanh())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(451, n_hidden_1),
                                    nn.Tanh())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.Tanh())
        
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.Tanh())


    def forward(self, x, client):
        if client ==1:
            z = (x[:,6:]).long()
            c = [6, 3, 2, 2, 8, 3, 5, 3, 3, 5, 3, 3, 2]

            for i in range(13):
                a = torch.unsqueeze(z[:,i], dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :6], b], 1)
                else:
                    x = torch.cat([x, b], 1)

        if client ==2:
            z = (x[:,1:]).long()
            c = [47, 17, 15, 52, 24, 38, 8, 9, 5, 43, 43, 10, 43,       10, 9, 10, 51, 6, 4, 6]
            for i in range(20):

                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :1], b], 1)
                else:
                    x = torch.cat([x, b], 1)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer3_out


class Client_LeakyreluNet_4_census(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, n_hidden_3=64, n_hidden_4=64,  n_hidden_5=64, client=1):
        super(Client_LeakyreluNet_4_census, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(54, n_hidden_1),
                                    nn.Tanh())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(451, n_hidden_1),
                                    nn.Tanh())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.Tanh())
        
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.Tanh())


    def forward(self, x, client):
        if client ==1:
            z = (x[:,6:]).long()
            c = [6, 3, 2, 2, 8, 3, 5, 3, 3, 5, 3, 3, 2]

            for i in range(13):
                a = torch.unsqueeze(z[:,i], dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :6], b], 1)
                else:
                    x = torch.cat([x, b], 1)

        if client ==2:
            z = (x[:,1:]).long()
            c = [47, 17, 15, 52, 24, 38, 8, 9, 5, 43, 43, 10, 43,       10, 9, 10, 51, 6, 4, 6]
            for i in range(20):

                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :1], b], 1)
                else:
                    x = torch.cat([x, b], 1)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer4_out


class Client_LeakyreluNet_5_census(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, n_hidden_3=64, n_hidden_4=64,  n_hidden_5=64, n_hidden_6=64, client=1):
        super(Client_LeakyreluNet_5_census, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(54, n_hidden_1),
                                    nn.Tanh())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(451, n_hidden_1),
                                    nn.Tanh())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.Tanh())
        
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.Tanh())
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),
                                    nn.Tanh())


    def forward(self, x, client):
        if client ==1:
            z = (x[:,6:]).long()
            c = [6, 3, 2, 2, 8, 3, 5, 3, 3, 5, 3, 3, 2]

            for i in range(13):
                a = torch.unsqueeze(z[:,i], dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :6], b], 1)
                else:
                    x = torch.cat([x, b], 1)

        if client ==2:
            z = (x[:,1:]).long()
            c = [47, 17, 15, 52, 24, 38, 8, 9, 5, 43, 43, 10, 43,       10, 9, 10, 51, 6, 4, 6]
            for i in range(20):

                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :1], b], 1)
                else:
                    x = torch.cat([x, b], 1)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer5_out




class Server_LeakyreluNet_4(nn.Module):
    def __init__(self, n_hidden_2=256, n_hidden_3=128, n_hidden_4=64, n_hidden_5= 32, out_dim=2):
        super(Server_LeakyreluNet_4, self).__init__()
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.Tanh())
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),
                                    nn.Tanh())
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim),
                                   )
     

    def forward(self, x1, x2):

        x= torch.cat([x1, x2], dim=1)
        # print('x.shape', x.shape)
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        layer6_out = self.layer6(layer5_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer6_out




class Server_LeakyreluNet_4_attack(nn.Module):
    def __init__(self, n_hidden_2=256, n_hidden_3=128, n_hidden_4=64, n_hidden_5= 32, out_dim=2):
        super(Server_LeakyreluNet_4_attack, self).__init__()
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.Tanh())
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),
                                    nn.Tanh())
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim),
                                   )


    def forward(self, x):
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        layer6_out = self.layer6(layer5_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()

        return layer6_out
