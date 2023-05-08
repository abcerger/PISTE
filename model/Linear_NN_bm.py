from torch import nn
import numpy as np
import torch
import math



# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


class Client_LeakyreluNet_1_bm(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, client=1):
        super(Client_LeakyreluNet_1_bm, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(29, n_hidden_1),
                                     nn.Tanh())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(28, n_hidden_1),
                                    nn.Tanh())

    def forward(self, x, client):
        if client ==1:
            z = (x[:,5:]-1).long()
            c = [11, 2, 7, 2, 2]

            for i in range(5):
                a = torch.unsqueeze(z[:,i], dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :5], b], 1)
                else:
                    x = torch.cat([x, b], 1)


        if client ==2:
            z = (x[:,5:]-1).long()
            c = [10, 3, 3, 2, 5]

            for i in range(5):
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :5], b], 1)
                else:
                    x = torch.cat([x, b], 1)

        layer1_out = self.layer1(x)

        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer1_out


class Client_LeakyreluNet_2_bm(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, client=1):
        super(Client_LeakyreluNet_2_bm, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(29, n_hidden_1),
                                    nn.Tanh())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(28, n_hidden_1),
                                    nn.Tanh())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    )

    def forward(self, x, client):
        if client ==1:
            z = (x[:,5:]-1).long()
            c = [11, 2, 7, 2, 2]

            for i in range(5):
                a = torch.unsqueeze(z[:,i], dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :5], b], 1)
                else:
                    x = torch.cat([x, b], 1)


        if client ==2:
            z = (x[:,5:]-1).long()
            c = [10, 3, 3, 2, 5]

            for i in range(5):
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :5], b], 1)
                else:
                    x = torch.cat([x, b], 1)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer2_out


class Client_LeakyreluNet_3_bm(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, n_hidden_3=64, client=1):
        super(Client_LeakyreluNet_3_bm, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(29, n_hidden_1),
                                    nn.Tanh())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(28, n_hidden_1),
                                    nn.Tanh())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                   nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    )

    def forward(self, x, client):
        if client ==1:
            z = (x[:,5:]-1).long()
            c = [11, 2, 7, 2, 2]

            for i in range(5):
                a = torch.unsqueeze(z[:,i], dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :5], b], 1)
                else:
                    x = torch.cat([x, b], 1)


        if client ==2:
            z = (x[:,5:]-1).long()
            c = [10, 3, 3, 2, 5]

            for i in range(5):
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :5], b], 1)
                else:
                    x = torch.cat([x, b], 1)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer3_out


class Client_LeakyreluNet_4_bm(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, n_hidden_3=64,  n_hidden_4=64, client=1):
        super(Client_LeakyreluNet_4_bm, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(29, n_hidden_1),
                                    nn.Tanh())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(28, n_hidden_1),
                                    nn.Tanh())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                   nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                   nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    )

    def forward(self, x, client):
        if client ==1:
            z = (x[:,5:]-1).long()
            c = [11, 2, 7, 2, 2]

            for i in range(5):
                a = torch.unsqueeze(z[:,i], dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :5], b], 1)
                else:
                    x = torch.cat([x, b], 1)


        if client ==2:
            z = (x[:,5:]-1).long()
            c = [10, 3, 3, 2, 5]

            for i in range(5):
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :5], b], 1)
                else:
                    x = torch.cat([x, b], 1)

        layer1_out = self.layer1(x)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)

        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer4_out


class Client_LeakyreluNet_5_bm(nn.Module):
    def __init__(self, in_dim=7, n_hidden_1=1024, n_hidden_2=64, n_hidden_3=64,  n_hidden_4=64, n_hidden_5=64, client=1):
        super(Client_LeakyreluNet_5_bm, self).__init__()

        if client ==1:
            self.layer1 = nn.Sequential(nn.Linear(29, n_hidden_1),
                                    nn.Tanh())

        if client ==2:
            self.layer1 = nn.Sequential(nn.Linear(28, n_hidden_1),
                                    nn.Tanh())

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                   nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                   nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                   nn.Tanh())
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),
                                   )

    def forward(self, x, client):
        if client ==1:
            z = (x[:,5:]-1).long()
            c = [11, 2, 7, 2, 2]

            for i in range(5):
                a = torch.unsqueeze(z[:,i], dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :5], b], 1)
                else:
                    x = torch.cat([x, b], 1)


        if client ==2:
            z = (x[:,5:]-1).long()
            c = [10, 3, 3, 2, 5]

            for i in range(5):
                a = torch.unsqueeze(z[:,i],dim=1).to(device)
                b = torch.zeros(len(z[:,i]), c[i]).to(device).scatter_(1, a, 1)
                if i ==0 :
                    x = torch.cat([x[:, :5], b], 1)
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




class Server_LeakyreluNet_6(nn.Module):
    def __init__(self, n_hidden_2=256, n_hidden_3=128, n_hidden_4=64, n_hidden_5= 32,n_hidden_6= 32,  n_hidden_7= 32, out_dim=2):
        super(Server_LeakyreluNet_6, self).__init__()
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.Tanh())
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),
                                    nn.Tanh())
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, n_hidden_6),
                                   nn.Tanh())
        self.layer7 = nn.Sequential(nn.Linear(n_hidden_6, n_hidden_7),
                                   nn.Tanh())
        self.layer8 = nn.Sequential(nn.Linear(n_hidden_7, out_dim),
                                )


    def forward(self, x1, x2):
        x= torch.cat([x1, x2], dim=1)
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        layer6_out = self.layer6(layer5_out)
        layer7_out = self.layer7(layer6_out)
        layer8_out = self.layer8(layer7_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()

        return layer8_out



class Server_LeakyreluNet_5(nn.Module):
    def __init__(self, n_hidden_2=256, n_hidden_3=128, n_hidden_4=64, n_hidden_5= 32,n_hidden_6= 32, out_dim=2):
        super(Server_LeakyreluNet_5, self).__init__()
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.Tanh())
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),
                                    nn.Tanh())
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, n_hidden_6),
                                   nn.Tanh())
        self.layer7 = nn.Sequential(nn.Linear(n_hidden_6, out_dim),
                                   )


    def forward(self, x1, x2):
        x= torch.cat([x1, x2], dim=1)
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        layer6_out = self.layer6(layer5_out)
        layer7_out = self.layer7(layer6_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()

        return layer7_out


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
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        layer6_out = self.layer6(layer5_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer6_out


class Server_LeakyreluNet_4_att(nn.Module):
    def __init__(self, n_hidden_2=256, n_hidden_3=128, n_hidden_4=64, n_hidden_5= 32, out_dim=2, mode_t_a='train'):
        super(Server_LeakyreluNet_4_att, self).__init__()
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.Tanh())
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),
                                    nn.Tanh())
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim),
                                   )
        self.mode_t_a = mode_t_a
    

        def forward(self, x):
            layer3_out = self.layer3(x1)
            layer4_out = self.layer4(layer3_out)
            layer5_out = self.layer5(layer4_out)
            layer6_out = self.layer6(layer5_out)
            if np.isnan(np.sum(x.data.cpu().numpy())):
                raise ValueError()
            return layer6_out




class Server_LeakyreluNet_3(nn.Module):
    def __init__(self, n_hidden_2=256, n_hidden_3=128, n_hidden_4=64, out_dim=2):
        super(Server_LeakyreluNet_3, self).__init__()
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.Tanh())
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, out_dim),
                                   )

    def forward(self, x1, x2):
        x= torch.cat([x1, x2], dim=1)
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer5_out


class Server_LeakyreluNet_2(nn.Module):
    def __init__(self, n_hidden_2=256, n_hidden_3=128, out_dim=2):
        super(Server_LeakyreluNet_2, self).__init__()
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim),
                                   )


    def forward(self, x1, x2):
        x= torch.cat([x1, x2], dim=1)
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)

        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer4_out


class Server_LeakyreluNet_1(nn.Module):
    def __init__(self, n_hidden_2=256, n_hidden_3=128, out_dim=2):
        super(Server_LeakyreluNet_1, self).__init__()
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),)


    def forward(self, x1, x2):
        x= torch.cat([x1, x2], dim=1)
        layer3_out = self.layer3(x)

        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer3_out


class Server_LeakyreluNet_1_attack(nn.Module):
    def __init__(self, n_hidden_2=256,  out_dim=2):
        super(Server_LeakyreluNet_1_attack, self).__init__()

        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim),
                                   nn.LeakyReLU())

    def forward(self, x):
        layer3_out = self.layer3(x)

        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()
        return layer3_out



class Server_LeakyreluNet_2_attack(nn.Module):
    def __init__(self, n_hidden_2=256, n_hidden_3=128, n_hidden_4=64, out_dim=2):
        super(Server_LeakyreluNet_2_attack, self).__init__()
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim),
                                  )

    def forward(self, x):
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)
        # layer5_out = self.layer5(layer4_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()

        return layer4_out


class Server_LeakyreluNet_3_attack(nn.Module):
    def __init__(self, n_hidden_2=256, n_hidden_3=128, n_hidden_4=64, out_dim=2):
        super(Server_LeakyreluNet_3_attack, self).__init__()
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.Tanh())
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, out_dim),
                                   )


    def forward(self, x):
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()

        return layer5_out



class Server_LeakyreluNet_4_attack(nn.Module):
    def __init__(self, n_hidden_2=256, n_hidden_3=128, n_hidden_4=64, n_hidden_5= 32, out_dim1=2, out_dim2=2):
        super(Server_LeakyreluNet_4_attack, self).__init__()
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.Tanh())
        # for main task
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),
                                    nn.Tanh())
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim1),
                                   )
        # for property task
        self.layer7 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5*3),
                                 nn.Tanh())
        self.layer8 = nn.Sequential(nn.Linear(n_hidden_5*3, n_hidden_5*2),
                                 nn.Tanh())
        self.layer9 = nn.Sequential(nn.Linear(n_hidden_5*2, n_hidden_5),
                                 nn.Tanh())
        self.layer10 = nn.Sequential(nn.Linear(n_hidden_5, out_dim2),
                                   )


    def forward(self, x):
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)
        # for main task
        layer5_out = self.layer5(layer4_out)
        layer6_out = self.layer6(layer5_out)
        # for property task
        layer7_out = self.layer7(layer4_out)
        layer8_out = self.layer8(layer7_out)
        layer9_out = self.layer9(layer8_out)
        layer10_out = self.layer10(layer9_out)


        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()

        return layer6_out, layer10_out


class Server_LeakyreluNet_6_attack(nn.Module):
    def __init__(self, n_hidden_2=256, n_hidden_3=128, n_hidden_4=64, n_hidden_5= 32,n_hidden_6= 32,  n_hidden_7= 32, out_dim=2):
        super(Server_LeakyreluNet_6_attack, self).__init__()
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3),
                                    nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4),
                                    nn.Tanh())
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5),
                                    nn.Tanh())
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, n_hidden_6),
                                   nn.Tanh())
        self.layer7 = nn.Sequential(nn.Linear(n_hidden_6, n_hidden_7),
                                   nn.Tanh())
        self.layer8 = nn.Sequential(nn.Linear(n_hidden_7, out_dim),
                                )


    def forward(self, x):
        layer3_out = self.layer3(x)
        layer4_out = self.layer4(layer3_out)
        layer5_out = self.layer5(layer4_out)
        layer6_out = self.layer6(layer5_out)
        layer7_out = self.layer7(layer6_out)
        layer8_out = self.layer8(layer7_out)
        if np.isnan(np.sum(x.data.cpu().numpy())):
            raise ValueError()

        return layer8_out

