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
from Generator.model import *
from torch.nn.utils import clip_grad_norm_ 
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity, mean_squared_error
from skimage.color import rgb2ycbcr
import skimage.io as io
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
    parser.add_argument('--attack_batch_size', default=8, type=int, help='attack_batch_size')
    parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
    parser.add_argument('--epochs', default=100, type=int, help='epochs')
    parser.add_argument('--num_cutlayer', default=1000, type=int, help='num_cutlayer')
    parser.add_argument('--pruning', default=0, type=float, help='pruning')
    parser.add_argument('--number_client', default=2, type=int, help='number_client')
    parser.add_argument('--Iteration', default=60, type=int, help='Iteration')
    parser.add_argument('--level2', default=1, type=int, help='level2')
    parser.add_argument('--level1', default=1, type=int, help='level1')
    parser.add_argument('--num_shadow', default=1, type=int, help='num_shadow')
    return parser.parse_args(argv[1:])


# Define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TVLoss(nn.Module):
	def __init__(self,TVLoss_weight=1):
		super(TVLoss,self).__init__()
		self.TVLoss_weight = TVLoss_weight

	def forward(self,x):
		batch_size = x.size()[0]
		h_x = x.size()[2]
		w_x = x.size()[3]
		count_h = self._tensor_size(x[:,:,1:,:])
		count_w = self._tensor_size(x[:,:,:,1:])
		h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
		w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
		return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
			
	@staticmethod
	def _tensor_size(t):
			return t.size()[1]*t.size()[2]*t.size()[3]

tv_loss = TVLoss()



if __name__ == '__main__':
    print('Start training')
    args = parse_args()
    dataset2 = args.dataset2
    dataset1 = args.dataset1
    model = args.model
    batch_size = args.batch_size
    attack_batch_size = args.attack_batch_size
    epochs = args.epochs
    lr = args.lr
    acti = args.acti
    attributes = args.attributes
    attack_label = args.attack_label
    num_cutlayer = args.num_cutlayer
    pruning = args.pruning
    number_client = args.number_client
    Iteration = args.Iteration
    level1 =args.level1
    level2 =args.level2
    num_shadow = args.num_shadow
    time_start_load_everything = time.time()

    if dataset2 == 'utkface':
        attributes = 'race_gender'

    if dataset2 == 'celeba':
        attributes = "attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr_attr"

    # Define record path
    save_path = f'Results_DR/{dataset1}-{dataset2}/{model}/level{level1}-level{level2}/num_shadow{num_shadow}/c{num_cutlayer}/client{number_client}/p{pruning}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = open(f'Results_DR/{dataset1}-{dataset2}/{model}/level{level1}-level{level2}/num_shadow{num_shadow}/c{num_cutlayer}/client{number_client}/p{pruning}/c{num_cutlayer}.txt', 'w+')
    

    ### Load data
    root_path = '.'
    data_path = os.path.join(root_path, '../data').replace('\\', '/')
    train_data, test_data, shadow_data, num_classes1, num_classes2, channel, hideen = load_data(args.dataset2, args.attack_label,
                                                                                   args.attributes, data_path,
                                                                                   batch_size)
    # Define model
    if model == 'resnet':
        client_model_1 = ResNet18(level=level1, hideen1=hideen, num_classes=num_cutlayer).to(device)
        client_model_2 = ResNet18(level=level1, hideen1=hideen, num_classes=num_cutlayer).to(device)
        # Define server-side model
        server_model = Server_ResNet(hideen2=num_cutlayer * 2, hideen3=256, hideen4=128, hideen5=64,
                                     num_classes=num_classes1).to(device)
        save_path1 = f'Results/{dataset1}/{model}/level{level1}/client{number_client}/p{pruning}/client1_c{num_cutlayer}.pth'
        save_path2 = f'Results/{dataset1}/{model}/level{level1}/client{number_client}/p{pruning}/client2_c{num_cutlayer}.pth'
        save_path3 = f'Results/{dataset1}/{model}/level{level1}/client{number_client}/p{pruning}/server_c{num_cutlayer}.pth'
        client_model_1 = torch.load(save_path1)
        client_model_2 = torch.load(save_path2)
        server_model = torch.load(save_path3)
    
        client_model_fake_1 = ResNet18(level=level2, hideen1=hideen, num_classes=num_cutlayer).to(device)
        save_path = f'Results_MS/{dataset1}-{dataset2}/{model}/level{level1}-level{level2}/num_shadow{num_shadow}/c{num_cutlayer}/client{number_client}/p{pruning}/client1_fake_c{num_cutlayer}.pth'
        client_model_fake_1 = torch.load(save_path)


    # Define criterion
    criterion = nn.CrossEntropyLoss()

    test_true = []
    test_fake = []
    train_loss = []
    test_acc = []

    # start training
    num_exp = 1

    x1_true = np.load(f'Results/{dataset1}/{model}/level{level1}/client{number_client}/p{pruning}/X1_c{num_cutlayer}.npy')
    fx1_true = np.load(f'Results/{dataset1}/{model}/level{level1}/client{number_client}/p{pruning}/fx1_c{num_cutlayer}.npy')

    x1_true = torch.tensor(x1_true).to(device) 
    fx1_true = torch.tensor(fx1_true).to(device) 


    history = []
    P_tv = 0.1
    plot_num = 30
    tp = transforms.Compose([transforms.ToPILImage()])
    
    for idx_exp in range(num_exp):
        g_in = 128
        G_ran_in = torch.randn(attack_batch_size, g_in).to(device)# initialize GRNN input
        Gnet1 = Generator(channel=3, shape_img=32, batchsize=attack_batch_size, g_in=g_in, iters=0).to(device)
        G_optimizer = torch.optim.RMSprop(Gnet1.parameters(), lr=0.0001, momentum=0.9)

        for iters in range(Iteration):

            Gout = Gnet1(G_ran_in, attack_batch_size, 0) # produce recovered data
            Gout = Gout.to(device)

            fx1_fake = client_model_fake_1(Gout)

            pruning_fx1_fake = torch.ones_like(fx1_fake).to(device)
            _, zis = torch.sort(fx1_fake, descending=False)
            zi = zis[:, :int(pruning*(fx1_fake.shape[-1]))]
            for i in range(len(zi)):
                pruning_fx1_fake[i, zi[i]] =0

            loss1 = (((fx1_fake * pruning_fx1_fake) - fx1_true[:attack_batch_size,]) ** 2).sum()
            # loss = loss1 + x_value2(Gout[:,:,:,16:]) + P_tv* tv_loss(Gout[:,:,:,16:])
            loss = loss1 + P_tv* tv_loss(Gout[:,:,:,16:])


            G_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(Gnet1.parameters(), max_norm=5, norm_type=2)
            G_optimizer.step()


            if iters % int(Iteration / plot_num) == 0:
                history.append([tp(Gout[imidx].detach().cpu()) for imidx in range(attack_batch_size)])
                


        save_G1 = f'Results_DR/{dataset1}-{dataset2}/{model}/level{level1}-level{level2}/num_shadow{num_shadow}/c{num_cutlayer}/client{number_client}/p{pruning}/G1_c{num_cutlayer}.pth'
        torch.save(Gnet1, save_G1)

        # visualization
        for imidx in range(attack_batch_size):
            plt.figure(figsize=(12, 8))
            plt.subplot(plot_num//10, 10, 1)
            plt.imshow(tp(x1_true[imidx].cpu()))
            for i in range(min(len(history), plot_num-1)):
                plt.subplot(plot_num//10, 10, i + 2)
                plt.imshow(history[i][imidx])
                plt.axis('off')

            if True:
                save_path = f'Results_DR/{dataset1}-{dataset2}/{model}/level{level1}-level{level2}/num_shadow{num_shadow}/c{num_cutlayer}/client{number_client}/p{pruning}/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                true_path = os.path.join(save_path, f'true_data/')
                fake_path = os.path.join(save_path, f'fake_data/')
                if not os.path.exists(true_path) or not os.path.exists(fake_path):
                    os.makedirs(true_path)
                    os.makedirs(fake_path)
                tp(x1_true[imidx].cpu()).save(os.path.join(true_path, f'{imidx}.png'))
                history[-1][imidx].save(os.path.join(fake_path, f'{imidx}.png'))
                plt.savefig(save_path + '/exp:%03d-imidx:%02d.png' % (idx_exp, imidx))
                plt.close()

    del loss1,  loss, Gout, history
    print('=======================================================', file=filename)

   





















