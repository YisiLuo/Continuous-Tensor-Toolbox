import torch
from torch import nn, optim 
dtype = torch.cuda.FloatTensor
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io
import math
from skimage.metrics import peak_signal_noise_ratio

data_all =["data/om1"]
c_all = ["case2"]
        
################### 
# Here are the hyperparameters. 
max_iter = 5001 
w_decay = 0.1 
lr_real = 0.0001 
phi = 5*10e-6
mu = 1.2
gamma = 0.1
down = 4
omega = 2
################### 

class soft(nn.Module):
    def __init__(self):
        super(soft, self).__init__()
    
    def forward(self, x, lam):
        x_abs = x.abs()-lam
        zeros = x_abs - x_abs
        n_sub = torch.max(x_abs, zeros)
        x_out = torch.mul(torch.sign(x), n_sub)
        return x_out

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=omega):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(5 / self.in_features) / self.omega_0, 
                                             np.sqrt(5 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Network(nn.Module):
    def __init__(self, r_1,r_2,r_3):
        super(Network, self).__init__()
        
        self.U_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_1))
        
        self.V_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_2))
        
        self.W_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True),
                                   SineLayer(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_3))
    def forward(self, centre, U_input, V_input, W_input):
        U = self.U_net(U_input)
        V = self.V_net(V_input)
        W = self.W_net(W_input)
        
        centre = centre.permute(1,2,0) 
        centre = centre @ U.t()
        centre = centre.permute(2,1,0)
        centre = centre @ V.t()
        centre = centre.permute(0,2,1)
        centre = centre @ W.t()
        return centre

for data in data_all:
    for c in c_all:
        soft_thres=soft()

        file_name = data+c+'.mat'
        mat = scipy.io.loadmat(file_name)
        X_np = mat["Nhsi"]
        X = torch.from_numpy(X_np).type(dtype).cuda()
        [n_1,n_2,n_3] = X.shape
        
        mid_channel = n_2 
        r_1 = int(n_1/down) 
        r_2 = int(n_2/down) 
        r_3 = int(n_3/down) 
        
        file_name = data+'gt.mat'
        mat = scipy.io.loadmat(file_name)
        gt_np = mat["Ohsi"]
        gt = torch.from_numpy(gt_np).type(dtype).cuda()
    
        mask = torch.ones(X.shape).type(dtype)
        mask[X == 0] = 0 
        X[mask == 0] = 0
        
        centre = torch.Tensor(r_1,r_2,r_3).type(dtype)
        S = torch.Tensor(n_1,n_2,n_3).type(dtype)
        stdv = 1 / math.sqrt(centre.size(0))
        centre.data.uniform_(-stdv, stdv)
        
        U_input = torch.from_numpy(np.array(range(1,n_1+1))).reshape(n_1,1).type(dtype)
        V_input = torch.from_numpy(np.array(range(1,n_2+1))).reshape(n_2,1).type(dtype)
        W_input = torch.from_numpy(np.array(range(1,n_3+1))).reshape(n_3,1).type(dtype)

        model = Network(r_1,r_2,r_3).type(dtype)
        params = []
        params += [x for x in model.parameters()]
        centre.requires_grad=True
        params += [centre]
        optimizier = optim.Adam(params, lr=lr_real, weight_decay=w_decay) 
        
        ps_best = 0
        for iter in range(max_iter):
            X_Out = model(centre, U_input, V_input, W_input)
            if iter == 0:
                X_Out_exp = X_Out.detach()
                D = torch.zeros([X.shape[0],X.shape[1],X.shape[2]]).type(dtype)
                S = (X-X_Out.clone().detach()).type(dtype)
                V = S.clone().detach().type(dtype)
                
            V = soft_thres(S + D / mu, gamma / mu)
            S = (2*X - 2 * X_Out.clone().detach()+ mu * V-D)/(2+mu)
            
            loss = torch.norm(X*mask-X_Out*mask-S*mask,2)
            loss = loss + phi*torch.norm(X_Out[1:,:,:]-X_Out[:-1,:,:],1) 
            loss = loss + phi*torch.norm(X_Out[:,1:,:]-X_Out[:,:-1,:],1) 
            
            optimizier.zero_grad()
            loss.backward(retain_graph=True)
            optimizier.step()
            D = (D + mu * (S  - V)).clone().detach()
            
            if iter % 100 == 0:
                ps = peak_signal_noise_ratio(np.clip(gt.cpu().detach().numpy(),0,1),
                                             X_Out.cpu().detach().numpy())
                print('iteration:',iter,'PSNR',ps)
                plt.figure(figsize=(15,45))
                show = [15,25,30] 
                plt.subplot(121)
                plt.imshow(np.clip(np.stack((gt[:,:,show[0]].cpu().detach().numpy(),
                                     gt[:,:,show[1]].cpu().detach().numpy(),
                                     gt[:,:,show[2]].cpu().detach().numpy()),2),0,1))
                plt.title('gt')
        
                plt.subplot(122)
                plt.imshow(np.clip(np.stack((X_Out[:,:,show[0]].cpu().detach().numpy(),
                                     X_Out[:,:,show[1]].cpu().detach().numpy(),
                                     X_Out[:,:,show[2]].cpu().detach().numpy()),2),0,1))
                plt.title('out')
                plt.show()