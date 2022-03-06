__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/11/03 20:46:23"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional

class RealNVP(nn.Module):
    def __init__(self, masks, hidden_dim):
        super(RealNVP, self).__init__()
        self.hidden_dim = hidden_dim

        self.masks = nn.ParameterList()
        for i in range(len(masks)):
            self.masks.append(nn.Parameter(torch.Tensor(masks[i][1]), requires_grad = False))

        self.transforms = nn.ModuleList()
        for i in range(len(masks)):
            if masks[i][0] == "Affine Coupling":
                self.transforms.append(Affine_Coupling(self.masks[i], self.hidden_dim))
            elif masks[i][0] == "Affine":
                self.transforms.append(Affine(self.masks[i]))
            elif masks[i][0] == "Scale":
                self.transforms.append(Scale(self.masks[i], masks[i][2], masks[i][3]))                        
    def forward(self, x):
        y = x
        logdet_tot = 0
        for i in range(len(self.transforms)):
            y, logdet = self.transforms[i](y)
            logdet_tot = logdet_tot + logdet
        
        return y, logdet_tot

    def inverse(self, y):
        x = y
        logdet_tot = 0
        for i in range(len(self.transforms)-1, -1, -1):
            x, logdet = self.transforms[i].inverse(x)
            logdet_tot = logdet_tot + logdet
            
        return x, logdet_tot

class Scale(nn.Module):
    def __init__(self, mask, low, high):
        super(Scale, self).__init__()
        self.mask = nn.Parameter(torch.Tensor(mask), requires_grad = False)
        self.low = nn.Parameter(torch.Tensor([low]), requires_grad = False)
        self.high = nn.Parameter(torch.Tensor([high]), requires_grad = False)
        
    def forward(self, x):
        y = self.mask*x + (1-self.mask)*(self.low + (self.high - self.low)*torch.sigmoid(x))
        x = (1-self.mask)*x
        logdet = torch.sum((1-self.mask)*(functional.logsigmoid(x) + functional.logsigmoid(-x) + torch.log(self.high - self.low)), -1)        
                
        return y, logdet

    def inverse(self, y):
        x = (1-self.mask)*(y-self.low)/(self.high - self.low) + self.mask*0.5
        x = self.mask*y + (1-self.mask)*torch.log(x/(1-x))

        p = (1-self.mask)*(y-self.low)/(self.high - self.low) + self.mask*0.5
        logdet = torch.sum((1-self.mask)*(-torch.log(p)-torch.log(1-p)-torch.log(self.high - self.low)), -1)        
        return x, logdet
    

class Affine(nn.Module):
    def __init__(self, mask):
        super(Affine, self).__init__()
        self.input_dim = len(mask)
        self.mask = nn.Parameter(torch.Tensor(mask), requires_grad = False)

        self.scale = nn.Parameter(torch.Tensor(self.input_dim))
        init.normal_(self.scale)

        self.translation = nn.Parameter(torch.Tensor(self.input_dim))
        init.normal_(self.translation)
        
    def forward(self, x):
        y = self.mask*x + (1-self.mask)*(x*torch.exp(self.scale) + self.translation)
        logdet = torch.sum((1 - self.mask)*self.scale, -1)
        return y, logdet

    def inverse(self, y):
        x = self.mask*y + (1-self.mask)*((y - self.translation)*torch.exp(-self.scale))
        logdet = torch.sum((1 - self.mask)*(-self.scale), -1)
        return x, logdet
        
    
class Affine_Coupling(nn.Module):
    def __init__(self, mask, hidden_dim):
        super(Affine_Coupling, self).__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim

        ## mask to seperate positions that do not change and positions that change.
        ## mask[i] = 1 means the ith position does not change.
        self.mask = nn.Parameter(mask, requires_grad = False)

        ## layers used to compute scale in affine transformation
        self.scale_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.scale_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_fc3 = nn.Linear(self.hidden_dim, self.input_dim)
        self.scale = nn.Parameter(torch.Tensor(self.input_dim))
        init.normal_(self.scale)

        ## layers used to compute translation in affine transformation 
        self.translation_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.translation_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.translation_fc3 = nn.Linear(self.hidden_dim, self.input_dim)

    def _compute_scale(self, x):
        # s = torch.relu(self.scale_fc1(x*self.mask))
        # s = torch.relu(self.scale_fc2(s))
        # s = torch.relu(self.scale_fc3(s)) * self.scale

        x = x*self.mask
        s = torch.tanh(self.scale_fc1(x))
        s = torch.tanh(self.scale_fc2(s) + s)
        s = torch.tanh(self.scale_fc3(s)) + x
        s = s * self.scale
        
        return s

    def _compute_translation(self, x):
        # t = torch.relu(self.translation_fc1(x*self.mask))
        # t = torch.relu(self.translation_fc2(t))
        # t = self.translation_fc3(t)

        x = x*self.mask
        t = torch.tanh(self.translation_fc1(x))
        t = torch.tanh(self.translation_fc2(t) + t)
        t = self.translation_fc3(t) + x
        
        return t
    
    def forward(self, x):
        s = self._compute_scale(x)
        t = self._compute_translation(x)
        
        y = self.mask*x + (1-self.mask)*(x*torch.exp(s) + t)        
        logdet = torch.sum((1 - self.mask)*s, -1)
        
        return y, logdet

    def inverse(self, y):
        s = self._compute_scale(y)
        t = self._compute_translation(y)
                
        x = self.mask*y + (1-self.mask)*((y - t)*torch.exp(-s))
        logdet = torch.sum((1 - self.mask)*(-s), -1)
        
        return x, logdet
    
    
class RealNVP_2D(nn.Module):    
    def __init__(self, masks, hidden_dim):
        super(RealNVP_2D, self).__init__()
        self.hidden_dim = hidden_dim        
        self.masks = nn.ParameterList(
            [nn.Parameter(torch.Tensor(m),requires_grad = False)
             for m in masks])

        self.affine_couplings = nn.ModuleList(
            [Affine_Coupling(self.masks[i], self.hidden_dim)
             for i in range(len(self.masks))])
        
    def forward(self, x):
        y = x
        logdet_tot = 0
        for i in range(len(self.affine_couplings)):
            y, logdet = self.affine_couplings[i](y)
            logdet_tot = logdet_tot + logdet

        logdet = torch.sum(torch.log(torch.abs(4*(1-(torch.tanh(y))**2))), -1)        
        y = 4*torch.tanh(y)
        logdet_tot = logdet_tot + logdet
        
        return y, logdet_tot

    def inverse(self, y):
        x = y        
        logdet_tot = 0

        logdet = torch.sum(torch.log(torch.abs(1.0/4.0* 1/(1-(x/4)**2))), -1)
        x  = 0.5*torch.log((1+x/4)/(1-x/4))
        logdet_tot = logdet_tot + logdet
        
        for i in range(len(self.affine_couplings)-1, -1, -1):
            x, logdet = self.affine_couplings[i].inverse(x)
            logdet_tot = logdet_tot + logdet
            
        return x, logdet_tot
