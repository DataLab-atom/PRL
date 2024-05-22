"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import autocast

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class hnet(nn.Module):
    """classify_net Hypernetwork"""

    def __init__(
        self,
        out_dim=10,
        target_net_in_features=50,
        ray_hidden_dim = 128,
        ray_out_dim = 16,
        alpha = 1.0,
        n_hidden=1,
        n_expert=3,
        use_norm = True
    ):
        super(hnet,self).__init__()
        # self.n_conv_layers = n_conv_layers
        self.n_hidden = n_hidden
        self.n_expert = n_expert
        self.target_net_wieight_shape = (target_net_in_features,out_dim)
        print(self.target_net_wieight_shape)
        self.use_norm = use_norm

        self.ray_mlp = nn.Sequential(
            nn.Linear(3, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_out_dim),
        )

        self.backbone_linears_weights = nn.ModuleList([nn.Linear(ray_out_dim, target_net_in_features * out_dim) for _ in range(n_expert)])
        
        if not use_norm:
            self.backbone_linears_bias = nn.Linear(ray_out_dim, out_dim * n_expert)
        
        self.alpha = alpha
        if self.alpha <= 0:
            self.alpha = torch.empty(1,).uniform_(0.0, 1.0)
        
        self.ray = torch.nn.Parameter(torch.rand(3), requires_grad=True)

    def forward(self,_ray=None):
        if _ray == None:
            if self.training:
                _ray = torch.from_numpy(
                    np.random.dirichlet([self.alpha]*3, 1).astype(np.float32).flatten()
                ).to(self.ray.device)
                _ray = (_ray - _ray.min())/(_ray.max() - _ray.min())
            else :
                #_ray = torch.tensor([0,0,1],device=self.ray.device,requires_grad=False).float()
                _ray = torch.ones_like(self.ray)/3
        
        ray = self.ray * _ray
        ray = (ray - ray.min())/(ray.max() - ray.min())
        #ray = torch.nn.functional.softmax(ray,dim=0)
        features = self.ray_mlp(ray)
        out_dict = {}
        
        for j in range(self.n_expert):
            weights = self.backbone_linears_weights[j](features)
            out_dict[f"backbone.linears{j}.weights"] = weights.reshape(self.target_net_wieight_shape)
            
        if not self.use_norm:
            biass = self.backbone_linears_bias(features)
            biass = torch.chunk(biass,self.n_expert,-1)
            for j in range(self.n_expert):
                out_dict[f"backbone.linears{j}.bias"] = biass[j].flatten()

        return out_dict

class ResNet(nn.Module):

    def __init__(self, block, layers, num_experts, dropout=None, 
                 num_classes=1000, use_norm=False, reduce_dimension=False, 
                 layer3_output_dim=None, layer4_output_dim=None, share_layer3=False, 
                 returns_feat=False, s=30, use_hnet=False,hnet_feature=False):
        self.inplanes = 64
        self.use_norm = use_norm
        self.num_experts = num_experts
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.inplanes = self.next_inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.inplanes = self.next_inplanes

        self.share_layer3 = share_layer3

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 192
            else:
                layer3_output_dim = 256

        if layer4_output_dim is None:
            if reduce_dimension:
                layer4_output_dim = 384
            else:
                layer4_output_dim = 512

        if self.share_layer3:
            self.layer3 = self._make_layer(block, layer3_output_dim, layers[2], stride=2)
        else:
            self.layer3s = nn.ModuleList([self._make_layer(block, layer3_output_dim, layers[2], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.layer4s = nn.ModuleList([self._make_layer(block, layer4_output_dim, layers[3], stride=2) for _ in range(num_experts)])
        self.inplanes = self.next_inplanes
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        self.use_dropout = True if dropout else False

        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.use_hnet = use_hnet
        self.hnet_feature = hnet_feature
        if hnet_feature :
            assert self.use_hnet # use_hnet must set true befor hnet_feature
            hnet_out_dim = layer4_output_dim * block.expansion
            H_use_norm = False
        else :
            hnet_out_dim = num_classes
            H_use_norm = use_norm

        if self.use_hnet:
            self.hnet2linears = hnet(hnet_out_dim,layer4_output_dim * block.expansion, alpha=1.2, use_norm=H_use_norm)
        
        if (not self.use_hnet) or hnet_feature:
            if use_norm:
                self.linears = nn.ModuleList([NormedLinear(layer4_output_dim * block.expansion, num_classes) for _ in range(num_experts)])
            else:
                self.linears = nn.ModuleList([nn.Linear(layer4_output_dim * block.expansion, num_classes) for _ in range(num_experts)])
                s = 1
        
        self.s = s
        self.returns_feat = returns_feat

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.next_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.next_inplanes, planes))

        return nn.Sequential(*layers)

    def _separate_part(self, x, ind):
        if not self.share_layer3:
            x = (self.layer3s[ind])(x)
        x = (self.layer4s[ind])(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)

        if self.use_dropout:
            x = self.dropout(x)

        self.feat.append(x)
        #x = (self.linears[ind])(x)
        #x = x * self.s
        return x

    
    def forward(self, x,ray = None):
        with autocast():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            if self.share_layer3:
                x = self.layer3(x)

            outs = []
            self.feat = []
            for ind in range(self.num_experts):
                out_ind = self._separate_part(x, ind)
                if self.use_hnet and (not self.hnet_feature):
                    hnet_out_dict = self.hnet2linears(ray)
                    #--------------------------------------------------------------
                    if self.use_norm:
                        weight = hnet_out_dict[f"backbone.linears{ind}.weights"]
                        out_ind = F.normalize(out_ind, dim=1).mm(F.normalize(weight, dim=0))
                    else:
                        weight = hnet_out_dict[f"backbone.linears{ind}.weights"]
                        bias = hnet_out_dict[f"backbone.linears{ind}.bias"]
                        out_ind = F.linear(out_ind,weight,bias)
                    #-------------------------------------------------------------    
                elif self.hnet_feature:
                    #--------------------------------------------------------------
                    hnet_out_dict = self.hnet2linears(ray)
                    weight = hnet_out_dict[f"backbone.linears{ind}.weights"]
                    bias = hnet_out_dict[f"backbone.linears{ind}.bias"]
                    out_ind = F.relu(F.linear(out_ind,weight,bias))
                    out_ind = self.linears[ind](out_ind)
                    #-------------------------------------------------------------    
                else :
                    out_ind = self.linears[ind](out_ind)

                out_ind = out_ind * self.s
                outs.append(out_ind)  
        final_out = torch.stack(outs, dim=1).mean(dim=1)

        if self.returns_feat:
            return {
                "output": final_out, 
                "feat": torch.stack(self.feat, dim=1),
                "logits": torch.stack(outs, dim=1)
            }
        else:
            return final_out
