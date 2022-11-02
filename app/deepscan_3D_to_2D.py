
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data.sampler import Sampler, RandomSampler, WeightedRandomSampler, SequentialSampler
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader



from torch.autograd import Variable, Function
from torch.nn import  ModuleList

from collections import OrderedDict

import numpy as np



from torch.optim.lr_scheduler import CosineAnnealingLR

INSTANCE_AFFINE=False
USE_ATTENTION=True



def reduce_3d_depth (in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad3d((1,1,1,1,0,0))),
            ("conv1", nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm3d(out_channel)),
            ("relu1", nn.ReLU()),
            #("dropout", nn.Dropout(p=0.2))
    ]))
    return layer

def down_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm2d(out_channel)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(p=0.0)),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.InstanceNorm2d(out_channel)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(p=0.0))]))
    return layer

def up_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm2d(out_channel)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(p=0.0)),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.InstanceNorm2d(out_channel)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(p=0.0))]))
    return layer

class DilatedDenseUnit(nn.Module):
    def __init__(self, in_channel, growth_rate , kernel_size, dilation):
        super(DilatedDenseUnit,self).__init__()
        self.layer = nn.Sequential(OrderedDict([
            ("bn1", nn.InstanceNorm2d(in_channel)),
            ("relu1", nn.ReLU()),
            ("pad1", nn.ReplicationPad2d(dilation)),
            ("conv1", nn.Conv2d(in_channel, growth_rate, kernel_size=kernel_size, dilation = dilation,padding=0)),
            ("dropout", nn.Dropout(p=0.0))]))
    def forward(self, x):
        out = x
        out = self.layer(out)
        out = concatenate(x, out)
        return out
    
class AttentionModule(nn.Module):
    def __init__(self, in_channel , intermediate_channel, out_channel, kernel_size=3):
        super(AttentionModule,self).__init__()
        self.layer = nn.Sequential(OrderedDict([
            ("bn1", nn.InstanceNorm2d(in_channel)),
            ("relu1", nn.ReLU()),
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, intermediate_channel, kernel_size=kernel_size,padding=0)),
            ("bn2", nn.InstanceNorm2d(intermediate_channel)),
            ("relu2", nn.ReLU()),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(intermediate_channel, out_channel, kernel_size=kernel_size,padding=0)),
            ("sigmoid", nn.Sigmoid())]))
    def forward(self, x):
        out = x
        out = self.layer(out)
        out = x * out
        return out
    
    

def center_crop(layer, target_size):
    _, _, layer_width, layer_height = layer.size()
    start = (layer_width - target_size) // 2
    crop = layer[:, :, start:(start + target_size), start:(start + target_size)]
    return crop

def concatenate(link, layer):
    #crop = center_crop(link, layer.size()[2])
    concat = torch.cat([link, layer], 1)
    return concat

def dense_atrous_bottleneck(in_channel, growth_rate = 12, depth = [4,4,4,4]):
    layer_dict = OrderedDict()
    for idx, growth_steps in enumerate(depth):
        dilation_rate = 2**idx
        for y in range(growth_steps):
            layer_dict["dilated_{}_{}".format(dilation_rate,y)] = DilatedDenseUnit(in_channel, 
                                                                        growth_rate, 
                                                                        kernel_size=3, 
                                                                        dilation = dilation_rate)
            in_channel = in_channel + growth_rate
        layer_dict["attention_{}".format(dilation_rate)] = AttentionModule(in_channel, in_channel//4, in_channel)
    return nn.Sequential(layer_dict), in_channel




class DeepSCAN_3D_to_2D_orig(nn.Module):
    def __init__(self, depth, channels_in = 1, 
                 channels_2d_to_3d=32, channels=32, output_channels = 1, slices=5, 
                 dilated_layers = [4,4,4,4],
                growth_rate = 12):
        super(DeepSCAN_3D_to_2D, self).__init__()        
        self.depth = depth
        self.slices = slices
        self.channels_in = channels_in

        
        self.depth_reducing_layers = ModuleList([reduce_3d_depth(in_channel, channels_2d_to_3d, kernel_size=3, padding=0)
                                                 for in_channel in [channels_in]+[channels_2d_to_3d]*(slices//2 - 1)])
        
        
        self.down1 = down_layer(in_channel=channels_2d_to_3d, out_channel=channels, kernel_size=3, padding=0)
        self.max1 = nn.MaxPool2d(2)
        self.down_layers = ModuleList([down_layer(in_channel = channels*(2**i), 
                                  out_channel = channels * (2**(i+1)),
                                  kernel_size = 3,
                                  padding=0
                                 ) for i in range(self.depth)])
        self.max_layers = ModuleList([nn.MaxPool2d(2) for i in range(self.depth)])
        
        self.bottleneck, bottleneck_features  = dense_atrous_bottleneck(channels*2**self.depth, growth_rate = growth_rate, 
                                                                       depth = dilated_layers)
        
        self.upsampling_layers = ModuleList([nn.Sequential(OrderedDict([
                ("upsampling",nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)),
                ("pad", nn.ReplicationPad2d(1)),
                ("conv", nn.Conv2d(in_channels= bottleneck_features, 
                                   out_channels=bottleneck_features, 
                                   kernel_size=3, 
                                   padding=0))]))  for i in range(self.depth, -1, -1)])
        self.up_layers = ModuleList([up_layer(in_channel= bottleneck_features+ channels*(2**(i)), 
                                   out_channel=bottleneck_features, 
                                   kernel_size=3, 
                                   padding=0) for i in range(self.depth, -1, -1)])
        
        self.last = nn.Conv2d(in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1)

    def forward(self, x):
        #assert(x.shape[2]>=self.slices)
        #assert(x.shape[1]>=self.channels_in)

        #assert(x.shape[2]==self.slices or x.shape[0]==1)
        
        # down
        
        out = x
        
        for i in range(self.slices//2):
            out = self.depth_reducing_layers[i](out)
        
        out.transpose_(1, 2).contiguous()
        size = out.size()
        out = out.view((-1, size[2], size[3], size[4]))
        
        links = []
        out = self.down1(out)
        links.append(out)
        out = self.max1(out)
        
        for i in range(self.depth):
            out = self.down_layers[i](out)
            links.append(out)
            out = self.max_layers[i](out)
        


        
        
        out = self.bottleneck(out)

        
        links.reverse()

        # up
        
        for i in range(self.depth+1):

            out = self.upsampling_layers[i](out)

            out = concatenate(links[i], out)
            out = self.up_layers[i](out)

        pred = self.last(out)
        
        return pred

    def predict_3D(self, x, do_mirroring=False, mirror_axes=None, 
                    use_sliding_window=True, use_gaussian = True,
                    step_size = 1, patch_size=(5,192,192), batch_size = 2 
                     ):
        self.eval()
        with torch.no_grad():
            
            logit_total = []
             
          
            num_batches = x.shape[1]

            stack_depth = patch_size[0]

            padding = stack_depth//2

            input = torch.nn.ConstantPad3d((0,0,0,0,padding,padding),0)(x)
          
            slice = 0

            for idx in range(x.shape[1]//batch_size+1):

                batch_list = []

                for y in range(batch_size):
                    if slice == x.shape[1]:
                        break
                    batch_list.append(input[:,slice:slice+stack_depth])  
                    slice +=1
                if len(batch_list) ==0:
                    break
                batch_tensor = torch.stack(batch_list, 0)  

                logit = self.forward(batch_tensor).transpose(0,1)

                logit_total.append(logit)

              
            full_logit = torch.cat(logit_total, 1)
          

        
        return None, full_logit.detach().cpu().numpy()

def reduce_3d_depth (in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad3d((1,1,1,1,0,0))),
            ("conv1", nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm3d(out_channel, affine = INSTANCE_AFFINE)),
            ("relu1", nn.ReLU()),
            #("dropout", nn.Dropout(p=0.2))
    ]))
    return layer

def down_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm2d(out_channel, affine = INSTANCE_AFFINE)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(p=0.0)),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.InstanceNorm2d(out_channel, affine = INSTANCE_AFFINE)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(p=0.0))]))
    return layer

def up_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm2d(out_channel, affine = INSTANCE_AFFINE)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(p=0.0)),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.InstanceNorm2d(out_channel, affine = INSTANCE_AFFINE)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(p=0.0))]))
    return layer

class DilatedDenseUnit(nn.Module):
    def __init__(self, in_channel, growth_rate , kernel_size, dilation, 
        conv_op = nn.Conv2d, pad_op = nn.ReplicationPad2d, norm_op = nn.InstanceNorm2d, nonlin = nn.ReLU):
        super(DilatedDenseUnit,self).__init__()
        self.layer = nn.Sequential(OrderedDict([
            ("bn1", norm_op(in_channel, affine = INSTANCE_AFFINE)),
            ("relu1", nonlin()),
            ("pad1", pad_op(dilation)),
            ("conv1", conv_op(in_channel, growth_rate, kernel_size=kernel_size, dilation = dilation,padding=0)),
            ("dropout", nn.Dropout(p=0.0))]))
    def forward(self, x):
        out = x
        out = self.layer(out)
        out = concatenate(x, out)
        return out
    
class AttentionModule(nn.Module):
    def __init__(self, in_channel , intermediate_channel, out_channel, kernel_size=3, 
        conv_op = nn.Conv2d, pad_op = nn.ReplicationPad2d, norm_op = nn.InstanceNorm2d, nonlin = nn.ReLU):
        super(AttentionModule,self).__init__()
        self.layer = nn.Sequential(OrderedDict([
            ("bn1", norm_op(in_channel, affine = INSTANCE_AFFINE)),
            ("relu1", nonlin()),
            ("pad1", pad_op(1)),
            ("conv1", conv_op(in_channel, intermediate_channel, kernel_size=kernel_size,padding=0)),
            ("bn2", norm_op(intermediate_channel, affine = INSTANCE_AFFINE)),
            ("relu2", nonlin()),
            ("pad2", pad_op(1)),
            ("conv2", conv_op(intermediate_channel, out_channel, kernel_size=kernel_size,padding=0)),
            ("sigmoid", nn.Sigmoid())]))
    def forward(self, x):
        out = x
        out = self.layer(out)
        out = x * out
        return out
    
    

def center_crop(layer, target_size):
    _, _, layer_width, layer_height = layer.size()
    start = (layer_width - target_size) // 2
    crop = layer[:, :, start:(start + target_size), start:(start + target_size)]
    return crop

def concatenate(link, layer):
    #crop = center_crop(link, layer.size()[2])
    concat = torch.cat([link, layer], 1)
    return concat

def dense_atrous_bottleneck(in_channel, growth_rate = 12, depth = [4,4,4,4], dilation_rates = None, USE_ATTENTION = True):
    layer_dict = OrderedDict()
    if dilation_rates is not None:
        assert len(depth) == len(dilation_rates)
    for idx, growth_steps in enumerate(depth):
        if dilation_rates is None:
            dilation_rate = 2**idx
        else:
            dilation_rate = dilation_rates[idx]
        for y in range(growth_steps):
            layer_dict["dilated_{}_{}_{}".format(idx,dilation_rate,y)] = DilatedDenseUnit(in_channel, 
                                                                        growth_rate, 
                                                                        kernel_size=3, 
                                                                        dilation = dilation_rate)
            in_channel = in_channel + growth_rate
        
        if USE_ATTENTION:
            layer_dict["attention_{}".format(idx)] = AttentionModule(in_channel, in_channel//4, in_channel)
    return nn.Sequential(layer_dict), in_channel




class UNET_3D_to_2D(nn.Module):
    def __init__(self, depth, channels_in = 1, 
                 channels_2d_to_3d=32, channels=32, output_channels = 1, slices=5, 
                 dilated_layers = [4,4,4,4],
                growth_rate = 12):
        super(UNET_3D_to_2D, self).__init__()
        self.main_modules = []
        
        self.depth = depth
        self.slices = slices

        
        self.depth_reducing_layers = ModuleList([reduce_3d_depth(in_channel, channels_2d_to_3d, kernel_size=3, padding=0)
                                                 for in_channel in [channels_in]+[channels_2d_to_3d]*(slices//2 - 1)])
        
        
        self.down1 = down_layer(in_channel=channels_2d_to_3d, out_channel=channels, kernel_size=3, padding=0)
        self.main_modules.append(self.down1)
        self.max1 = nn.MaxPool2d(2)
        self.down_layers = ModuleList([down_layer(in_channel = channels*(2**i), 
                                  out_channel = channels * (2**(i+1)),
                                  kernel_size = 3,
                                  padding=0
                                 ) for i in range(self.depth)])
        self.main_modules.append(self.down_layers)
        self.max_layers = ModuleList([nn.MaxPool2d(2) for i in range(self.depth)])
        
        self.bottleneck, bottleneck_features  = dense_atrous_bottleneck(channels*2**self.depth, growth_rate = growth_rate, 
                                                                       depth = dilated_layers)
        self.main_modules.append(self.bottleneck)
        
        self.upsampling_layers = ModuleList([nn.Sequential(OrderedDict([
                ("upsampling",nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)), # since v0.4.0 align_corners= False is default, before was True
                ("pad", nn.ReplicationPad2d(1)),
                ("conv", nn.Conv2d(in_channels= bottleneck_features, 
                                   out_channels=bottleneck_features, 
                                   kernel_size=3, 
                                   padding=0))]))  for i in range(self.depth, -1, -1)])
        self.main_modules.append(self.upsampling_layers)
        self.up_layers = ModuleList([up_layer(in_channel= bottleneck_features+ channels*(2**(i)), 
                                   out_channel=bottleneck_features, 
                                   kernel_size=3, 
                                   padding=0) for i in range(self.depth, -1, -1)])
        
        self.main_modules.append(self.up_layers)
        self.last = nn.Conv2d(in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1)
        self.main_modules.append(self.last)
        
        self.logvar = nn.Conv2d(in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1)

    def forward(self, x):
        
        
        # down
        
        out = x
        
        for i in range(self.slices//2):
            out = self.depth_reducing_layers[i](out)
        
        out.transpose_(1, 2).contiguous()
        size = out.size()
        out = out.view((-1, size[2], size[3], size[4]))
        
        links = []
        out = self.down1(out)
        links.append(out)
        out = self.max1(out)
        
        for i in range(self.depth):
            out = self.down_layers[i](out)
            links.append(out)
            out = self.max_layers[i](out)
        


        
        
        out = self.bottleneck(out)

        
        links.reverse()

        # up
        
        for i in range(self.depth+1):

            out = self.upsampling_layers[i](out)

            out = concatenate(links[i], out)
            out = self.up_layers[i](out)

        pred = self.last(out)
    


        return pred

    def predict_3D(self, x, do_mirroring=False, mirror_axes=None, 
                    use_sliding_window=True, use_gaussian = True,
                    step_size = 1, patch_size=(5,192,192), batch_size = 2 
                     ):
        self.eval()
        with torch.no_grad():
            
            logit_total = []
             
          
            num_batches = x.shape[1]

            stack_depth = patch_size[0]

            padding = stack_depth//2

            input = torch.nn.ConstantPad3d((0,0,0,0,padding,padding),0)(x)
          
            slice = 0

            for idx in range(x.shape[1]//batch_size+1):

                batch_list = []

                for y in range(batch_size):
                    if slice == x.shape[1]:
                        break
                    batch_list.append(input[:,slice:slice+stack_depth])  
                    slice +=1
                if len(batch_list) ==0:
                    break
                batch_tensor = torch.stack(batch_list, 0)  

                logit = self.forward(batch_tensor).transpose(0,1)

                logit_total.append(logit)

              
            full_logit = torch.cat(logit_total, 1)
          

        
        return None, full_logit.detach().cpu().numpy()


class NET_3D_to_2D(nn.Module):
    def __init__(self, depth_reducing_net, net_2D, net_predict = None):
        super(NET_3D_to_2D, self).__init__()
        self.depth_reducing_net = depth_reducing_net
        self.net_2D = net_2D
        self.net_predict = net_predict

    def forward(self, x):
        
        
        # down
        
        out = x
        
        out = self.depth_reducing_net(out)
        
        out.transpose_(1, 2).contiguous()
        size = out.size()
        out = out.view((-1, size[2], size[3], size[4]))
        
        out = self.net_2D(out)

        if self.net_predict is not None:
            out = self.net_predict(out)

        return out

    def predict_3D(self, x, do_mirroring=False, mirror_axes=None, 
                    use_sliding_window=True, use_gaussian = True,
                    step_size = 1, patch_size=(5,192,192), batch_size = 2 
                     ):
        self.eval()
        with torch.no_grad():
            
            logit_total = []
             
          
            num_batches = x.shape[1]

            stack_depth = patch_size[0]

            padding = stack_depth//2

            input = torch.nn.ConstantPad3d((0,0,0,0,padding,padding),0)(x)
          
            slice = 0

            for idx in range(x.shape[1]//batch_size+1):

                batch_list = []

                for y in range(batch_size):
                    if slice == x.shape[1]:
                        break
                    batch_list.append(input[:,slice:slice+stack_depth])  
                    slice +=1
                if len(batch_list) ==0:
                    break
                batch_tensor = torch.stack(batch_list, 0)  

                logit = self.forward(batch_tensor).transpose(0,1)

                logit_total.append(logit)

              
            full_logit = torch.cat(logit_total, 1)
          

        
        return None, full_logit.detach().cpu().numpy()


class Depth_reduce_simple(nn.Module):
    def __init__(self, channels_in, channels_2d_to_3d=32, slices=5):
        super(Depth_reduce_simple, self).__init__()
        if slices == 1:
            in_channels = []
        else:
            in_channels = [channels_in]+[channels_2d_to_3d]*(slices//2 - 1)
        self.depth_reducing_layers = ModuleList([reduce_3d_depth(in_channel, channels_2d_to_3d, kernel_size=3, padding=0)
                                                 for in_channel in in_channels])
        self.slices = slices

    def forward(self, x):
        
        
        # down
        
        out = x
        
        for i in range(self.slices//2):
            out = self.depth_reducing_layers[i](out)
        
        return out

class DeepSCAN_2D(nn.Module):
    def __init__(self, depth, channels_in = 1, channels=32, output_channels = 1,
                 dilated_layers = [4,4,4,4], dilation_rates = None,
                growth_rate = 12):
        super(DeepSCAN_2D, self).__init__()
        
        self.depth = depth

        self.output_channels = output_channels



        self.down_layer_features_in = [channels_in]+ [channels*(2**i) for i in range(self.depth -1)]

        self.down_layer_features_out = [channels*(2**i) for i in range(self.depth)]


        self.down_layers = ModuleList([down_layer(in_channel = i, 
                                  out_channel =j,
                                  kernel_size = 3,
                                  padding=0
                                 ) for i, j  in zip(self.down_layer_features_in, self.down_layer_features_out)])

        self.max_layers = ModuleList([nn.MaxPool2d(2) for i in range(self.depth)])

        if depth == 0:
            bottleneck_channels_in = channels_in
        else:
            bottleneck_channels_in = self.down_layer_features_out[-1]
        
        self.bottleneck, bottleneck_features  = dense_atrous_bottleneck(bottleneck_channels_in, growth_rate = growth_rate, 
                                                                       depth = dilated_layers, dilation_rates = dilation_rates)
        
        self.upsampling_layers = ModuleList([nn.Sequential(OrderedDict([
                ("upsampling",nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)), # since v0.4.0 align_corners= False is default, before was True
                ("pad", nn.ReplicationPad2d(1)),
                ("conv", nn.Conv2d(in_channels= bottleneck_features, 
                                   out_channels=bottleneck_features, 
                                   kernel_size=3, 
                                   padding=0))]))  for i in range(self.depth-1, -1, -1)])
        self.up_layers = ModuleList([up_layer(in_channel= bottleneck_features+ self.down_layer_features_out[i], 
                                   out_channel=bottleneck_features, 
                                   kernel_size=3, 
                                   padding=0) for i in range(self.depth-1, -1, -1)])
        
        if output_channels is not None:
            self.last = nn.Conv2d(in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1)
        

    def forward(self, x):
        
        
        # down
        
        out = x
                
        links = []

        
        for i in range(self.depth):
            out = self.down_layers[i](out)
            links.append(out)
            out = self.max_layers[i](out)
        
        out = self.bottleneck(out)

        
        links.reverse()

        # up
        
        for i in range(self.depth):

            out = self.upsampling_layers[i](out)

            out = concatenate(links[i], out)
            out = self.up_layers[i](out)
        
        if self.output_channels is not None:
            out = self.last(out)
    


        return out



