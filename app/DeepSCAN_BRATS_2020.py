#!/usr/bin/env python

import torch
import sys
import os

device = torch.cuda.current_device()





NETWORK_TYPE = '3D-to-2D'
#set this to True, if you want all the 3D image files to be loaded into memory
IN_MEMORY = True
#set this above zero to limit the number of training cases
NUM_TRAINING_CASES = 0
#set this above zero to limit the number of validation cases (to make epochs faster)
NUM_VALID_CASES = 0
#data loader/augmentation setting
NUM_THREADS = 4
ENSEMBLE_AXES = ['axial','sagittal','coronal']
BG_ZERO=True
#training settings
BATCHES_PER_EPOCH =  20000
PATCH_SIZE = (5, 194, 194)
BATCH_SIZE = 2
GRADIENT_MASK_ZERO_VOXELS = True
HETEROSCEDASTIC_ENTROPY_TERM = True
TARGET_LABEL_SETS = [ [4],
                     [1,4],
                     [1,2,4]
                    ]
TARGET_LABEL_NAMES = ['enhancing', 'tumor_core', 'whole_tumor']



import numpy as np
import pandas as pd
import nibabel as nib
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, Function
import copy
from old_dataloader import make_sampler
from batchgenerators.augmentations.utils import pad_nd_image
from scipy.ndimage.filters import gaussian_filter1d

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--pad', default = 194, type=int, help='Size to pad to before applying classfier.  Should be larger than the input and divisible by 2 :-)')
parser.add_argument('-l', default = False, action = 'store_true', dest='lite_mode')
parser.add_argument('--indir', type=str)

my_args = parser.parse_args()

INPUT_DIRS = my_args.indir
OUTPUT_DIR = os.path.join(my_args.indir, "results")



PATCH_SIZE = (5, my_args.pad, my_args.pad)

if my_args.lite_mode:
	print ('LITE MODE')


# In[29]:


if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

def load_case(subject):
    flair = nib.load(f'{subject}/FLAIR.nii.gz').get_data()
    t1 = nib.load(f'{subject}/T1.nii.gz').get_data()
    t2 = nib.load(f'{subject}/T2.nii.gz').get_data()
    t1ce = nib.load(f'{subject}/T1GD.nii.gz').get_data()
    gt = nib.load(f'{subject}/gt.nii.gz').get_data()

    return np.stack([flair,t1,t2,t1ce,gt]).astype(np.int16)

def load_training_case(subject):
    return load_case(subject)

def load_validation_case(subject):
    return load_case(subject)


def load_subject_volumes_from_nifti():
    subject_volumes = {}
    for subject in train_ids:
        subject_volumes[subject] = load_training_case(subject)

    for subject in validate_ids:
        subject_volumes[subject] = load_validation_case(subject)

    return subject_volumes



def load_subject(subject, axis = 'axial'):
        assert axis in ['axial', 'sagittal', 'coronal']

        if isinstance(subject, np.ndarray):
            data = subject
        else:
            data = subject_volumes[subject]

        if axis == 'coronal':
          data = np.swapaxes(data, 1,2)

        if axis == 'axial':
          data = np.swapaxes(data, 1,3)

        metadata = {}

        metadata['means'] = []

        metadata['sds'] = []

        for modality in range(data.shape[0]-1):

            metadata['means'].append(np.mean(data[modality][data[modality]>0]))
            metadata['sds'].append(np.std(data[modality][data[modality]>0]))

        metadata['means'] = np.array(metadata['means'])
        metadata['sds'] = np.array(metadata['sds'])


        #metadata = load_pickle(subject + ".pkl")

        return data, metadata

def load_subject_and_preprocess(subject, axis, bg_zero=True):
    all_data, subject_metadata = load_subject(subject, axis=axis)
    image_data =  all_data[:-1]
    zero_mask = (image_data != 0).astype(np.uint8)
    image_data = image_data - np.array(subject_metadata['means'])[:,np.newaxis,np.newaxis,np.newaxis]
    sds_expanded = np.array(subject_metadata['sds'])[:,np.newaxis,np.newaxis,np.newaxis]
    image_data = np.divide(image_data, sds_expanded,                            out=np.zeros_like(image_data), where=sds_expanded!=0)

    if bg_zero:
        print(subject)
        print(image_data.shape)
        print(image_data.nbytes)
        image_data = image_data * zero_mask
        print(image_data.nbytes)
        image_data = image_data.astype(float)
        # image_data = (image_data * zero_mask).astype(np.float)

    return image_data

def get_gradient_mask(volumes):
    if GRADIENT_MASK_ZERO_VOXELS:
        nonzero_mask_all = np.any(volumes>0, axis=0)
        return (nonzero_mask_all).astype(np.float)
    else:
        return (np.ones_like(volumes[0]).astype(np.float))

def make_target(gt, train=True, network_type = NETWORK_TYPE):
    target = np.concatenate([np.isin(gt,np.array(labelset)).astype(np.float) for labelset in TARGET_LABEL_SETS], axis =1)
    if train and network_type=='3D-to-2D':
        target = target[:,:,PATCH_SIZE[0]//2]
    return target


# In[36]:


from loss_functions import BCE_from_logits_focal, DiceLoss, Heteroscedastic_loss

criteria = [BCE_from_logits_focal(2,indices = [0,1,2]),
          DiceLoss(indices=[0,1,2]),
          Heteroscedastic_loss(2,0,target_indices = [0,1,2],flip_indices = [3,4,5],use_entropy =HETEROSCEDASTIC_ENTROPY_TERM)]


# In[37]:


from old_dataloader import rotateImage
import cv2

def rotate_image_on_axis(image, angle, rot_axis):
    return np.swapaxes(rotateImage(np.swapaxes(image,2,rot_axis),angle,cv2.INTER_LINEAR)
                                 ,2,rot_axis)

def rotate_stack(stack, angle, rot_axis):
    images = []
    for idx in range(stack.shape[0]):
        images.append(rotate_image_on_axis(stack[idx], angle, rot_axis))
    return np.stack(images, axis=0)


# In[67]:



def apply_to_case_uncert(models, subject, do_mirroring=[False], rot_angles = [-15,0,15], rot_axes = [0], patch_size=PATCH_SIZE, pad_size = (192,192),axes=['axial'],
                 use_gaussian= True, bg_zero = True):
    print(f'applying {len(models)} model(s) over {len(axes)} axes rotating through {len(rot_angles)} angle(s)')
    with torch.no_grad():
        ensemble_logits = []
        ensemble_flips = []
        slice_masks = []
        case_data = load_subject_and_preprocess(subject, axis='sagittal', bg_zero=bg_zero)

        if do_mirroring == False:
            do_mirroring = [False]
        if do_mirroring == True:
            do_mirroring = [True,False]

        for model in models:
            model.eval()

            for axis in axes:

                for mirror in do_mirroring:

                    for angle in rot_angles:

                        for rot_axis in rot_axes:

                            if angle != 0:
                                image_data = rotate_stack(case_data.copy(), angle, rot_axis)

                            else:
                                image_data = case_data.copy()

                            if mirror:
                                image_data = image_data[:, ::-1].copy()

                            if axis == 'coronal':
                                image_data = np.swapaxes(image_data, 1, 2)

                            if axis == 'axial':
                                image_data = np.swapaxes(image_data, 1, 3)






                            if NETWORK_TYPE == '3D-to-2D':

                                input, slicer =  pad_nd_image(image_data, (0, patch_size[1], patch_size[2]), return_slicer=True)

                            if NETWORK_TYPE == '3D':

                                input, slicer =  pad_nd_image(image_data, patch_size, return_slicer=True)


                            slicer[0] = slice(0, len(TARGET_LABEL_SETS)*2, None)



                            output = model.predict_3D(torch.from_numpy(input).float().cuda(),do_mirroring=do_mirroring, patch_size=patch_size,
                                                     use_sliding_window=True, use_gaussian = use_gaussian)




                            output = output[1][tuple(slicer)]


                            slice_sums = np.sum(np.any(image_data>0, 0), (1,2))

                            slice_mask = np.stack([np.stack([slice_sums>2500]*image_data.shape[2],-1)]*image_data.shape[3],-1)

                            slice_mask = np.stack([slice_mask]*len(TARGET_LABEL_SETS)).astype(np.uint8)


                            if axis == 'coronal':
                                output = np.swapaxes(output, 1,2)
                                image_data = np.swapaxes(image_data, 1, 2)
                                slice_mask = np.swapaxes(slice_mask, 1, 2)


                            if axis == 'axial':
                                output = np.swapaxes(output, 1,3)
                                image_data = np.swapaxes(image_data, 1, 3)
                                slice_mask = np.swapaxes(slice_mask, 1, 3)

                            if mirror:
                                output = output[:, ::-1].copy()
                                image_data = image_data[:, ::-1].copy()
                                slice_mask = slice_mask[:, ::-1].copy()

                            if angle != 0:
                                output = rotate_stack(output.copy(), -angle, rot_axis)
                                slice_mask = (rotate_stack(slice_mask, -angle, rot_axis)>0).astype(np.uint8)


                            output[:len(TARGET_LABEL_NAMES)][np.logical_not(slice_mask)] = np.nan
                            ensemble_logits.append(output[:len(TARGET_LABEL_NAMES)])

                            flip = ((-torch.from_numpy(output[len(TARGET_LABEL_NAMES):]).exp()).sigmoid()).numpy()

                            flip_logit = ((-torch.from_numpy(output[len(TARGET_LABEL_NAMES):]).exp())).numpy()

                            flip[np.logical_not(slice_mask)] = np.nan
                            ensemble_flips.append(flip)

                            slice_masks.append(slice_mask)

    ensemble_counts = np.sum(slice_masks, axis=0)


    uncertainty_weighted_logits = -np.sign(np.array(ensemble_logits))*np.array(flip_logit)

    full_logit = np.sum(np.divide(np.nan_to_num(np.array(ensemble_logits),0), ensemble_counts,
                        out = np.zeros_like(np.array(ensemble_logits)), where = ensemble_counts!=0),axis=0)



    ensemble_predictions = np.greater(np.nan_to_num(ensemble_logits,-10),0)

    full_predictions = np.stack([np.greater(full_logit,0)]*(len(axes)*len(do_mirroring)*len(rot_angles)*len(rot_axes)*len(models)),0)

    preds_agree = np.equal(full_predictions, ensemble_predictions).astype(np.uint8)

    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        full_flips = np.sum(np.nan_to_num(np.array(ensemble_flips),0)*(preds_agree) +
                         np.nan_to_num((1 - np.array(ensemble_flips)),0)*(1-preds_agree),axis=0)/ensemble_counts

    full_flips = np.nan_to_num(full_flips, 0)


    return np.concatenate([full_logit,full_flips])

def apply_to_case_uncert_and_get_target_mask_probs_logits(model, subject, do_mirroring=False, rot_angles = [0],
                                                          rot_axes = [0],patch_size=PATCH_SIZE,
                                                   pad_size = (192,192),axes=['axial'],
                                                   use_gaussian= True):

        subject_data, subject_metadata = load_subject(subject, axis='sagittal')

        logits = apply_to_case_uncert(model, subject=subject, do_mirroring=do_mirroring, rot_angles=rot_angles,
                                      rot_axes = rot_axes, patch_size=patch_size,
                               pad_size = pad_size ,axes=axes, use_gaussian= use_gaussian)




        target = make_target(subject_data[-1][None,None], train=False)[0]

        target_logits = logits[:len(TARGET_LABEL_NAMES)]

        target_probs = torch.from_numpy(target_logits).sigmoid().numpy()

        gradient_mask = get_gradient_mask(subject_data[:-1]).astype(np.float)

        target_logits = target_logits*gradient_mask[None]

        target_probs = target_probs*gradient_mask[None]

        return logits, gradient_mask, target, target_logits, target_probs


# In[39]:


def get_TP_FP_FN_TN(target, prediction, from_logits=True):
    if from_logits:
        threshold=0
    else:
        threshold=0.5

    TP = np.sum(np.logical_and(prediction>threshold , target>0), axis=(1,2,3))
    FP = np.sum(np.logical_and(prediction>threshold , target==0), axis=(1,2,3))
    FN = np.sum(np.logical_and(prediction<=threshold , target>0), axis=(1,2,3))
    TN = np.sum(np.logical_and(prediction<=threshold , target==0), axis=(1,2,3))

    return TP, FP, FN, TN

def get_dices_from_TP_FP_FN_TN(TP, FP, FN, TN):
    epsilon = 0.000001
    dices = (2*TP+epsilon)/(2*TP+FP+FN+epsilon)
    return dices

def get_dices_from_prediction(target, prediction, from_logits=True):
    TP, FP, FN, TN = get_TP_FP_FN_TN(target, prediction, from_logits=from_logits)
    return get_dices_from_TP_FP_FN_TN(TP, FP, FN, TN)




# In[40]:


class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count



# In[42]:


USE_ATTENTION=True
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function

from torch.nn import BCELoss, NLLLoss, BCEWithLogitsLoss, MSELoss, ModuleList, ReplicationPad2d

from collections import OrderedDict


class GradMultiplier(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * self.lambd)

def grad_multiply(x, lambd):
    return GradMultiplier(lambd)(x)

def reduce_3d_depth (in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad3d((1,1,1,1,0,0))),
            ("conv1", nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm3d(out_channel, affine = False)),
            ("relu1", nn.ReLU()),
            #("dropout", nn.Dropout(p=0.2))
    ]))
    return layer

def down_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(p=0.0)),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(p=0.0))]))
    return layer

def up_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(p=0.0)),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(p=0.0))]))
    return layer

class DilatedDenseUnit(nn.Module):
    def __init__(self, in_channel, growth_rate , kernel_size, dilation):
        super(DilatedDenseUnit,self).__init__()
        self.layer = nn.Sequential(OrderedDict([
            ("bn1", nn.InstanceNorm2d(in_channel, affine = False)),
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
            ("bn1", nn.InstanceNorm2d(in_channel, affine = False)),
            ("relu1", nn.ReLU()),
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, intermediate_channel, kernel_size=kernel_size,padding=0)),
            ("bn2", nn.InstanceNorm2d(intermediate_channel, affine = False)),
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

        if USE_ATTENTION:
            layer_dict["attention_{}".format(dilation_rate)] = AttentionModule(in_channel, in_channel//4, in_channel)
    return nn.Sequential(layer_dict), in_channel




class UNET_3D_to_2D(nn.Module):
    def __init__(self, depth, channels_in = 1,
                 channels_2d_to_3d=32, channels=32, output_channels = 1, slices=5,
                 dilated_layers = [4,4,4,4],
                growth_rate = 12):
        super(UNET_3D_to_2D, self).__init__()

        self.output_channels = output_channels
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
        logvar = self.logvar(out)




        return torch.cat([pred, logvar], axis=1)

    def predict_3D(self, x, do_mirroring=False, mirror_axes=None,
                    use_sliding_window=True, use_gaussian = True,
                    step_size = 1, patch_size=(5,194,194), batch_size = 2):
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


# In[43]:


def load_checkpoint(net, checkpoint_file):
        if os.path.isfile(checkpoint_file):
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            net.load_state_dict(checkpoint['state_dict'])


# In[44]:


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]

def crop_to_nonzero(data, seg=None, nonzero_label=0):

    bbox = get_bbox_from_mask(np.all(data[:-1]>0, axis=0), 0)

    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    return data, bbox


# In[45]:


from postprocess_brats_seg import make_brats_segmentation, postprocess_brats_segmentation


# In[46]:



unets = []
for fold in [0,1,2,3,4]:
    model = UNET_3D_to_2D(0,channels_in=4,channels=64, growth_rate =8, dilated_layers=[4,4,4,4], output_channels=len(TARGET_LABEL_NAMES))

    model.load_state_dict(torch.load(f'fold{fold}_retrained_nn.pth.tar',map_location='cpu'))

    model.to(device)

    unets.append(model)


# In[47]:


def mask_flip_probabilities(output,
                            gradient_mask):

    flip_probs = output[len(TARGET_LABEL_NAMES):] * gradient_mask[None]


    return flip_probs


# In[49]:





def resolve_file(casepath, modality):
    filenames =  [x for x in os.listdir(casepath) if modality+".nii" in x]
    print(filenames)
    assert len(filenames) == 1
    return filenames[0]


# In[11]:


# In[12]:



def predict_with_uncertainty(id=INPUT_DIRS, models=unets,
                            save_intemediate=False,
                             val_segs = OUTPUT_DIR,
                             post_segs = OUTPUT_DIR,
                             export_folder = OUTPUT_DIR,
                             PATCH_SIZE = PATCH_SIZE):

    print(id)
    if not os.path.isdir(val_segs):
        os.mkdir(val_segs)

    if not os.path.isdir(post_segs):
        os.mkdir(post_segs)
    if not os.path.isdir(export_folder):
        os.mkdir(export_folder)

    flair = np.copy(nib.load(id+resolve_file(id, 'FLAIR')).get_fdata())
    t1 = np.copy(nib.load(id+resolve_file(id, 'T1')).get_fdata())
    t2 = np.copy(nib.load(id+resolve_file(id, 'T2')).get_fdata())
    t1ce = np.copy(nib.load(id+resolve_file(id, 'T1GD')).get_fdata())



    gt = np.zeros_like(flair)

    val_subject_volume = np.stack([flair,t1,t2,t1ce,gt]).astype(np.int16)

    cropped, bbox = crop_to_nonzero(val_subject_volume)

    im_size = np.max(cropped.shape)

    print(f'cropped input image max dimension = {im_size}')

    if im_size > PATCH_SIZE[1]:
    	PATCH_SIZE = (5, 2*((im_size+1)//2), 2*((im_size+1)//2))
    	print(f'cropped image exceeds patch size: new patch size = {PATCH_SIZE}')

    if my_args.lite_mode:
    	angles = [0]
    else:
    	angles = [-45,0,45]

    logits, gradient_mask, target, target_logits, target_probs = apply_to_case_uncert_and_get_target_mask_probs_logits(models, cropped, axes = ENSEMBLE_AXES,
                                                                                                                      rot_angles=angles, rot_axes = [0,1,2],
                                                                                                                      patch_size=PATCH_SIZE,
                                                                                                                      do_mirroring=True)

    target_probs_uncropped = np.zeros_like(np.stack([val_subject_volume[0]]*len(TARGET_LABEL_SETS))).astype(np.float32)

    target_probs_uncropped[:, bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = target_probs

    nifti_orig = nib.load(id+resolve_file(id, 'FLAIR'))
    nifti_affine = nifti_orig.affine
    all_nifti_100 = nib.Nifti1Image((target_probs_uncropped*100).transpose((1,2,3,0)).astype(np.uint8), nifti_affine)

    #nib.save(all_nifti_100, f'{val_segs}/{id}_probs.nii.gz')




    #nib.save(all_nifti_100, f'{val_segs}/{id}_probs.nii.gz')

    flip = mask_flip_probabilities(logits, gradient_mask)

    flip_uncropped = np.zeros_like(np.stack([val_subject_volume[0]]*len(TARGET_LABEL_SETS))).astype(np.float32)

    flip_uncropped[:, bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = flip


    flip_nifti_100 = nib.Nifti1Image((flip_uncropped*100).transpose((1,2,3,0)).astype(np.uint8), nifti_affine)

    #nib.save(flip_nifti_100, f'{val_segs}/{id}_flips.nii.gz')

    flip_prob_fusion = ((target_probs_uncropped>0.5)*(1-flip_uncropped) + (target_probs_uncropped<0.5)*(flip_uncropped))

    flip_prob_fusion_100 = nib.Nifti1Image((flip_prob_fusion*100).transpose((1,2,3,0)).astype(np.uint8), nifti_affine)


    #nib.save(flip_prob_fusion_100, f'{val_segs}/{id}_flip_prob_fusion.nii.gz')


    seg = make_brats_segmentation(flip_prob_fusion*100)

    seg_nifti = nib.Nifti1Image((seg).astype(np.uint8), nifti_affine)




    #nib.save(seg_nifti, f'{val_segs}/{id}_seg.nii.gz')

    seg_postprocessed, uncertainty = postprocess_brats_segmentation(seg, (flip_prob_fusion*100).transpose((1,2,3,0)), flair,t1)

    postprocessed_nifti = nib.Nifti1Image((seg_postprocessed).astype(np.uint8), nifti_affine)


    nib.save(postprocessed_nifti, f'{post_segs}/tumor_SCAN2020_class.nii.gz')

    #nib.save(postprocessed_nifti, f'{export_folder}/{id}.nii.gz')


    for idx, name in zip([0,1,2],['enhance','core','whole']):
        unc_map = nib.Nifti1Image((uncertainty[:,:,:,idx]).astype(np.uint8), nifti_affine)
        nib.save(unc_map, f'{export_folder}/tumor_SCAN2020_unc_{name}.nii.gz')

    return seg_postprocessed


# In[72]:

def resolve_seg(casepath):
    filenames =  [x for x in os.listdir(casepath) if "_seg.nii.gz" in x]
    print(filenames)
    if len(filenames) == 1:
        return filenames[0]
    else:
        return None

existing_seg = resolve_seg(INPUT_DIRS)

if  existing_seg is None:
	if my_args.lite_mode:
	    seg_postprocessed = predict_with_uncertainty(models = [unets[0]])
	else:
		seg_postprocessed = predict_with_uncertainty()


else:

	seg_postprocessed = nib.load(INPUT_DIRS+existing_seg).get_fdata()


def resolve_csv(casepath):
    filenames =  [x for x in os.listdir(casepath) if ".csv" in x]
    print(filenames)
    if len(filenames) == 1:
        return filenames[0]
    else:
        return None

csv_file = resolve_csv(INPUT_DIRS)

if csv_file is not None:
    import pickle
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import  RandomForestClassifier
    from scipy.ndimage import label


    linear_regressor = pickle.load(open('linear_regression.sav','rb'))
    rf_classifier = pickle.load(open('random_forest_classifier.sav','rb'))
    survival_csv_test = pd.read_csv(INPUT_DIRS+csv_file).dropna()

    if len(survival_csv_test) == 1:
        patient_id = survival_csv_test.iloc[0][0]
        age = survival_csv_test.iloc[0][1]



        core = np.logical_or(seg_postprocessed==4, seg_postprocessed==1)
        num_cores = label(core)[1]



        tumor = seg_postprocessed>0
        num_tumors = label(tumor)[1]

        linear_model_prediction = linear_regressor.predict([[age, num_tumors, num_cores]])
        rf_prediction = rf_classifier.predict([[age, num_tumors, num_cores]])
        rf_probs = rf_classifier.predict_proba([[age, num_tumors, num_cores]])

        survival = linear_model_prediction

        if np.max(rf_probs)>=0.5:
            def get_survival_class(days):
                if days < 300:
                    return 'short'
                elif days <450:
                    return 'mid'
                else:
                    return 'long'

            def class_to_days(string):
                if string == 'short':
                    return 290
                elif string == 'mid':
                    return 375
                else:
                    return 500

            if get_survival_class(linear_model_prediction) != rf_prediction:
                survival = class_to_days(rf_prediction)


        pd.DataFrame([[patient_id,int(survival)]]).to_csv(f'{OUTPUT_DIR}/{patient_id}_tumor_SCAN2020_survival.csv', header=False, index=False)






