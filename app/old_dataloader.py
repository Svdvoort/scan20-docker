#this cell defines the main function for extracting a stack of slices from a volume and normalizing it, possibly with augmentation
import cv2
from torch.utils.data.sampler import Sampler, RandomSampler, WeightedRandomSampler, SequentialSampler
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
# from torch._six import int_classes as _int_classes
int_classes = int
import numpy as np

import itertools

TARGET_LABEL_SETS = [ [4],
                     [1,4],
                     [1,2,4]
                    ]


def rotateImage(image, angle, interp = cv2.INTER_NEAREST):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=interp)
  return result


def get_stack(axis, 
              volume, 
              central_slice, 
              first_slice=None, 
              last_slice=None, 
              stack_depth=5, 
              size = (192,192), 
              rotate_angle = None, 
              rotate_axis = 0, 
              flipLR = False,
              bg_zero = False,
             lower_threshold = None,
             upper_threshold= None,
             return_nonzero= True,
             fliplr_axis = 0):
    if (first_slice is not None or last_slice is not None) and central_slice is not None:
        raise ValueError('Stack location overspecified: use either first/last slice or central slice')
    if (first_slice is not None and last_slice is not None) and stack_depth is not None:
        raise ValueError('Stack location overspecified: use either first AND last slice or stack_depth')
    if (central_slice is not None and stack_depth%2 == 0):
        raise ValueError('Specifying central slice and stack depth only possible if stack depth is odd')
    image_data = volume
    if flipLR:
        #image_data = image_data[:,:,::-1]
        image_data = np.flip(image_data, fliplr_axis)

    if rotate_angle is not None:
        image_data = np.swapaxes(rotateImage(np.swapaxes(image_data,2,rotate_axis),rotate_angle,cv2.INTER_LINEAR)
                                 ,2,rotate_axis)
    
    image_data = np.array(np.swapaxes(image_data, 0, axis), copy = True)
    
    mean = np.mean(image_data[image_data>0])
    sd = np.sqrt(np.var(image_data[image_data>0]))
    
    
    if lower_threshold is None:
        lower_threshold = 0
    
    if upper_threshold is None:
        upper_threshold = np.max(image_data[image_data>0])
    
    
    
    image_data[image_data<lower_threshold]=lower_threshold
    image_data[image_data>upper_threshold]=upper_threshold
    
    
    
    if first_slice is None:
        if central_slice is not None:
            first_slice = central_slice - stack_depth//2
            last_slice = central_slice + stack_depth//2 + 1
        elif last_slice is not None:
            first_slice = last_slice - stack_depth
    elif last_slice is None:
            last_slice = min(first_slice + stack_depth, len(image_data))
    pad_up = max(0, -first_slice)

    pad_down = -min(0, len(image_data)-last_slice)

    first_slice = max(first_slice,0)
    last_slice = min(last_slice, len(image_data))
    initial_stack = image_data[first_slice:last_slice]
    initial_shape = initial_stack.shape[1:]
    shape_difference = (size[0] - initial_shape[0],size[1] - initial_shape[1])
    pad_size = ((pad_up,pad_down),
                (shape_difference[0]//2, shape_difference[0] - shape_difference[0]//2),
                (shape_difference[1]//2, shape_difference[1] - shape_difference[1]//2) )
    initial_stack = np.pad(initial_stack, pad_size, mode = 'constant', constant_values = lower_threshold)
    
    

    nonzero_mask = (initial_stack>lower_threshold).astype(np.int)

    image_data = (initial_stack - mean)/sd
    
    
    return image_data,  nonzero_mask
    
def get_gt_stack(axis, 
              gt_volume,  
              central_slice, 
              first_slice=None, 
              last_slice=None, 
              stack_depth=5, 
              size = (192,192), 
              rotate_angle = None, 
              rotate_axis = 0, 
              flipLR = False,
             lower_threshold = None,
             upper_threshold= None,
             return_gt = True,
             return_nonzero= True,
             fliplr_axis = 0):
    if (first_slice is not None or last_slice is not None) and central_slice is not None:
        raise ValueError('Stack location overspecified: use either first/last slice or central slice')
    if (first_slice is not None and last_slice is not None) and stack_depth is not None:
        raise ValueError('Stack location overspecified: use either first AND last slice or stack_depth')
    if (central_slice is not None and stack_depth%2 == 0):
        raise ValueError('Specifying central slice and stack depth only possible if stack depth is odd')
    gt = gt_volume

    if flipLR:
        #gt = gt[:,:,::-1]
        gt = np.flip(gt, fliplr_axis)
    if rotate_angle is not None:
        gt = np.swapaxes(rotateImage(np.swapaxes(gt,2,rotate_axis),rotate_angle,cv2.INTER_LINEAR)
                                 ,2,rotate_axis)

    gt  = np.swapaxes(gt, 0, axis)

    if first_slice is None:
        if central_slice is not None:
            first_slice = central_slice - stack_depth//2
            last_slice = central_slice + stack_depth//2 + 1
        elif last_slice is not None:
            first_slice = last_slice - stack_depth
    elif last_slice is None:
            last_slice = min(first_slice + stack_depth, len(gt))
    pad_up = max(0, -first_slice)

    pad_down = -min(0, len(gt)-last_slice)

    first_slice = max(first_slice,0)
    last_slice = min(last_slice, len(gt))
    gt_stack = gt[first_slice:last_slice]   
    initial_shape = gt_stack.shape[1:]

    shape_difference = (size[0] - initial_shape[0],size[1] - initial_shape[1])
    pad_size = ((pad_up,pad_down),
                (shape_difference[0]//2, shape_difference[0] - shape_difference[0]//2),
                (shape_difference[1]//2, shape_difference[1] - shape_difference[1]//2) )

        

    gt_stack = np.pad(gt_stack, pad_size, mode = 'constant', constant_values = 0)

    
        
    
    return gt_stack





#pytorch dataset for extracting slices, with augmentation, for training



class BrainData(Dataset):
    def __init__(self, patient_volumes, train_ids, datapoints,axes = [0,1,2], patch_size = (5,192,192),bg_zero=False,fliplr_axis = 0):
        self.axes = axes
        self.length = len(train_ids)
        self.datapoints = datapoints
        self.patch_size= patch_size
        self.patient_volumes = patient_volumes
        self.train_ids = train_ids
        self.bg_zero = bg_zero
        self.fliplr_axis = fliplr_axis
    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        index, image_idx, random_flip, rotate_axis, rotate_angle, shift, scale, drop_modality= index
        stack_depth = self.patch_size[0]
        size = (self.patch_size[1],self.patch_size[2])
        case, axis, random_slice = self.datapoints[index]

        volumes = self.patient_volumes[self.train_ids[case]][:-1]

        gt = self.patient_volumes[self.train_ids[case]][-1]

        train_data = []

        for modality in range(volumes.shape[0]):
        	this_modality, nonzero = get_stack(axis=axis, volume=self.patient_volumes[self.train_ids[case]][modality], 
                                                   central_slice=random_slice, stack_depth=stack_depth, size = size, 
                                                   rotate_angle = rotate_angle, rotate_axis = rotate_axis, flipLR = random_flip, bg_zero = self.bg_zero, fliplr_axis = self.fliplr_axis)


        	if drop_modality == modality:
        		this_modality = np.random.normal(size = this_modality.shape)

        	this_modality = (this_modality*scale[modality])+shift[modality]

        	if self.bg_zero:
        		this_modality[nonzero==0] = 0.0

        	train_data.append(this_modality)

        gt_stack = get_gt_stack(axis=axis, gt_volume=self.patient_volumes[self.train_ids[case]][-1], 
                                                   central_slice=random_slice, stack_depth=stack_depth, size = size, 
                                                   rotate_angle = rotate_angle, rotate_axis = rotate_axis, flipLR = random_flip, fliplr_axis = self.fliplr_axis)

        
        central_slice = stack_depth//2
        
        images = np.stack(train_data).astype(np.float32)
        
        nonzero_masks = nonzero[central_slice].astype(np.float32)
        return images, gt_stack, nonzero_masks

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.length







class AugmentationSampler(object):
    """Wraps a sampler to yield a mini-batch of multiple indices with data augmentation parameters

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, iterations, num_modalities = 4, max_rot_angle = 15, rot_probs = [0.3,0.3,0.3,0.1], fortyfive = False,
                 drop_modality_prob = 0.0, drop_last=False):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.iterations = iterations
        self.num_modalities = num_modalities
        self.max_rot_angle = max_rot_angle
        self.rot_probs = rot_probs
        self.drop_modality_prob = drop_modality_prob
        self.fortyfive = fortyfive

    def __iter__(self):
        batch = []
        for y in range(self.iterations):
            for idx in self.sampler:
                random_masking = np.random.randint(2)
                random_flip = np.random.choice([False, True])
                rotate_axis = np.random.choice([0,1,2,None],p=self.rot_probs)
                shift = np.random.normal(0,0.5, self.num_modalities)
                scale = np.random.normal(1,0.2, self.num_modalities)
                if rotate_axis is not None:
                    rotate_angle = np.random.uniform(-self.max_rot_angle,self.max_rot_angle)
                    if self.fortyfive:
                      rotate_angle = rotate_angle  + np.random.choice([-45, 0, 45])
                else:
                    rotate_angle = None
                drop_modality = np.random.choice(list(range(self.num_modalities))+[None],p=[self.drop_modality_prob/self.num_modalities]*self.num_modalities + [1-self.drop_modality_prob])
                batch.append((idx, random_masking, random_flip, rotate_axis, rotate_angle, shift, scale,drop_modality))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch
                batch = []

    def __len__(self):
        if self.drop_last:
            return (len(self.sampler) // self.batch_size)*self.iterations
        else:
            return ((len(self.sampler) + self.batch_size - 1) // self.batch_size)*self.iterations
        
#Take a subset of the data, a label label_must_be_present and a fraction of slices where that label should be present, 
#and build a Dataloader to produce training examples

def convert_slice_map_to_datapoints(slice_map):
    datapoints = [[list(itertools.product([case],
                                          [axis],
                                          np.where(slice_map[case][0][axis])[0])) for axis in [0,1,2]] for case in range(len(slice_map))]

    datapoints = [x for  y in datapoints for x in y]
    datapoints = [x for  y in datapoints for x in y]
    
    return datapoints

        
        
def make_sampler(patient_volumes, train_ids, slice_maps, 
                 weights, batch_size = 2, num_modalities = 4, num_samples=None, 
                 num_threads = 0, patch_size = (5,192,192), bg_zero = True, 
                 fliplr_axis = 0, max_rot_angle = 15, rot_probs = [0.3,0.3,0.3,0.1], fortyfive = False):
    
    #we build a list of datapoints, where each datapoint is a tuple case, axis, slice, where
    #slices are listed only if there are sufficient nonempty gt voxels in the slice

    datapoints = []
    datapoint_weights = []

    for slice_map, weight in zip(slice_maps, weights):
    	new_datapoints = convert_slice_map_to_datapoints(slice_map)
    	datapoints = datapoints + new_datapoints
    	datapoint_weights += [weight]*len(new_datapoints)
            
    if num_samples is None:
        num_samples = len(datapoints)

        
    subsample_loader = DataLoader(BrainData(patient_volumes, train_ids, datapoints, patch_size = patch_size, bg_zero = bg_zero, fliplr_axis = fliplr_axis), batch_sampler = AugmentationSampler(WeightedRandomSampler(datapoint_weights, num_samples), batch_size=batch_size, iterations = 1, num_modalities=num_modalities, max_rot_angle = max_rot_angle, rot_probs = rot_probs,
                                            fortyfive = fortyfive, drop_last=False), 
                          num_workers=num_threads,pin_memory=True)
    
    return subsample_loader 
