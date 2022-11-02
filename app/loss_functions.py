import torch
import torch.nn as nn

from torch.nn.functional import cross_entropy, binary_cross_entropy, binary_cross_entropy_with_logits


class BCE_from_logits(nn.modules.Module):
    def __init__(self, indices=None):
        super(BCE_from_logits,self).__init__()
        self.name = 'BCE loss'
        self.indices = indices
    def forward(self, input, target, mask=None):
        if  self.indices is not None:
            input_target = input[:,self.indices]
        else:
            input_target = input
        assert input_target.shape == target.shape
        if mask is not None:
            if self.indices is not None:
                mask_target =  mask[:,self.indices]
            else:
                mask_target = mask
            assert input_target.shape == mask_target.shape
        max_val = (-input_target).clamp(min=0)
        loss = input_target - input_target * target + max_val + ((-max_val).exp() + (-input_target - max_val).exp()).log()
        if mask is None:
            return loss.mean()
        else:
            eps = 1e-10
            return (loss * mask_target).sum()/(torch.sum(mask_target) + eps)

class Categorical_crossentropy_with_implicit_background(nn.modules.Module):
    def __init__(self, indices=None):
        super(BCE_from_logits,self).__init__()
        self.name = 'Categorical Crossentropy'
        self.indices = indices
    def forward(self, input, target, mask=None):
        if  self.indices is not None:
            input_cce = input[:,self.indices]
            target_cce = target[:, self.indices]
        else:
            input_cce = input
            target_cce = target
        assert input_target.shape == target.shape
        if mask is not None:
            if self.indices is not None:
                mask_cce =  mask[:,self.indices]
            else:
                mask_target = mask
            assert input_cce.shape == mask_cce.shape
        bg_target = target_cce.all(1).logical_not()
        bg_input = torch.zeros_like(bg_target)

        input_cce = torch.cat([bg_input, input_cce])
        target_cce = torch.cat([bg_target, target_cce])

        loss = cross_entropy(input_cce, torch.argmax(target_cce, 1), reduction='none')
        if mask is None:
            return loss.mean()
        else:
            eps = 1e-10
            return (loss * mask_target).sum()/(torch.sum(mask_target) + eps)
    
class BCE_from_logits_focal(nn.modules.Module):
    def __init__(self, gamma, indices=None):
        super(BCE_from_logits_focal,self).__init__()
        self.gamma = gamma
        self.name = 'focal loss'
        self.indices =  indices
    def forward(self, input, target, mask=None):
        if  self.indices is not None:
            input_target = input[:,self.indices]
        else: 
            input_target = input
        assert input_target.shape == target.shape
        if mask is not None:
            if self.indices is not None:
                mask_target =  mask[:,self.indices]
            else:
                mask_target = mask
            assert input_target.shape == mask_target.shape
        max_val = (-input_target).clamp(min=0)
        loss = input_target - input_target * target + max_val + ((-max_val).exp() + (-input_target - max_val).exp()).log()
        p = input_target.sigmoid()
        pt = (1-p)*(1-target) + p*target
        focal_loss = (((1-pt).pow(self.gamma))*loss)
        if mask is None:
            return focal_loss.mean()
        else:
            eps = 1e-10
            return (focal_loss * mask_target).sum()/(torch.sum(mask_target) + eps)

class Heteroscedastic_loss(nn.modules.Module):
    def __init__(self, target_gamma, flip_gamma, target_indices, flip_indices, use_entropy = False):
        super(Heteroscedastic_loss,self).__init__()
        # both the prediction of the target and the prediction of the disgreement
        # can be focal.  Recommendation is to set flip_gamma to be zero, since the 
        # error rate may (hopefully) decrease over time.
        self.target_gamma = target_gamma
        self.flip_gamma = flip_gamma
        self.name = 'heteroscedastic loss'
        self.target_indices = target_indices
        self.flip_indices = flip_indices
        self.use_entropy = use_entropy
    def forward(self, input, target, mask=None):
        input_target = input[:,self.target_indices]
        input_flip = input[:,self.flip_indices]
        assert input_target.shape == target.shape
        assert input_flip.shape == target.shape
        if mask is not None:
            mask_target = mask[:, self.target_indices]
            assert input_target.shape == mask_target.shape
        else:
            mask_target = None


         #flip output (range[-inf , inf]) is log (-logit(p)). where p is the flip probability
         #this ensures that the flip prob is below 0.5
        flip_prob = torch.sigmoid(-torch.exp(input_flip))

        #attenuate the target, by reducing its label in places where the classifier predicts disagreement

        flip_attenuated_target = (1- target)*flip_prob +  target * (1 - flip_prob)

        #generate a 'ground truth' for where the classifier and label set disagree

        false_neg = ((-input_target).sign().clamp(min=0))*target
        false_pos = input_target.sign().clamp(min=0)*(1-target)
        label_disagreement = false_neg+false_pos
    

        
        flip_loss =    BCE_from_logits_focal(self.flip_gamma)(-torch.exp(input_flip), label_disagreement, mask_target)

        if self.use_entropy:
            flip_logit = -torch.exp(input_flip)
            max_val = (-input_target).clamp(min=0)
            max_val_flip = (-flip_logit).clamp(min=0)
            loss = input_target - input_target * flip_attenuated_target + max_val + ((-max_val).exp() + (-input_target - max_val).exp()).log()
            entropy = flip_logit - flip_logit * flip_prob + max_val_flip + ((-max_val_flip).exp() + (-flip_logit - max_val_flip).exp()).log()
            p = input_target.sigmoid()
            pt = torch.abs(1-(p-flip_attenuated_target)) #(1-p)*(1-flip_attenuated_target) + p*flip_attenuated_target
            focal_loss = (((1-pt).pow(self.target_gamma))*(loss-entropy))
            if mask is None:
                target_loss =  focal_loss.mean()
            else:
                eps = 1e-10
                target_loss =  (focal_loss * mask_target).sum()/(torch.sum(mask_target) + eps)
        else:
            target_loss =  BCE_from_logits_focal(self.target_gamma)(input_target, flip_attenuated_target, mask_target) 



        return target_loss + flip_loss 

class SoftDiceLoss(nn.Module):
    def __init__(self, batch_dice=False, dice_per = False, square = False, smooth=1.):
        super(SoftDiceLoss, self).__init__()

        self.batch_dice = batch_dice
        self.smooth = smooth
        self.square = square

    def forward(self, x, y_onehot, loss_mask=None):

        if self.batch_dice:
            axes = [0] + list(range(2, len(x.shape)))
        else:
            axes = list(range(2, len(x.shape)))

        intersect = x * y_onehot
        
        if self.square:

            denominator = x ** 2 + y_onehot ** 2

        else:
            denominator = x  + y_onehot 


        intersect = intersect.sum(axes) 

        denominator = denominator.sum(axes) 


        dc = ((2 * intersect) + self.smooth) / (denominator + self.smooth)

        dc = dc.mean()


        return -dc
    
class DiceSquaredLoss(nn.modules.Module):
    def __init__(self, batch_dice=True, from_logits=True, indices=None):
        super(DiceSquaredLoss,self).__init__()
        self.batch_dice = batch_dice
        self.from_logits = from_logits
        self.name = 'Dice loss'
        self.indices = indices
    def forward(self, input, target, mask=None):
        if  self.indices is not None:
            input_target = input[:,self.indices]
        else:
            input_target = input
        assert input_target.shape == target.shape
        if mask is not None:
            if self.indices is not None:
                mask_target =  mask[:,self.indices]
            else:
                mask_target = mask
            assert input_target.shape == mask_target.shape
        if self.from_logits:
            sigmoid_input = input_target.sigmoid()
        else:
            sigmoid_input = input_target
        if mask is not None:    
            sigmoid_input = sigmoid_input*mask_target
        
        return 1 + SoftDiceLoss(batch_dice=self.batch_dice, square=True)(target, sigmoid_input)

class DiceLoss(nn.modules.Module):
    def __init__(self, batch_dice=True, from_logits=True, indices=None):
        super(DiceLoss,self).__init__()
        self.batch_dice = batch_dice
        self.from_logits = from_logits
        self.name = 'Dice loss'
        self.indices = indices
    def forward(self, input, target, mask=None):
        if  self.indices is not None:
            input_target = input[:,self.indices]
        else:
            input_target = input
        assert input_target.shape == target.shape
        if mask is not None:
            if self.indices is not None:
                mask_target =  mask[:,self.indices]
            else:
                mask_target = mask
            assert input_target.shape == mask_target.shape
        if self.from_logits:
            sigmoid_input = input_target.sigmoid()
        else:
            sigmoid_input = input_target
        if mask is not None:    
            sigmoid_input = sigmoid_input*mask_target
        
        return 1 + SoftDiceLoss(batch_dice=self.batch_dice, square=False)(target, sigmoid_input)
        
        
        

