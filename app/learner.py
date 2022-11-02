import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from torch.autograd import Variable, Function


from tqdm.notebook import tqdm

import copy


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.

        from https://github.com/davidtvs/pytorch-lr-finder/
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

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

class Trainer():
    def __init__(self, net, loader, criteria, evaluator, weights, make_target, label_names, batch_size = 2,
                optimizer = None, use_tqdm = True, device = torch.cuda.current_device()):
        self.net = net
        self.criteria = criteria
        self.evaluator = evaluator
        self.loader = loader
        self.weights = weights
        self.use_tqdm = use_tqdm
        self.device = device
        self.batch_size = batch_size
        if optimizer is None:
            optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        else:
            self.optimizer = optimizer 
        
        self.lr_history = []

        self.loss_history = []
        
        self.gradient_voxels_seen = []
        
        self.batches_on_validate = []
        
        self.dice_history = []

        self.make_target = make_target

        self.label_names = label_names
        


    def train(self, num_epochs = 50, 
              lr_max = 0.1,
              lr_min = 0.1, 
              scheduler=None, lr_find = False, warmup_optimizer = False,
              lr_find_steps = 100, warmup_steps = 100, evaluate = True):
        
        lr_history_this_run = []
        loss_history_this_run = []
        
        if lr_find:
            num_epochs = 1
            evaluate = False
            initial_model = copy.deepcopy(self.net.state_dict()) 
            initial_optimizer = copy.deepcopy(self.optimizer.state_dict())
            scheduler = 'exponential'
            num_iter = lr_find_steps
            print('finding optimal learning rate')
        elif warmup_optimizer:
            num_epochs = 1
            evaluate = False
            scheduler = 'exponential'
            num_iter = warmup_steps
            print('warming up optimizer')


          



        if scheduler == 'exponential':
            print('using exponential lr schedule')
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_min 
                param_group['initial_lr'] = lr_min   
     
            schedule = ExponentialLR(self.optimizer, end_lr = lr_max, num_iter = num_iter) 
        elif scheduler == 'cosine':
            print('using per-epoch cosine annealing')
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_max 
                param_group['initial_lr'] = lr_max   

            schedule = CosineAnnealingLR(self.optimizer, T_max = len(self.loader), eta_min=lr_min)
        elif scheduler == 'one-cycle':
            div_factor = 25
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_max/div_factor
                param_group['initial_lr'] = lr_max/div_factor   

            schedule = OneCycleLR(self.optimizer, max_lr = lr_max, epochs = num_epochs, 
                                  steps_per_epoch = len(self.loader), div_factor = div_factor)
        else:
            scheduler=None
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_max 
                param_group['initial_lr'] = lr_max   

            print('using constant learning rate')

        if self.use_tqdm:
            epoch_iterator = tqdm(range(0, num_epochs))
        else:
            epoch_iterator = range(0, num_epochs)

        for epoch in epoch_iterator:

            if scheduler == 'cosine':
                print(f'resetting lr schedule: lr max = {lr_max}, lr_min = {lr_min}')
                for param_group in self.optimizer.param_groups:
                    param_group['initial_lr'] = lr_max   
                schedule = CosineAnnealingLR(self.optimizer, T_max = len(self.loader), eta_min=lr_min)

            train_loss_trackers = [Average() for x in self.criteria]

            self.net.train()

            nonempty_batch_count = 0
            
            batch = 0

            for data in  tqdm(self.loader, leave=False):
                batch = batch+1
                if (lr_find and batch == lr_find_steps) or (warmup_optimizer and batch ==  warmup_steps):
                    break
                    
                images, gt, gradient_masks = data
                
                


                #images = images.to(device, non_blocking=True) # non_blocking similar to async=True in v0.3.1
                #targets = targets.to(device, non_blocking=True)
                gradient_masks = gradient_masks.to(self.device, non_blocking=True)
                #well this is ugly
                
                
                targets = torch.from_numpy((self.make_target(gt.unsqueeze(1).numpy())))
                
            

                self.optimizer.zero_grad()

                outputs = self.net(Variable(images).to(self.device, non_blocking=True))
                

                gradient_masks = gradient_masks.unsqueeze(1).expand_as(outputs)




                total_loss = 0


                targets = (targets).cuda(non_blocking=True)
                gradient_masks = (gradient_masks).cuda(non_blocking=True) 



                loss_values = [criterion(outputs, targets, gradient_masks) for criterion in self.criteria]

                for tracker, loss_value in zip(train_loss_trackers, loss_values):
                    tracker.update(loss_value.item(), 1)


                total_loss = torch.sum(torch.stack([x*y for x, y in zip(loss_values, self.weights)]))
                total_loss.backward()

                nn.utils.clip_grad_norm_(self.net.parameters(),1)

                self.optimizer.step()
                
                if not lr_find:
                    self.gradient_voxels_seen.append(gradient_masks.sum().cpu().cuda())
                    self.lr_history.append(self.optimizer.state_dict()["param_groups"][0]["lr"])
                    self.loss_history.append([x.cpu().item() for x in loss_values])
                
                lr_history_this_run.append(self.optimizer.state_dict()["param_groups"][0]["lr"])
                loss_history_this_run.append(total_loss.item())
                loss_report = ' , '.join([f'{lossfunction.name} = {tracker.avg:.4f}' for lossfunction, tracker in zip(self.criteria, train_loss_trackers)])

                print(loss_report + f' lr = {self.optimizer.state_dict()["param_groups"][0]["lr"]:.3e}', end="\r")

                if scheduler is not None:
                    schedule.step()


            print(loss_report)


            all_dices = []

            if evaluate:
              #val_loss_trackers = [Average() for x in criteria]

              self.net.eval()

              all_dices, global_dices, loss_values = self.evaluator.evaluate_model()

              loss_report = ' , '.join([f'{lossfunction.name} = {loss_value:.4f}' for lossfunction, loss_value in zip(self.criteria, loss_values)])
              print(f'Epoch = {epoch},  \nDices =    ' , [f'{x}:{y:.3f}' for x, y in zip(self.label_names, np.mean(all_dices, 0))], loss_report,
                    '\nGlobal Dices = ', [f'{x}:{y:.3f}' for x, y in zip(self.label_names, global_dices)],
                    '\n')  
                      
              if self.batches_on_validate == []:
                      self.batches_on_validate = [len(self.loader)*self.batch_size]
              else:
                      self.batches_on_validate.append(self.batches_on_validate[-1]+len(self.loader)*self.batch_size)
        
              self.dice_history.append(all_dices)
                      
        if lr_find:
          print('restoring model and optimizer')
          self.net.load_state_dict(initial_model)  
          self.optimizer.load_state_dict(initial_optimizer)            
        return all_dices, lr_history_this_run, loss_history_this_run
                      
    def warmup(self, warmup_steps = 100, lr_min = 1e-8, lr_max=1e-6):
        self.train(num_epochs = 1, 
              lr_max = lr_max,
              lr_min = lr_min, 
              warmup_optimizer=True,
              warmup_steps = warmup_steps, evaluate = False)
                      
    def lr_find(self, lr_find_steps = 3000, lr_min = 1e-9, lr_max=1e-2):
        all_dices, lr_history_this_run, loss_history_this_run = self.train(num_epochs = 1, 
              lr_max = lr_max,
              lr_min = lr_min, 
              lr_find = True,
              lr_find_steps = lr_find_steps, evaluate = False)
        return lr_history_this_run, loss_history_this_run
                      
                      
    def load_trainer_state(self, filename):
        trainer_state = torch.load(filename)
        self.net.load_state_dict(trainer_state['net_state_dict']) 
        self.optimizer.load_state_dict(trainer_state['optimizer_state_dict']) 


        self.weights = trainer_state['weights']   
        self.use_tqdm  = trainer_state['use_tqdm'] 

        self.lr_history = trainer_state['lr_history']  

        self.loss_history = trainer_state['loss_history']  

        self.gradient_voxels_seen = trainer_state['gradient_voxels_seen']    

        self.batches_on_validate = trainer_state['batches_on_validate']  

        self.dice_history = trainer_state['dice_history'] 
                      
    def save_trainer_state(self, filename):
        trainer_state = {}
        trainer_state['net_state_dict'] = self.net.state_dict()
        trainer_state['optimizer_state_dict'] = self.optimizer.state_dict()


        trainer_state['weights'] = self.weights 
        trainer_state['use_tqdm']= self.use_tqdm 

        trainer_state['lr_history'] = self.lr_history 

        trainer_state['loss_history'] = self.loss_history

        trainer_state['gradient_voxels_seen'] =  self.gradient_voxels_seen 

        trainer_state['batches_on_validate'] = self.batches_on_validate

        trainer_state['dice_history'] = self.dice_history

        torch.save(trainer_state, filename)

    
    
    def train_consistency(self, num_epochs = 50, 
              lr_max = 0.1,
              lr_min = 0.1, 
              scheduler=None, lr_find = False, warmup_optimizer = False,
              lr_find_steps = 100, warmup_steps = 100, evaluate = True, consistency_loss = True,
                         ensemble_loss=True):
        
        lr_history_this_run = []
        loss_history_this_run = []
        
        if lr_find:
            num_epochs = 1
            evaluate = False
            initial_model = copy.deepcopy(self.net.state_dict()) 
            initial_optimizer = copy.deepcopy(self.optimizer.state_dict())
            scheduler = 'exponential'
            num_iter = lr_find_steps
            print('finding optimal learning rate')
        elif warmup_optimizer:
            num_epochs = 1
            evaluate = False
            scheduler = 'exponential'
            num_iter = warmup_steps
            print('warming up optimizer')


          



        if scheduler == 'exponential':
            print('using exponential lr schedule')
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_min      
            schedule = ExponentialLR(self.optimizer, end_lr = lr_max, num_iter = num_iter) 
        elif scheduler == 'cosine':
            print('using per-epoch cosine annealing')
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_max 
            schedule = CosineAnnealingLR(self.optimizer, T_max = len(self.loader), eta_min=lr_min)
        elif scheduler == 'one-cycle':
            div_factor = 25
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_max/div_factor
            schedule = OneCycleLR(self.optimizer, max_lr = lr_max, epochs = num_epochs, 
                                  steps_per_epoch = len(self.loader), div_factor = div_factor)
        else:
            scheduler=None
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_max    
            print('using constant learning rate')

        if self.use_tqdm:
            epoch_iterator = tqdm(range(0, num_epochs))
        else:
            epoch_iterator = range(0, num_epochs)

        for epoch in epoch_iterator:

            if scheduler == 'cosine':
                print(f'resetting lr schedule: lr max = {lr_max}, lr_min = {lr_min}')
                for param_group in self.optimizer.param_groups:
                    param_group['initial_lr'] = lr_max   
                schedule = CosineAnnealingLR(self.optimizer, T_max = len(self.loader), eta_min=lr_min)

            train_loss_trackers = [Average() for x in self.criteria]
                      
            consistency_loss_tracker = Average()

            self.net.train()

            nonempty_batch_count = 0
            
            batch = 0

            for data in  tqdm(self.loader, leave=False):
                batch = batch+1
                if (lr_find and batch == lr_find_steps) or (warmup_optimizer and batch ==  warmup_steps):
                    break
                    
                images, gt, gradient_masks = data
                
                


                #images = images.to(device, non_blocking=True) # non_blocking similar to async=True in v0.3.1
                #targets = targets.to(device, non_blocking=True)
                gradient_masks = gradient_masks.to(self.device, non_blocking=True)
                #well this is ugly
                
                
                targets = torch.from_numpy((self.make_target(gt.unsqueeze(1).numpy())))
                
            

                self.optimizer.zero_grad()
                      
                images = perturb_image(images, channels_to_perturb = [0,1,2,3], 
                                       drop_prob = 0.1, blur_prob = 0.1, degrade_2D_prob=0.1)

                outputs = self.net(Variable(images).to(self.device, non_blocking=True))
                

                gradient_masks = gradient_masks.unsqueeze(1).expand_as(outputs)



                total_loss = 0


                targets = (targets).cuda(non_blocking=True)
                gradient_masks = (gradient_masks).cuda(non_blocking=True) 



                loss_values = [criterion(outputs, targets, gradient_masks) for criterion in self.criteria]

                for tracker, loss_value in zip(train_loss_trackers, loss_values):
                    tracker.update(loss_value.item(), 1)


                total_loss = torch.sum(torch.stack([x*y for x, y in zip(loss_values, self.weights)]))
                
                                            
                images_perturbed = perturb_image(images,  channels_to_perturb = [0,1,2,3],
                                                 drop_prob = 0.5, blur_prob = 0.5, degrade_2D_prob=0.5)
                      
                
                      
                      
                outputs_drop = self.net(Variable(images_perturbed).to(self.device, non_blocking=True))
                




                drop_loss_values = [criterion(outputs_drop, targets, gradient_masks) for criterion in self.criteria]

                for tracker, loss_value in zip(train_loss_trackers, drop_loss_values):
                    tracker.update(loss_value.item(), 1)


                      
                total_loss = 0.5*total_loss + 0.5*torch.sum(torch.stack([x*y for x, y in zip(drop_loss_values, self.weights)]))
                      
                
                      
                consistency_loss  = nn.functional.binary_cross_entropy_with_logits(outputs_drop[:,:len(self.label_names)],
                                                                                   outputs[:,:len(self.label_names)].sigmoid(),
                                                                              reduction = 'none')     
                consistency_loss = consistency_loss*gradient_masks[:,:len(self.label_names)]    
                consistency_loss = consistency_loss.sum()/gradient_masks[:,:len(self.label_names)].sum()
                consistency_loss_tracker.update(consistency_loss.item(), 1)
                      
                if consistency_loss:
                    total_loss = total_loss + consistency_loss
                      
                if ensemble_loss:
                      
                    weighted_sum = outputs*0.5 + outputs_drop* 0.5 
                      
                    weighted_sum_loss_values = [criterion(weighted_sum, targets, gradient_masks) for criterion in self.criteria]

                    for tracker, loss_value in zip(train_loss_trackers, weighted_sum_loss_values):
                        tracker.update(loss_value.item(), 1)
                      
                    total_loss = 0.66*total_loss + 0.33*torch.sum(torch.stack([x*y for x, y in zip(weighted_sum_loss_values, self.weights)]))

                      
                total_loss.backward()

                nn.utils.clip_grad_norm_(self.net.parameters(),1)

                self.optimizer.step()
                
                if not lr_find:
                    self.gradient_voxels_seen.append(gradient_masks.sum().cpu().cuda())
                    self.lr_history.append(self.optimizer.state_dict()["param_groups"][0]["lr"])
                    self.loss_history.append([x.cpu().item() for x in loss_values])
                
                lr_history_this_run.append(self.optimizer.state_dict()["param_groups"][0]["lr"])
                loss_history_this_run.append(total_loss.item())
                loss_report = ' , '.join([f'{lossfunction.name} = {tracker.avg:.4f}' for lossfunction, tracker in zip(self.criteria, train_loss_trackers)])
                consistency_loss_report = f' Consistency loss = {consistency_loss_tracker.avg:.4f}'

                print(loss_report + consistency_loss_report+ f' lr = {self.optimizer.state_dict()["param_groups"][0]["lr"]:.3e}', end="\r")

                if scheduler is not None:
                    schedule.step()


            print(loss_report)


            all_dices = []

            if evaluate:
              #val_loss_trackers = [Average() for x in criteria]

              self.net.eval()

              all_dices, global_dices, loss_values = self.evaluator.evaluate_model()

              loss_report = ' , '.join([f'{lossfunction.name} = {loss_value:.4f}' for lossfunction, loss_value in zip(self.criteria, loss_values)])
              print(f'Epoch = {epoch},  \nDices =    ' , [f'{x}:{y:.3f}' for x, y in zip(self.label_names, np.mean(all_dices, 0))], loss_report,
            '\nGlobal Dices = ', [f'{x}:{y:.3f}' for x, y in zip(self.label_names, global_dices)],
            '\n')  
                      
              if self.batches_on_validate == []:
                      self.batches_on_validate = [len(loader)*self.batch_size]
              else:
                      self.batches_on_validate.append(self.batches_on_validate[-1]+len(loader)*self.batch_size)
        
              self.dice_history.append(all_dices)
                      
        if lr_find:
          print('restoring model and optimizer')
          self.net.load_state_dict(initial_model)  
          self.optimizer.load_state_dict(initial_optimizer)            
        return all_dices, lr_history_this_run, loss_history_this_run
                      