import torch
import numpy as np

from losses import compute_loss_positive, compute_total_variation, compute_loss_negative, compute_gradient_penalty
import time as timelib
from torch.nn.parameter import Parameter
from torch.cuda.amp import GradScaler

from diffusion import NoiseSampler
from resample import create_named_schedule_sampler

from utils import ModelEmaV2, SpectralNormalizer

import math


class Trainer:
    def __init__(self, model, dataset, dataset_fake, args, logger, scheduler, optim_energy, device, dataloader_func=None):
        self.val_pred_real       = []
        self.val_pred_fake       = []
        self.val_loss_energy     = []
        self.val_loss_fake       = []
        self.val_loss_positive   = []
        self.val_loss_negative   = []
        self.val_loss_gradient   = []
        self.val_loss_tv         = []
        self.val_pred_lang       = []
        self.val_pred_fake_min_time = []
        self.val_pred_real_min_time = []
        self.val_pred_lang_iters = []
        
        self.model = model
        self.ema_model = None
        self.ema_started = False
        if hasattr(self.model, "module"):
            self.model_module = self.model.module
        else: 
            self.model_module = self.model
        if args.use_ema == True and logger is not None:
            self.ema_model = ModelEmaV2(self.model_module, decay=args.ema_decay)
            
            
        
        self.dataset = dataset
        self.args = args
        self.logger = logger
        self.scheduler = scheduler
        self.optim_energy = optim_energy
        self.device = device
        self.scaler = GradScaler(enabled=args.mixed_precision)
        if args.use_time_coefficient:
            self.time_coefficient = self.get_time_coefficient(args).to(self.device)
        if args.noisy_real:
            self.noise_sampler = NoiseSampler(args.length_time)#, start=args.noise_start, end=args.noise_end)
        if args.penalty_type == "spectral_norm":
            self.spec_normalizer = SpectralNormalizer(model)
        self.init_data(dataset, dataset_fake, args, dataloader_func)
        self.noise = torch.randn(args.size_batch, args.channel_data, args.height_data, args.height_data).to(self.device)


    def init_data(self, dataset, dataset_fake, args, dataloader_func):
        
        if dataset_fake is not None:
            if dataloader_func is None:
                self.dataloader_fake = torch.utils.data.DataLoader(dataset=dataset_fake, batch_size=args.size_batch, drop_last=True, shuffle=True)
            else: #in case of multi-GPU
                self.dataloader_fake = dataloader_func(dataset_fake, args.size_batch)
            self.data_fake_iter = iter(self.dataloader_fake)
            self.fake_epoch = 0
            if self.args.fake_data_warmup_at == 0:
                self.warmup_fake_dataset(self.args)

        if dataset is not None:
            num_data_fake   = args.num_data_fake
            num_batch_fake  = num_data_fake // args.size_batch
            num_time_subset = num_data_fake // args.length_time

            self.num_data_fake = num_data_fake
            self.num_batch_fake = num_batch_fake
            self.num_time_subset = num_time_subset

            if dataloader_func is None:
                self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.size_batch, drop_last=True, shuffle=True)
            else: #in case of multi-GPU
                self.dataloader = dataloader_func(dataset, args.size_batch)

            self.data_iter = iter(self.dataloader)
            self.real_epoch = 0

            self.weight_sampler = create_named_schedule_sampler("loss-second-moment", args.length_time, normalize_type=args.weighting_type)
        
            

    def start_train(self):
        self.current_step = 0
        for i in range(self.args.length_epoch):
            
            tic = timelib.time()
            self.val_pred_real       = []
            self.val_pred_fake       = []
            self.val_loss_energy     = []
            self.val_loss_fake       = []
            self.val_loss_positive   = []
            self.val_loss_negative   = []
            self.val_loss_gradient   = []
            self.val_loss_tv         = []
            self.val_pred_lang       = []
            self.val_pred_lang_iters = []
            self.val_pred_fake_min_time = []
            self.val_pred_real_min_time = []
            self.val_gradient_penalty = []

            self.current_step = i
            
            self.accumulate_iters = (i+1) % self.args.accumulate_iters
            if i == self.args.ema_start and self.ema_model is not None:
                self.ema_started = True
                # self.ema_model.set(self.model.module)
                self.ema_model.set(self.model_module)
                self.logger.model = self.ema_model.module

            try:
                fake, time, label, idx_batch_fake = next(self.data_fake_iter) 
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                try:
                    self.dataloader_fake.sampler.set_epoch(self.fake_epoch)
                except:
                    pass
                self.fake_epoch += 1
                self.data_fake_iter = iter(self.dataloader_fake)
                fake, time, label, idx_batch_fake = next(self.data_fake_iter) 

            fake, time, label = fake.to(self.device), time.to(self.device), label.to(self.device)
            if self.args.noisy_real == True:
                real, fake, time = self.update_model(fake, time, label, self.args)
            elif self.args.update_order == "model_first":
                real, fake, time = self.update_model(fake, time, label, self.args)
                fake = self.update_fake(fake, time, label, self.args)
                fake, time, label = self.update_time(fake, time, label)
            
            elif self.args.update_order == "model_fake_model":
                real, fake, time = self.update_model(fake, time, label, self.args)
                fake = self.update_fake(fake, time, label, self.args)
                real, fake , time = self.update_model(fake, time, label, self.args)
                fake, time, label = self.update_time(fake, time, label, idx_batch_fake)

            elif self.args.update_order == "fake_model_time":
                fake_original = fake
                fake = self.update_fake(fake, time, label, self.args)
                real, fake, time = self.update_model(fake, time, label, self.args)
                fake, time, label = self.update_time(fake, time, label, fake_original)

            elif self.args.update_order == "fake_first": #TODO
                for fup in range(self.args.fake_update_num):
                    fake_original = fake
                    fake = self.update_fake(fake, time, label, self.args)
                    fake, time, label = self.update_time(fake, time, label, fake_original)
                real, fake, time = self.update_model(fake, time, label, self.args)
                
            self.save_fake_data(fake, time, label, idx_batch_fake)
            toc = timelib.time()

            epoch_time = toc - tic

            if i+1 == self.args.fake_data_warmup_at:
                self.warmup_fake_dataset(self.args)
            
            
            if ((i+1) % self.args.fake_data_warmup_every) == 0 and self.args.fake_data_warmup_every>0:
                if self.args.do_jump_scaling:
                    if self.args.fake_train_jumps > 1:
                        self.args.fake_train_jumps = self.args.fake_train_jumps // self.args.jump_scaling_div
                else:
                    self.warmup_fake_dataset(self.args)

            # -------------------------------------------------------------------
            # save the loss for each epoch
            # -------------------------------------------------------------------
            if self.logger is not None:
                self.logger.push_metrics(self)         
                self.logger.print_training_status(real, fake, time, self, epoch_time)
                # for name, param in self.model.named_parameters():
                #     if len( param.size()) > 2:
                #         print(name, param.size(), torch.linalg.matrix_norm(param, 2).max())
                #     else:
                #         print(name, param.size(), param.max())

            # if (i+1)% 200 == 0 and self.args.lr_fake > 15:

        if self.logger is not None:
            self.logger.save_model(self.optim_energy, self.scheduler, self)
            self.logger.save_graphs(real, fake, time, self)


    def requires_grad(self, parameters, flag=True):
        for p in parameters:
            p.requires_grad = flag

    def update_fake(self, fake, time, label_fake, args, statistics=True):
        ###########################
        # UPDATE FAKE DATA
        ###########################
    
        # -------------------------------------------------------------------
        # update input fake  
        # -------------------------------------------------------------------
        
        fake_param = fake.detach().clone() #Parameter(fake, requires_grad=True) #
        
        self.model.eval() #eval()
        # self.requires_grad(self.model.parameters(), False)
        pred_lang_iters = []

        for k in range(args.length_langevin):  
            fake_param = Parameter(fake_param, requires_grad=True)
            fake_temp   = torch.randn_like(fake_param)

            noise       = torch.randn_like(fake_param)
            # noise = self.noise
            # noise.normal_(0, 1)
            #fake_param.data = fake_param.data + np.sqrt(2 * args.lr_fake) * args.lr_langevin * noise
            fake_param.data = fake_param.data + args.lr_langevin * noise
            pred_negative   = self.model(fake_param, time, label_fake)
            loss_negative   = compute_loss_negative(pred_negative, args.option_loss)
            loss_tv         = compute_total_variation(fake_param)
            loss_fake       = loss_negative + args.weight_total_variation * loss_tv
            loss_fake.backward()

            # fake_param.grad.data.clamp_(-0.01, 0.01)
            pred_lang_iters.append(pred_negative.mean().item())
            if args.random_jumps:
                self.temp_jumps  = torch.randint_like(time, 1, args.fake_train_jumps)
                train_jumps = self.temp_jumps.view(time.size()[0], 1, 1, 1)
            else:
                train_jumps = args.fake_train_jumps
            #adaptive lr based on time
            if args.use_time_coefficient:
                # time_coefficient = (2-time/args.length_time*2).view(args.size_batch, 1, 1, 1)
                # time_coefficient = (args.lr_fake - args.lr_fake*(args.length_time - time)/args.length_time).view(args.size_batch, 1, 1, 1)

                # time_coefficient = (time/(args.length_time-1)*10).view(args.size_batch, 1, 1, 1)
                time_coefficient = self.time_coefficient[time].view(time.size()[0], 1, 1, 1)#.expand([time.size()[0], fake_param.grad.size()[1]])#, fake_param.grad.size()[2], fake_param.grad.size()[3]])
                # print(time_coefficient.size(), fake_param.grad.size())
                
                fake_param.data   = fake_param.data - train_jumps*args.lr_fake * torch.mul(time_coefficient, fake_param.grad) #+ args.lr_langevin * noise
                tempgradmean = fake_param.grad.mean()
            elif args.anneling_fake:
                fake_param.data   = fake_param.data - train_jumps*args.lr_fake/(k+1) * fake_param.grad + ((args.lr_fake/(k+1))**2)/4 * noise
            elif args.momentum_gamma > 0:
                fake_momentum = args.lr_fake *fake_param.grad + args.momentum_gamma*fake_momentum
                fake_param.data   = fake_param.data -  fake_momentum  + args.lr_langevin * noise
                tempgradmean = fake_momentum.mean()
            else:
                # time_temp = time.view(args.size_batch, 1, 1, 1).expand(-1, 1, 32, 32)

                # fake_temp = fake_param.data - ((args.length_time - time_temp ) *args.lr_fake) * fake_param.grad + args.lr_langevin * noise
                
                fake_param.data   = fake_param.data - train_jumps* args.lr_fake * fake_param.grad + args.lr_langevin * noise
                tempgradmean = fake_param.grad.mean()
                # print(tempgradmean)
            # fake_param.grad.detach_()
            fake_param.grad.zero_()
            fake_param = fake_param.detach()
            if args.clamp_fake:
                # fake_param.data.clamp_(min=0.0, max=1.0) #TODO: try to get rid of
                fake_param.data = torch.clip(fake_param.data, -1, 1)

        
        
        # self.dataset_fake[idx_batch_fake] = fake_param.data.detach().cpu().clone()
        
       
    
        # -------------------------------------------------------------------
        # store loss values 
        # -------------------------------------------------------------------
        if statistics:
            self.val_loss_gradient.append(tempgradmean.mean().item()) 

            self.val_loss_fake.append(loss_fake.item())  
            self.val_loss_negative.append(loss_negative.item()) 
            self.val_loss_tv.append(loss_tv.item())
            self.val_pred_lang.append(pred_negative.mean().item())
            self.val_pred_lang_iters.append(pred_lang_iters)


        # self.requires_grad(self.model.parameters(), True)
        return fake_param.data.detach()
                    
        #print('epoch', i, 'lr', optim_energy.param_groups[0]["lr"], flush=True)


    def compute_loss(self, real, fake, time, label_real, label_fake, args, ups=-1):
        # -------------------------------------------------------------------
            # make predictions
            # ------------------------------------------------------------------- 
            real_max_t = []
            fake_max_t = []

            pred_positive       = self.model(real, time, label_real)    
            pred_negative       = self.model(fake, time, label_fake)

            # print(pred_positive,pred_negative ) 

            max_t_indexes = (time == args.length_time-args.fake_train_jumps).nonzero().squeeze()
            real_max_t.append(pred_positive[max_t_indexes].mean().item())
            fake_max_t.append(pred_negative[max_t_indexes].mean().item())

            if args.penalty_type == 'squares':
                time_weights = (time+1)**2
                time_weights = time_weights/(time_weights.max())
                time_weights = time_weights + (1-torch.mean(time_weights))
                loss_regularization = torch.square(time_weights*pred_negative).mean() + torch.square(time_weights*pred_positive).mean()
            elif 'interpolate' in args.penalty_type  or args.penalty_type == 'real_and_fake' or args.penalty_type == 'real':
                

                if args.penalty_type == 'interpolate':
                    reg_real = real


                    # -------------------------------------------------------------------
                    # interpolation
                    # -------------------------------------------------------------------         
                    alpha       = torch.rand(args.size_batch, 1, 1, 1)
                    self.last_alphas = alpha#.flatten()
                    alpha       = alpha.expand_as(reg_real).to(self.device)
                    interpolate = alpha * reg_real.data + (1 - alpha) * fake.data
                    self.last_interpolate = interpolate#.clone()#.cpu().clone().numpy()
                    interpolate = Parameter(interpolate, requires_grad=True)

                    pred_interpolate    = self.model(interpolate, time, label_real)
                    self.last_pred_interpolate = pred_interpolate#.flatten().cpu().clone.numpy()
                    
                    loss_regularization, temp_pen = compute_gradient_penalty(interpolate, pred_interpolate)
                    self.last_pred_interpolate = temp_pen
                elif args.penalty_type == 'interpolate05':
                    reg_real = real


                    # -------------------------------------------------------------------
                    # interpolation
                    # -------------------------------------------------------------------         
                    alpha       = torch.ones(args.size_batch, 1, 1, 1)*0.5
                    self.last_alphas = alpha#.flatten()
                    alpha       = alpha.expand_as(reg_real).to(self.device)
                    interpolate = alpha * reg_real.data + (1 - alpha) * fake.data
                    self.last_interpolate = interpolate#.clone()#.cpu().clone().numpy()
                    interpolate = Parameter(interpolate, requires_grad=True)

                    pred_interpolate    = self.model(interpolate, time, label_real)
                    self.last_pred_interpolate = pred_interpolate#.flatten().cpu().clone.numpy()
                    
                    loss_regularization, temp_pen = compute_gradient_penalty(interpolate, pred_interpolate)
                    self.last_pred_interpolate = temp_pen
                    
                elif args.penalty_type == 'interpolatew':
                    
                    reg_real = real
                    # -------------------------------------------------------------------
                    # interpolation
                    # -------------------------------------------------------------------         
                    alpha       = torch.rand(args.size_batch, 1, 1, 1)
                    self.last_alphas = alpha#.flatten()
                    alpha       = alpha.expand_as(reg_real).to(self.device)
                    interpolate = alpha * reg_real.data + (1 - alpha) * fake.data
                    self.last_interpolate = interpolate#.clone()#.cpu().clone().numpy()
                    interpolate = Parameter(interpolate, requires_grad=True)

                    time_weights = self.time_coefficient[time].view(time.size()[0], 1, 1, 1)
                    # time_weights = self.get_time_weights(time, args)

                    pred_interpolate    =  self.model(interpolate, time, label_real)
                    self.last_pred_interpolate = pred_interpolate#.flatten().cpu().clone.numpy()
                    
                    loss_regularization, temp_pen = compute_gradient_penalty(interpolate, pred_interpolate)
                    loss_regularization = torch.mean(time_weights * temp_pen)
                    self.last_pred_interpolate = temp_pen
                elif args.penalty_type == 'interpolateclip':
                    
                    reg_real = real
                    # -------------------------------------------------------------------
                    # interpolation
                    # -------------------------------------------------------------------         
                    alpha       = torch.rand(args.size_batch, 1, 1, 1)
                    self.last_alphas = alpha#.flatten()
                    alpha       = alpha.expand_as(reg_real).to(self.device)
                    interpolate = alpha * reg_real.data + (1 - alpha) * fake.data
                    self.last_interpolate = interpolate#.clone()#.cpu().clone().numpy()
                    interpolate = Parameter(interpolate, requires_grad=True)


                    pred_interpolate    =  self.model(interpolate, time, label_real)
                    self.last_pred_interpolate = pred_interpolate#.flatten().cpu().clone.numpy()
                    
                    loss_regularization = compute_gradient_penalty(interpolate, pred_interpolate)
                    
                    loss_regularization = torch.clip(loss_regularization, min=0, max=1)
                elif args.penalty_type == 'real_and_fake':
                    reg_real = real.data
                    noise       = torch.randn_like(reg_real)
                    # noise.normal_(-1, 1)
                    reg_real = reg_real + 0.0001*noise
                    reg_real = Parameter(reg_real, requires_grad=True)
                
                    pred_reg_real    = self.model(reg_real, time, label_real)

                    loss_reg_real = compute_gradient_penalty(reg_real, pred_reg_real)

                    reg_fake = fake.data
                    reg_fake = Parameter(reg_fake, requires_grad=True)
                
                    pred_reg_fake    = self.model(reg_fake, time, label_fake)
    
                    loss_reg_fake = compute_gradient_penalty(reg_fake, pred_reg_fake)
                    loss_regularization =  loss_reg_real + loss_reg_fake 

                elif args.penalty_type == 'real':
                    reg_real = real.data
                    noise       = torch.randn_like(reg_real)
                    # noise.normal_(-1, 1)
                    reg_real = reg_real + 0.00005*noise
                    reg_real = Parameter(reg_real, requires_grad=True)
                
                    pred_reg_real    = self.model(reg_real, time, label_real)

                    loss_reg_real = compute_gradient_penalty(reg_real, pred_reg_real)

                    loss_regularization =  loss_reg_real  
            
            elif args.penalty_type == 'spectral_norm':
                # loss_regularization = self.spec_normalizer.spectral_norm_parallel()
                loss_regularization = self.model.module.spectral_norm_parallel()
            elif args.penalty_type == 'l2':
                loss_regularization = (pred_positive ** 2 + pred_negative ** 2).mean()
            else:
                loss_regularization = torch.tensor([0]).to(self.device)
            
            
            if args.use_time_weighting:
                time_weights = self.get_time_weights(time, args)
                loss_positive   = compute_loss_positive(time_weights*pred_negative, time_weights *pred_positive, args.option_loss) #
            else:
                loss_positive   = compute_loss_positive(pred_negative, pred_positive, args.option_loss)
            # print(loss_positive, loss_regularization)
            loss_energy     = loss_positive + args.weight_gradient_penalty * loss_regularization#.detach()
            # loss_energy += 0.2*self.model.module.spectral_norm_parallel()

            if ups == args.model_update_num-1:
                self.val_pred_real.append(pred_positive.mean().item())
                self.val_pred_fake.append(pred_negative.mean().item())
                self.val_loss_energy.append(loss_energy.item()) 
                self.val_loss_positive.append(loss_positive.item())
                self.val_pred_fake_min_time.append(np.ma.array(fake_max_t, mask=np.isnan(fake_max_t)).mean())
                self.val_pred_real_min_time.append(np.ma.array(real_max_t, mask=np.isnan(real_max_t)) .mean())

                self.val_gradient_penalty.append(loss_regularization.item()) 

            if "resampler" in args.weighting_type:
                self.weight_sampler.update_with_local_losses(time, pred_negative - pred_positive)
            return loss_energy


    def update_model(self, fake, time, label_fake, args):
        ###########################
        # UPDATE ENERGY MODEL
        ###########################
        self.model.train()
        torch.autograd.set_detect_anomaly(True)
        for ups in range(0, args.model_update_num):
            
            try:
                real, label_real = next(self.data_iter) 
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                try:
                    self.dataloader.sampler.set_epoch(self.real_epoch)
                except:
                    pass
                self.real_epoch += 1
                self.data_iter = iter(self.dataloader)
                real, label_real = next(self.data_iter) 

            real = real.to(self.device)
            label_real = label_real.to(self.device).unsqueeze(1)

            # real_clean = real.clone()
            if args.noisy_real == True:
                try:
                    fake, label_fake = next(self.data_iter) 
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    try:
                        self.dataloader.sampler.set_epoch(self.real_epoch)
                    except:
                        pass
                    self.real_epoch += 1
                    self.data_iter = iter(self.dataloader)
                    fake, label_fake = next(self.data_iter)

                fake = fake.to(self.device)
                label_fake = label_fake.to(self.device).unsqueeze(1)

                time =  torch.randint(0, args.length_time, (args.size_batch,)).to(self.device)

                inv_t = args.length_time - time -1
                real_noisy = self.noise_sampler.forward_diffusion_sample(fake, inv_t)
                fake = real_noisy #torch.ones_like(fake).to(self.device) # real
                # real = self.q_sample(real, time, noise)

            loss_energy = self.compute_loss( real, fake, time, label_real, label_fake, args, ups)
            
            # -------------------------------------------------------------------
            # update energy model 
            # -------------------------------------------------------------------   
            if args.option_optim == "sam":
                def closure():
                    loss = self.compute_loss(real, fake, time, label_real, label_fake, args, ups)
                    loss.backward()
                    return loss
                self.optim_energy.zero_grad()
                with self.model.no_sync():
                    loss_energy.backward()
                self.optim_energy.step(closure)
                
                self.scheduler.step()
                if self.ema_started: 
                    # self.ema_model.update(self.model.module)
                    self.ema_model.update(self.model_module)
            else:   
                self.optim_energy.zero_grad()

                if args.mixed_precision:
                    self.scaler.scale(loss_energy/args.accumulate_iters).backward()
                else:
                    (loss_energy/args.accumulate_iters).backward()
                if self.accumulate_iters == 0:
                    if args.mixed_precision:
                        self.scaler.step(self.optim_energy)
                        self.scaler.update()
                    else:
                        
                        self.optim_energy.step()
                    self.scheduler.step()
                    if self.ema_started:
                        # self.ema_model.update(self.model.module)
                        self.ema_model.update(self.model_module)
            
            if np.isnan(loss_energy.item()):
                sys.exit('error')
        

        return real, fake, time


    def update_time(self, fake, time, label, fake_original):
        
        if self.args.use_time:
            if self.args.random_jumps:
                time += self.temp_jumps
            else:
                time += self.args.fake_train_jumps
            
            # MAKE FRESH FAKES FOR ELEMENTS THAT ARE PAST MAX TIME
            maxt_indexes = (time >= self.args.length_time).nonzero().squeeze()
            if self.args.buffer_type == "replay_buffer" or self.args.buffer_type == "replay_buffer_time" :
                # print(maxt_indexes)
                time[maxt_indexes] = self.args.length_time-1
            else:
                # time[indexes] -= self.args.length_time
                # print(maxt_indexes.nelement())
                indexes = torch.logical_and(time >= self.args.length_time, torch.rand(time.nelement()).to(self.device) < self.args.refresh_rate).nonzero().squeeze()

                time[maxt_indexes] = self.args.length_time-1
                fake[indexes] = fake_original[indexes]
                time[indexes] *= 0
                # if indexes.nelement() > 1: #todo make it work for 1 too
                #     indexes = indexes[torch.rand(indexes.nelement()) >= (1-self.args.refresh_rate)]
                
                label[indexes] = torch.randint(0, 9, (indexes.nelement(), 1)).to(label.device)
                
                # fake[indexes] = torch.randn(indexes.nelement(), self.args.channel_data, self.args.height_data, self.args.width_data).to(fake.device)
                fake[indexes] = self.dataloader_fake.dataset.init_random(self.args, indexes.nelement(), label[indexes]).to(fake.device)


                # init_at_time =  (self.current_step - 10000) // 5000
                # if init_at_time > 16:
                
                if indexes.nelement()==1:
                    indexes = indexes.unsqueeze(0)
                    
                # if self.args.init_at_time > 0 and indexes.nelement()>0 and self.current_step > self.args.ema_start*2:
                    
                #     temp_f = fake[indexes]
                #     temp_t = time[indexes]
                #     temp_l = label[indexes]
                #     # print(temp_t.size())
                #     for i in range(self.args.init_at_time):
                #         if torch.rand(1).item() < self.args.refresh_rate2: #0.5 ginga
                #             break
                #         temp_f = self.update_fake(temp_f, temp_t, temp_l, self.args, statistics=False)
                #         temp_t += 1
                #     fake[indexes] = temp_f
                #     time[indexes] = temp_t

        return fake, time, label
        
    def save_fake_data(self, fake, time, label, idx_fake):
        fake, time, label = fake.detach().cpu().clone(), time.detach().cpu().clone(), label.detach().cpu().clone()

        self.dataloader_fake.dataset._update_data(fake, idx_fake)
        self.dataloader_fake.dataset._update_time(time, idx_fake)
        self.dataloader_fake.dataset._update_label(label, idx_fake)
        

    def warmup_fake_dataset(self, args, inference=False):
        needed_iters = args.length_time*args.num_data_fake//(args.size_batch*args.world_size*args.fake_train_jumps)
        dummy_trainer = Trainer(self.model, None, None, args, None, None, None, self.device)
        # if inference == True

        for i in range(needed_iters):
            try:
                fake, time, label, idx_batch_fake = next(self.data_fake_iter) 
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                self.dataloader_fake.sampler.set_epoch(self.fake_epoch)
                self.fake_epoch += 1
                self.data_fake_iter = iter(self.dataloader_fake)
                fake, time, label, idx_batch_fake = next(self.data_fake_iter) 

            fake, time, label = fake.to(self.device), time.to(self.device), label.to(self.device)
            
            # print(idx_batch_fake)
            fake_original = fake
            fake = dummy_trainer.update_fake(fake, time, label, self.args)
            
            if inference == True and time[0] + args.fake_train_jumps >= args.length_time:
                self.save_fake_data(fake, time, label, idx_batch_fake)
                # continue
            else:
                fake, time, label = self.update_time(fake, time, label, fake_original)
                self.save_fake_data(fake, time, label, idx_batch_fake)
            print ("RESAMPLING:",i+1,"out of", needed_iters ,"warmup steps complete", end='\r')
        print('')

    def get_time_weights(self, time_raw, args):
        time = time_raw.clone()
        # if args.inference_at_t_minus_2:
        #     indexes = (time >= args.length_time-1).nonzero().squeeze()
        #     time[indexes] = args.length_time-2
        #     args.length_time -= 1
        
        if args.weighting_type == "linear":
            time_weights = (time+1)/args.length_time *2
        elif args.weighting_type == "sigmoid6":
            time_weights = (time+1)/args.length_time
            time_weights = 2/(1+torch.exp(-6*(time_weights-0.5)))
        elif args.weighting_type == "sigmoid12":
            time_weights = (time+1)/args.length_time
            time_weights = 2/(1+torch.exp(-12*(time_weights-0.5)))
        elif args.weighting_type == "sigmoid18":
            time_weights = (time+1)/args.length_time
            time_weights = 2/(1+torch.exp(-18*(time_weights-0.5)))
        elif args.weighting_type == "logsigmoid6":
            time_weights = (time+1)/args.length_time
            time_weights = 4/(1+torch.exp(-6*(time_weights-0)))-2
        elif args.weighting_type == "logsigmoid12":
            time_weights = (time+1)/args.length_time
            time_weights = 4/(1+torch.exp(-12*(time_weights-0)))-2
        elif args.weighting_type == "linear02":
            b = 0.8
            a = 0.2
            time_weights = a + (time+1)*(b-a)/args.length_time 
            time_weights *= 2
        elif args.weighting_type == "linear005":  
            b = 0.95
            a = 0.05
            time_weights = a + (time+1)*(b-a)/args.length_time 
            time_weights *= 2
        #elif args.weighting_type == "resampler" or args.weighting_type == "resampler_norm" or args.weighting_type == "resampler_mean_one":
        elif args.weighting_type == "sigmoid005":
            b = 0.95
            a = 0.05
            time_weights = a + (time+1)*(b-a)/args.length_time 
            time_weights = 2/(1+torch.exp(-12*(time_weights-0.5)))
        elif "resampler" in args.weighting_type:
            time_weights = self.weight_sampler.sample(time, self.device)
            # if ups == args.model_update_num-1: # BIG CHANGE POTENTIALLY
            #     self.weight_sampler.update_with_local_losses(time, pred_negative - pred_positive)

        elif args.weighting_type == "cosine":
            # lfunc = lambda t: torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            # test = []
            # for i in range(args.length_time,0,-1):
            #     t1 = i / args.length_time
            #     test.append(lfunc(t1)*2)
            t = (args.length_time -time-1) / args.length_time
            time_weights = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            time_weights = time_weights*2
        elif args.weighting_type == "cosine05":
            t = (args.length_time -time-1) / args.length_time
            time_weights = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            time_weights = time_weights+0.5

        elif args.weighting_type == "exp50000":
            all_times = torch.arange(0, args.length_time).to(self.device)
            time_weights = 50000**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/50000 #normalize
            time_weights *= 1/time_weights.mean()
            time_weights = time_weights[time]
        
        elif args.weighting_type == "exp20000":
            all_times = torch.arange(0, args.length_time).to(self.device)
            time_weights = 20000**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/20000 #normalize
            time_weights *= 1/time_weights.mean()
            time_weights = time_weights[time]

        elif args.weighting_type == "exp10000":
            all_times = torch.arange(0, args.length_time).to(self.device)
            time_weights = 10000**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/10000 #normalize
            time_weights *= 1/time_weights.mean()
            time_weights = time_weights[time]

        elif args.weighting_type == "exp5000":
            all_times = torch.arange(0, args.length_time).to(self.device)
            time_weights = 5000**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/5000 #normalize
            time_weights *= 1/time_weights.mean()
            time_weights = time_weights[time]

        elif args.weighting_type == "exp3000":
            all_times = torch.arange(0, args.length_time).to(self.device)
            time_weights = 3000**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/3000 #normalize
            time_weights *= 1/time_weights.mean()
            time_weights = time_weights[time]

        elif args.weighting_type == "exp2000":
            all_times = torch.arange(0, args.length_time).to(self.device)
            time_weights = 2000**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/2000 #normalize
            time_weights *= 1/time_weights.mean()
            time_weights = time_weights[time]

        elif args.weighting_type == "exp1500":
            all_times = torch.arange(0, args.length_time).to(self.device)
            time_weights = 1500**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/1500 #normalize
            time_weights *= 1/time_weights.mean()
            time_weights = time_weights[time]

        elif args.weighting_type == "exp1000":
            all_times = torch.arange(0, args.length_time).to(self.device)
            time_weights = 1000**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/1000 #normalize
            time_weights *= 1/time_weights.mean()
            time_weights = time_weights[time]

        elif args.weighting_type == "exp500":
            all_times = torch.arange(0, args.length_time).to(self.device)
            time_weights = 500**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/500 #normalize
            time_weights *= 1/time_weights.mean()
            time_weights = time_weights[time]
        elif args.weighting_type == "exp200":
            all_times = torch.arange(0, args.length_time).to(self.device)
            time_weights = 200**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/200 #normalize
            time_weights *= 1/time_weights.mean()
            time_weights = time_weights[time]
        elif args.weighting_type == "exp100":
            all_times = torch.arange(0, args.length_time).to(self.device)
            time_weights = 100**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/100 #normalize
            time_weights *= 1/time_weights.mean()
            time_weights = time_weights[time]
        elif args.weighting_type == "exp20":
            all_times = torch.arange(0, args.length_time).to(self.device)
            time_weights = 20**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/20 #normalize
            time_weights *= 1/time_weights.mean()
            time_weights = time_weights[time]
        elif args.weighting_type == "exp500_raw":
            time_weights = 500**(time/(args.length_time-1)) 
        elif args.weighting_type == "exp200_raw":
            time_weights = 200**(time/(args.length_time-1)) 
        elif args.weighting_type == "exp20_raw":
            time_weights = 20**(time/(args.length_time-1))
        elif args.weighting_type == "exp50_raw":
            time_weights = 50**(time/(args.length_time-1)) # 
        elif args.weighting_type == "none":
            time_weights = time*0 + 1.00005
        elif args.weighting_type == "schedule_zeros":
            numbers = (self.current_step - 10000) // 5000
            if numbers < 0:
                numbers = 0
            time_weights = time*0 + 1.00005
            total = time_weights.sum()
            time_weights[time<numbers] = 0
            new_total = time_weights.sum()
            time_weights *= total/new_total
        elif args.weighting_type == "schedule_sigmoid6":
            time_weights = (time+1)/args.length_time
            time_weights = 2/(1+torch.exp(-6*(time_weights-0.5)))
            numbers = (self.current_step - 10000) // 2500
            if numbers < 0:
                numbers = 0

            time_weights[time<numbers] *= 0.1
        elif args.weighting_type == "schedule_sigmoid62":
            time_weights = (time+1)/args.length_time
            time_weights = 2/(1+torch.exp(-6*(time_weights-0.5)))
            numbers = (self.current_step - 10000) // 2500
            if numbers < 0:
                numbers = 0
            elif numbers > 7:
                numbers = 7
            time_weights[time<numbers] *= 0.01
            # new_total = time_weights.sum()
            # time_weights *= total/new_total
        # if args.inference_at_t_minus_2:
        #     args.length_time += 1
        return time_weights

    def get_time_coefficient(self, args): # TODO: THIS DOES NOT WORK SHOULD FIX
        time = torch.arange(0, args.length_time)
        time = args.length_time - time
        if args.coefficient_type == "linear":
            time_weights = (time+1)/args.length_time *2
        elif args.weighting_type == "linear02":
            b = 0.8
            a = 0.2
            time_weights = a + (time+1)*(b-a)/args.length_time 
            time_weights *= 2
        elif args.weighting_type == "linear005":  
            b = 0.95
            a = 0.05
            time_weights = a + (time+1)*(b-a)/args.length_time 
            time_weights *= 2
        elif args.coefficient_type == "sigmoid6":
            time_weights = (time+1)/args.length_time
            time_weights = 2/(1+torch.exp(-6*(time_weights-0.5)))
        elif args.coefficient_type == "sigmoid12":
            time_weights = (time+1)/args.length_time
            time_weights = 2/(1+torch.exp(-12*(time_weights-0.5)))
        elif args.coefficient_type == "sigmoid18":
            time_weights = (time+1)/args.length_time
            time_weights = 2/(1+torch.exp(-18*(time_weights-0.5)))
        elif args.coefficient_type == "cosine":
            time = torch.arange(0, self.args.length_time).type(torch.LongTensor)
            t = (args.length_time - time-1) / args.length_time
            time_weights = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            time_weights = time_weights+0.5
            # time_weights = self.get_time_weights(time, args)
            time_weights = time_weights/2
            time_weights = (1 - time_weights)/(1- time_weights)**0.5
            time_weights = time_weights*2
        elif args.coefficient_type == "cosine05":
            t = (args.length_time -time-1) / args.length_time
            time_weights = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            time_weights = time_weights+0.5

            time_weights = time_weights/2
            time_weights = (1 - time_weights)/(1- time_weights)**0.5
            time_weights = time_weights*2
        elif args.coefficient_type == "exp200":
            all_times = torch.arange(0, args.length_time).to(self.device)
            all_times = args.length_time-1 - all_times
            time_weights = 200**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/200 #normalize
            time_weights *= 1/time_weights.mean()
        elif args.coefficient_type == "exp20":
            all_times = torch.arange(0, args.length_time).to(self.device)
            all_times = args.length_time-1 - all_times
            time_weights = 20**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/20 #normalize
            time_weights *= 1/time_weights.mean()
        elif args.coefficient_type == "exp10":
            all_times = torch.arange(0, args.length_time).to(self.device)
            all_times = args.length_time-1 - all_times
            time_weights = 10**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/10 #normalize
            time_weights *= 1/time_weights.mean()
        elif args.coefficient_type == "exp5":
            all_times = torch.arange(0, args.length_time).to(self.device)
            all_times = args.length_time-1 - all_times
            time_weights = 5**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/5 #normalize
            time_weights *= 1/time_weights.mean()
        elif args.coefficient_type == "exp3":
            all_times = torch.arange(0, args.length_time).to(self.device)
            all_times = args.length_time-1 - all_times
            time_weights = 3**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/3 #normalize
            time_weights *= 1/time_weights.mean()
        elif args.coefficient_type == "exp2":
            all_times = torch.arange(0, args.length_time).to(self.device)
            all_times = args.length_time-1 - all_times
            time_weights = 2**(all_times/(args.length_time-1)) # 
            # time_weights = time_weights + (1-time_weights.mean()) #adjust to be mean 1
            time_weights = time_weights/2 #normalize
            time_weights *= 1/time_weights.mean()
        elif args.coefficient_type == "none":
            time_weights = torch.ones(self.args.length_time)#.type(torch.LongTensor)
        return time_weights


    