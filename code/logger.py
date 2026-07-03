import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import datetime 
import numpy as np
import json

from inference.inference import inference
from inference.figures import save_inference, save_50, compute_fid, save_spike_train
from torchvision.utils import make_grid, save_image
from torch.nn.parameter import Parameter

class Logger:
    def __init__(self, model, dataset, args, dataloader_func=None, fake_data_init_func=None):
        self.args = args
        length_epoch = args.length_epoch
        self.model = model
        self.val_pred_real_mean      = np.zeros(length_epoch)
        self.val_pred_real_std       = np.zeros(length_epoch)
        self.val_pred_fake_mean      = np.zeros(length_epoch)
        self.val_pred_fake_std       = np.zeros(length_epoch)
        self.val_loss_energy_mean    = np.zeros(length_epoch)
        self.val_loss_energy_std     = np.zeros(length_epoch)
        self.val_loss_fake_mean      = np.zeros(length_epoch)
        self.val_loss_fake_std       = np.zeros(length_epoch)
        self.val_loss_positive_mean  = np.zeros(length_epoch)
        self.val_loss_positive_std   = np.zeros(length_epoch)
        self.val_loss_negative_mean  = np.zeros(length_epoch)
        self.val_loss_negative_std   = np.zeros(length_epoch)
        self.val_loss_gradient_mean  = np.zeros(length_epoch)
        self.val_loss_gradient_std   = np.zeros(length_epoch)
        self.val_loss_tv_mean        = np.zeros(length_epoch)
        self.val_loss_tv_std         = np.zeros(length_epoch)
        self.val_lang_mean           = np.zeros(length_epoch)
        self.val_lang_std            = np.zeros(length_epoch)
        self.val_pred_fake_min_time  = np.zeros(length_epoch)
        self.val_pred_real_min_time  = np.zeros(length_epoch)
        self.val_gradient_penalty  = np.zeros(length_epoch)
        
        self.fid_scores              = []
        self.val_pred_lang_iters     = []
        self.current_step = 0
        self.init_file_paths(args)

        # torch.randn(args.size_batch, args.channel_data, args.height_data, args.width_data)
        
        
        # self.sample_fake            = self.sample_fake * 2 - 1
        self.sample_label_fake      = torch.randint(0, 9, (args.size_batch, 1))
        self.sample_fake            = fake_data_init_func(args, args.size_batch, self.sample_label_fake)

        self.did_spike = False

        self.scheduler_vals(args)
        if args.compute_fid:
            if dataloader_func is None:
	            self.tmp_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.size_batch, shuffle=True, drop_last=True)
            else: 
                self.tmp_dataloader = dataloader_func(dataset, args.size_batch)
    
    def scheduler_vals(self, args):
        class DummyModel(nn.Module):
            def __init__(self, input_size, output_size):
                super(DummyModel, self).__init__()
                self.fc = nn.Linear(input_size, output_size)

            def forward(self, x):
                x = self.fc(x)
                return x

        from arguments import get_scheduler
        dummy_param = DummyModel(2, 4).parameters()
        dummy_optim = torch.optim.SGD(dummy_param, lr=args.lr_energy)
        dummy_scheduler =  get_scheduler(args, dummy_optim)
        self.lr_schdeule  = np.zeros(args.length_epoch*args.model_update_num)
        for i in range(args.length_epoch*args.model_update_num):
            self.lr_schdeule[i] = dummy_scheduler.get_last_lr()[0]
            dummy_scheduler.step()
            

    def init_file_paths(self, args):
        # ======================================================================
        # path for the results
        # ======================================================================
        now         = datetime.datetime.now()
        date_stamp  = now.strftime('%Y_%m_%d') 
        self.time_stamp  = now.strftime('%H_%M_%S') 

        dir_dataset = args.name_model.upper()
        self.dir_dataset = dir_dataset
        dir_figure  = os.path.join(args.dir_work, 'figure')
        dir_option  = os.path.join(args.dir_work, 'option')
        dir_result  = os.path.join(args.dir_work, 'result')
        dir_model   = os.path.join(args.dir_work, 'model')
        dir_graph  = os.path.join(args.dir_work, 'graph')

        path_figure = os.path.join(dir_figure, dir_dataset)
        path_option = os.path.join(dir_option, dir_dataset)
        path_result = os.path.join(dir_result, dir_dataset)
        path_model  = os.path.join(dir_model, dir_dataset)
        path_graph  = os.path.join(dir_graph, dir_dataset)

        self.date_figure = os.path.join(path_figure, date_stamp)
        date_option = os.path.join(path_option, date_stamp)
        date_result = os.path.join(path_result, date_stamp)
        date_model  = os.path.join(path_model, date_stamp)
        date_graph  = os.path.join(path_graph, date_stamp)

        self.file_figure = os.path.join(self.date_figure, '{}.png'.format(self.time_stamp))
        self.file_option = os.path.join(date_option, '{}.json'.format(self.time_stamp))
        self.file_result = os.path.join(date_result, '{}.txt'.format(self.time_stamp))
        self.file_model  = os.path.join(date_model, '{}.pth'.format(self.time_stamp))
        self.file_graph = os.path.join(date_graph, '{}.png'.format(self.time_stamp))

        if not os.path.exists(dir_figure):
            os.mkdir(dir_figure)

        if not os.path.exists(dir_option):
            os.mkdir(dir_option)

        if not os.path.exists(dir_result):
            os.mkdir(dir_result)

        if not os.path.exists(dir_model):
            os.mkdir(dir_model)

        if not os.path.exists(dir_graph):
            os.mkdir(dir_graph)

        if not os.path.exists(path_figure):
            os.mkdir(path_figure)

        if not os.path.exists(path_option):
            os.mkdir(path_option)

        if not os.path.exists(path_result):
            os.mkdir(path_result)

        if not os.path.exists(path_model):
            os.mkdir(path_model)

        if not os.path.exists(path_graph):
            os.mkdir(path_graph)

        if not os.path.exists(self.date_figure):
            os.mkdir(self.date_figure)

        if not os.path.exists(date_option):
            os.mkdir(date_option)

        if not os.path.exists(date_result):
            os.mkdir(date_result)
            
        if not os.path.exists(date_model):
            os.mkdir(date_model)

        if not os.path.exists(date_graph):
            os.mkdir(date_graph)

        # -------------------------------------------------------------------
        # save the options
        # -------------------------------------------------------------------         
        with open(self.file_option, 'w') as f:
            f.write(json.dumps(vars(args),
                indent=4
            ))
        f.close()

    def push_metrics(self, trainer):# val_loss_energy, val_loss_fake, val_loss_positive, val_loss_negative, val_loss_gradient, val_loss_tv, val_pred_real, val_pred_fake, val_lang):

        self.val_pred_real_mean[self.current_step]    = np.mean(trainer.val_pred_real)
        self.val_pred_real_std[self.current_step]    = np.std(trainer.val_pred_real)
        self.val_pred_fake_mean[self.current_step]   = np.mean(trainer.val_pred_fake)
        self.val_pred_fake_std[self.current_step]    = np.std(trainer.val_pred_fake)
        self.val_loss_energy_mean[self.current_step]     = np.mean(trainer.val_loss_energy)
        self.val_loss_energy_std[self.current_step]      = np.std(trainer.val_loss_energy)
        self.val_loss_fake_mean[self.current_step]       = np.mean(trainer.val_loss_fake)
        self.val_loss_fake_std[self.current_step]        = np.std(trainer.val_loss_fake)
        self.val_loss_positive_mean[self.current_step]   = np.mean(trainer.val_loss_positive)
        self.val_loss_positive_std[self.current_step]    = np.std(trainer.val_loss_positive)
        self.val_loss_negative_mean[self.current_step]   = np.mean(trainer.val_loss_negative)
        self.val_loss_negative_std[self.current_step]    = np.std(trainer.val_loss_negative)
        self.val_loss_gradient_mean[self.current_step]   = np.mean(trainer.val_loss_gradient)
        self.val_loss_gradient_std[self.current_step]    = np.std(trainer.val_loss_gradient)
        self.val_loss_tv_mean[self.current_step]         = np.mean(trainer.val_loss_tv)
        self.val_loss_tv_std[self.current_step]          = np.std(trainer.val_loss_tv)
        self.val_lang_mean[self.current_step]            = np.mean(trainer.val_pred_lang)
        self.val_lang_std[self.current_step]             = np.std(trainer.val_pred_lang)
        self.val_pred_fake_min_time[self.current_step]   = np.mean(trainer.val_pred_fake_min_time)
        self.val_pred_real_min_time[self.current_step]   = np.mean(trainer.val_pred_real_min_time)
        self.val_gradient_penalty[self.current_step]   = np.mean(trainer.val_gradient_penalty)
        

        self.val_pred_lang_iters.append(trainer.val_pred_lang_iters)

    def print_training_status(self, real, fake, time, trainer, epoch_time):
        # Compute time left
        

        learning_rate = trainer.scheduler.get_last_lr()[0]
        time_left_sec = (self.args.length_epoch- (self.current_step+1)) * epoch_time
        time_left_sec = int(time_left_sec) #.astype(np.int32)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60, time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>9}"

        log = '[%4d/%4d] pred(real)=%8.4f, pred(fake)=%8.4f, loss(pos)=%8.4f, loss(neg)=%8.4f, loss(grad)=%14.10f, penalty=%8.4f lr(energy)=%7.12f' % (self.current_step, 
                self.args.length_epoch, self.val_pred_real_mean[self.current_step], self.val_pred_fake_mean[self.current_step],
                self.val_loss_energy_mean[self.current_step], self.val_loss_negative_mean[self.current_step], 
                self.val_loss_gradient_mean[self.current_step], self.val_gradient_penalty[self.current_step], learning_rate)
        print(log+', eta='+time_left_hms, flush=True)
        
        if self.current_step % self.args.print_freq == 0: 
            filename = os.path.join(self.date_figure, '{0}_epoch_{1:03d}.png'.format(self.time_stamp, self.current_step))
            self.save_iteration_info(real, fake, time, trainer, filename)

        if (self.current_step+1) % self.args.validation_freq == 0:
            self.did_spike = False
            print("Running test inference at epoch", self.current_step)
            inference_data, label_fake = inference(self.model, self.sample_fake, self.sample_label_fake, self.args)
            inference_data = np.clip(np.array(inference_data), -1, 1)
            
            save_inference(inference_data, self.file_figure[:-4] + "_test_" + str(self.current_step), label_fake, self.args)
            if self.args.inference_at_t_minus_2:
                sample_t = 2
                save_50(inference_data,  self.file_figure[:-4] + "_test50_" + str(self.current_step), label_fake, self.args, sample_t=sample_t)
            else:
                sample_t = 1
                save_50(inference_data,  self.file_figure[:-4] + "_test50_" + str(self.current_step), label_fake, self.args)
            
            

            self.save_model(trainer.optim_energy, trainer.scheduler, trainer, path=self.file_model[:-4] + "_" + str(self.current_step)+ ".pth")
            if self.args.compute_fid:
                fake_batch=[]
                fake_batch.append(torch.from_numpy(inference_data[-sample_t]))#[0:32])
                try:
                    fid = compute_fid(self.tmp_dataloader, fake_batch, self.args.size_batch) # 32)#
                    self.fid_scores.append(fid)
                except ValueError as e:
                    print(e)
                    self.fid_scores.append(-1)
            self.save_graphs(real, fake, time, trainer)
            print(self.file_figure[:-4])

        # REPORT SPIKES DURING TRAINING
        # if self.val_gradient_penalty[self.current_step] > 2 and self.current_step > 5000 and self.did_spike==False: #  self.val_gradient_penalty[self.current_step] > 2 or 
        #     self.did_spike = True
        #     print("LOSS SPIKE DETECTED")
        #     file_fake   = self.file_figure[:-4] + "_penalty_spike_train_" + str(self.current_step) + ".png"

        #     save_spike_train(fake.cpu().clone().numpy(), file_fake, self.args, trainer.last_alphas.flatten().cpu().numpy())
            

        #     print(trainer.last_pred_interpolate.shape)
        #     file_fake   = self.file_figure[:-4] + "_penalty_spike_interpolate_" + str(self.current_step) + ".png"
        #     save_spike_train(trainer.last_interpolate.detach().cpu().numpy(), file_fake, self.args, trainer.last_pred_interpolate.flatten().detach().cpu().numpy())



        self.current_step += 1
        
    def save_iteration_info(self, real, fake, time, trainer, filename):

        dataset_time = trainer.dataloader_fake.dataset.time
        dataset_fake = trainer.dataloader_fake.dataset.data
             
        real_plot = real.detach().cpu().numpy().squeeze()
        fake_plot = fake.detach().cpu().numpy().squeeze()
        time_plot = time.detach().cpu().numpy()

        

        max_time_indices = (dataset_time == np.minimum(self.args.length_time-1, self.current_step)).nonzero().squeeze()
        max_time_indices = torch.argsort(dataset_time, dim=0, descending=True)
        max_time_plot = dataset_fake[max_time_indices].numpy()
        
        

        nRow    = 4 
        nCol    = 4
        fSize   = 3

        plt_cmap = 'gray'
        max_time_plot = np.clip(np.moveaxis(max_time_plot, 1, -1), -1, 1)


        dataset_fake_moved = np.clip(np.moveaxis(np.array(dataset_fake[:4]), 1, -1), -1, 1)

        if self.args.channel_data ==3:
            real_plot = np.moveaxis(real_plot, 1, -1)
            fake_plot = np.clip(np.moveaxis(fake_plot, 1, -1), -1, 1)
            real_plot = (real_plot+1)/2
            fake_plot = (fake_plot+1)/2
            max_time_plot = (max_time_plot+1)/2
            dataset_fake_moved = (dataset_fake_moved+1)/2
            plt_cmap = 'viridis'

        fig, ax = plt.subplots(nRow, nCol+1, figsize=(fSize * (nCol+1), fSize * nRow))

        for c in range(nCol):
            im = ax[0][c].imshow(real_plot[c], cmap=plt_cmap)

            ax[1][c].set_title('time = ' + str(time_plot[c]))
            im = ax[1][c].imshow(fake_plot[c], cmap=plt_cmap)

            ax[2][c].set_title('time = ' + str(np.minimum(self.args.length_time-1, self.current_step)))
            im = ax[2][c].imshow(max_time_plot[c], cmap=plt_cmap)

            ax[3][c].set_title('time = ' + str(dataset_time[c].item()))
            im = ax[3][c].imshow(dataset_fake_moved[c], cmap=plt_cmap)
        
        ax[0][nCol].set_title('time distribution')
        kwargs = dict(alpha=0.5, bins=self.args.length_time, density=True, stacked=True)
        ax[0][nCol].hist(trainer.dataloader_fake.dataset.time, **kwargs, color='b', label='times')
        ax[0][nCol].legend()

        ax[1][nCol].set_title('time distribution batch')
        kwargs = dict(alpha=0.5, bins=self.args.length_time, density=True, stacked=True)
        ax[1][nCol].hist(time_plot, **kwargs, color='g', label='times')
        ax[1][nCol].legend()

        device = torch.device('cpu')

        time = torch.range(0, trainer.args.length_time-1).type(torch.LongTensor)
        weights = trainer.get_time_weights(time, trainer.args).to(device)
        ax[3][nCol].set_title('loss weights mean='+str(weights.mean().item()))

        ax[3][nCol].plot(weights, color='blue')
        
        weights1 = trainer.get_time_weights(time[0:trainer.args.length_time//2], trainer.args).to(device)# trainer.weight_sampler.sample(, device)

        weights2 = trainer.get_time_weights(time[trainer.args.length_time//2:trainer.args.length_time-1], trainer.args).to(device) #trainer.weight_sampler.sample(time[trainer.args.length_time//2:trainer.args.length_time-1], device)
        weights_all = torch.cat((weights1, weights2), 0)
        if trainer.args.use_time_coefficient:
            ax[3][nCol].plot(trainer.time_coefficient[time].to(device), color='red')

        plt.tight_layout()
        fig.savefig(filename, bbox_inches='tight', dpi=100)
        plt.close(fig)
        
    def save_model(self, optimizer, scheduler, trainer, path=''):
        
        # -------------------------------------------------------------------
        # save the trained model
        # -------------------------------------------------------------------  
        if path == '':
            path = self.file_model
        time = torch.range(0, trainer.args.length_time-1).type(torch.LongTensor)
        weights = trainer.get_time_weights(time, trainer.args)
        torch.save({
            'model_state_dict'      : self.model.state_dict(), 
            'scheduler_state_dict'  : scheduler.state_dict(),
            'optimizer_state_dict'  : optimizer.state_dict(),
            'loss_weights' : weights,
        }, path)


    def save_graphs(self, real, fake, time, trainer):
        batch_size  = fake.shape[0]

        nrow        = int(np.ceil(np.sqrt(batch_size)))
        file_fake   = self.file_figure[:-4] + "_train_" + str(self.current_step) + ".png"

        grid_fake   = make_grid((fake+1)/2, nrow=nrow, normalize=False)
        save_image(grid_fake, file_fake)

        # -------------------------------------------------------------------
        # save the figure
        # -------------------------------------------------------------------      
        np.savetxt(self.file_result, np.array([self.val_pred_real_mean, self.val_loss_energy_mean, self.val_loss_fake_mean]))
        real_plot = real.detach().cpu().numpy().squeeze()
        fake_plot = fake.detach().cpu().numpy().squeeze()
        time_plot = time.detach().cpu().numpy().squeeze()
        nRow    = 4
        nCol    = 4
        fSize   = 3
            
        fig, ax = plt.subplots(nRow, nCol, figsize=(fSize * nCol, fSize * nRow))

        # ax[0][0].set_title('total loss')
        # ax[0][0].plot(self.val_pred_real_mean, color='red', label='total loss')
        # ax[0][0].fill_between(list(range(self.args.length_epoch)), self.val_pred_real_mean-self.val_pred_real_std, self.val_pred_real_mean+self.val_pred_real_std, color='blue', alpha=0.2)
        # ax[0][0].set_ylim([-10000, 10000])
        # ax[0][0].legend()
        ax[0][0].set_title('smoothed predictions')
        window_size = 20
        
        ax[0][0].plot(np.convolve(self.val_pred_real_mean[:self.current_step+1], np.ones(window_size) / window_size, mode='valid'), color='red', label='pred_real')
        ax[0][0].plot(np.convolve(self.val_pred_fake_mean[:self.current_step+1], np.ones(window_size) / window_size, mode='valid'), color='green', label='pred_fake')
        ax[0][0].plot(np.convolve(self.val_lang_mean[:self.current_step+1], np.ones(window_size) / window_size, mode='valid'), color='teal', label='pred_lang')
        # ax[0][0].plot(np.convolve(self.val_pred_fake_min_time[:self.current_step+1], np.ones(window_size) / window_size, mode='valid'), color='purple', label='fake_first_up')
        # ax[0][0].plot(np.convolve(self.val_pred_real_min_time[:self.current_step+1], np.ones(window_size) / window_size, mode='valid'), color='orange', label='real_first_up')
        
        # ax[0][0].set_ylim([-10000, 10000])
        ax[0][0].legend()


        ax[0][1].set_title('fid_score')
        ax[0][1].plot(self.fid_scores, color='yellow', label='fid_score')
        # ax[0][1].set_ylim([-10000, 10000])
        ax[0][1].legend()

        ax[0][2].set_title('loss energy')
        ax[0][2].plot(self.val_loss_energy_mean[:self.current_step+1], color='red', label='energy')
        ax[0][2].plot(np.convolve(self.val_loss_energy_mean[:self.current_step+1], np.ones(window_size) / window_size, mode='valid'), color='teal', label='energy_smoothed')
        ax[0][2].plot(np.convolve(self.val_loss_positive_mean[:self.current_step+1], np.ones(window_size) / window_size, mode='valid'), color='black', label='pos_loss')
        # ax[0][2].set_ylim([-250, 50])
        ax[0][2].legend()

        ax[0][3].set_title('loss without grad pen')
        ax[0][3].plot(self.val_loss_positive_mean[:self.current_step+1], color='blue', label='loss')
        # ax[0][3].set_ylim([-10000, 10000])
        ax[0][3].legend()

        plt_cmap = 'gray'
        if self.args.channel_data == 3:
            real_plot = np.clip(np.moveaxis(real_plot, 1, -1), -1, 1)
            fake_plot = np.clip(np.moveaxis(fake_plot, 1, -1), -1, 1)
            real_plot = (real_plot+1)/2
            fake_plot = (fake_plot+1)/2
            plt_cmap = 'viridis'


        # ax[1][0].set_title('lr*loss_avg, max t')
        # learning_rates = self.lr_schdeule[trainer.args.model_update_num-1:(self.current_step+1)*trainer.args.model_update_num:trainer.args.model_update_num]
        # ax[1][0].plot(learning_rates*np.convolve(self.val_loss_energy_mean[:self.current_step+1], np.ones(window_size) / window_size, mode='same'), color='blue', label='energy')
        ax[1][0].set_title('gradient penalty')
        ax[1][0].plot(self.val_gradient_penalty[:self.current_step+1], color='purple', label='penalty')
        ax[1][0].legend()

        ax[2][0].set_title('pred real - pred fake, max t')
        ax[2][0].plot(self.lr_schdeule, color='red', label='lr schedule')
        ax[2][0].legend()

        
        for c in range(1, nCol):
            ax[1][c].set_title('real t ='+str(time_plot[c]))
            ax[1][c].imshow(real_plot[c], cmap=plt_cmap)#, vmin=0.0, vmax=1.0)
        for c in range(1, nCol):    
            ax[2][c].set_title('fake first time t ='+str(time_plot[c]))
            ax[2][c].imshow(fake_plot[c], cmap=plt_cmap)#, vmin=0.0, vmax=1.0)

        ax[3][0].set_title('fake_pred grad mean')

        # Fill in NaN's...
        mask = np.isnan(self.val_pred_fake_min_time)
        self.val_pred_fake_min_time[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), self.val_pred_fake_min_time[~mask])
        self.val_pred_real_min_time[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), self.val_pred_real_min_time[~mask])
        ax[3][0].plot(np.convolve(self.val_pred_fake_min_time[:self.current_step+1], np.ones(window_size) / window_size, mode='valid'), color='purple', label='fake_max_t')
        ax[3][0].plot(np.convolve(self.val_pred_real_min_time[:self.current_step+1], np.ones(window_size) / window_size, mode='valid'), color='orange', label='real_max_t')
        
        # ax[3][0].plot(np.convolve(self.val_loss_gradient_mean[:self.current_step+1], np.ones(window_size) / window_size, mode='valid'), color='green', label='real - fake')
        ax[3][0].legend()

        ax[3][1].set_title('pred real - pred fake, max t')
        ax[3][1].plot(np.convolve(self.val_pred_real_min_time[:self.current_step+1] - self.val_pred_fake_min_time[:self.current_step+1], np.ones(window_size) / window_size, mode='valid'), color='teal', label='real - lang')
        ax[3][1].legend()


        # ax[3][2].set_title('time distribution')
        # kwargs = dict(alpha=0.5, bins=self.args.length_time, density=True, stacked=True)
        # ax[3][2].hist(trainer.dataloader_fake.dataset.time, **kwargs, color='b', label='times')
        # ax[3][2].hist(time_plot, **kwargs, color='g', label='times')
        # ax[3][2].legend()
        plt.savefig('../inference_png/time_histogram.png')
        ax[3][2].set_title('pred lang inters - last '+str(self.args.print_freq))
        ax[3][2].plot(self.val_pred_real_mean[self.current_step-self.args.print_freq+1:self.current_step+1], color='red', label='pred_real')
        ax[3][2].plot(self.val_pred_fake_mean[self.current_step-self.args.print_freq+1:self.current_step+1], color='green', label='pred_fake')
        count = 0
        npme = np.array(self.val_pred_lang_iters)

        for lang in np.array(self.val_pred_lang_iters).T :


            ax[3][2].plot(lang.flatten()[-self.args.print_freq:], label='lang_'+str(count))
            count += 1
        ax[3][2].legend()


        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
        

        device = torch.device('cpu')

        time = torch.range(0, trainer.args.length_time-1).type(torch.LongTensor)
        weights = trainer.get_time_weights(time, trainer.args).to(device)
        ax[3][3].set_title('loss weights mean='+str(weights.mean().item()))

        ax[3][3].plot(weights, color='blue')

        # w = trainer.weight_sampler.weights()
        # p = w / np.sum(w)
        # p = torch.from_numpy(p).float().to(device)
        # weights = 1 / (len(p) * p)
        # ax[3][3].plot(weights, color='red')
        
        weights1 = trainer.get_time_weights(time[0:trainer.args.length_time//2], trainer.args).to(device)# trainer.weight_sampler.sample(, device)

        weights2 = trainer.get_time_weights(time[trainer.args.length_time//2:trainer.args.length_time-1], trainer.args).to(device) #trainer.weight_sampler.sample(time[trainer.args.length_time//2:trainer.args.length_time-1], device)
        weights_all = torch.cat((weights1, weights2), 0)
        # ax[3][3].plot(weights_all, color='green')

        # ax[3][3].legend()
        # ax[3][3].set_title('weights mean over time')
        if trainer.args.use_time_coefficient:
            ax[3][3].plot(trainer.time_coefficient[time].to(device), color='red')
            
        plt.tight_layout()
        fig.savefig(self.file_figure, bbox_inches='tight', dpi=100)

        fig.savefig(self.file_graph, bbox_inches='tight', dpi=100)
        plt.close(fig)


        
        
            