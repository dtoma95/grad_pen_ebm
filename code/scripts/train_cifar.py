#!/usr/bin/env python3

import os
import sys


dir_work        = '../../'
program         = '{}/{}'.format(dir_work, 'code/main_parallel.py')
# program         = '{}/{}'.format(dir_work, 'code/main.py')

device_cuda     = [0]
size_batch      = [256]
length_epoch    = [300000]
num_data_fake   = [256*100]

length_time_and_fake_lr     =  [(18, 1.7)] #(20, 55) [(6, 180)] #(800, 1.375)] #[(500, 2.1), (500, 3.5), (500, 3), (500, 4.9)]  (1000, 1.1) (10, 110), (5, 220)
length_langevin = [1]
# lr_energy       = [0.00005]
# lr_fake         = [0.828]
lr_langevin     = [0.005]#[0.0001]
weight_gradient_penalty = [10] #second has no spec regularization
fake_update_num  = [1]
model_update_num = [1]
# scheduler_args = [
                    #  ("dropoffexp", 0.0002, 5000, 1e-06, 0, 0),
                    
                    # ("dropoff", 0.00005, 12000, 0, 0, 0),
                    # ("exp", 0.0002, 0, 1e-08, 0, 0)
                    # ]
 # 2023.10.17 YOU CHANGED BETA to 0.5 DONT FORGET
scheduler_args = [
                    # ("warmupconst", 0.0003, 5000, 0, 0, 0),
                    ("cosineannealingwarmrestarts", 0.0002, 0, 5000, 0, 0),
                    ]

update_order = ['sigmoid6'] 
wrmup_every  = [99999999]
fake_train_jumps = [1]
coeff_style = ['none']
comment = '\"clip fake each iter\"'

print(length_time_and_fake_lr)
for p0 in length_epoch:
    for p1, p5 in length_time_and_fake_lr:
        for p2 in length_langevin:
            for stype, lr, step, arg1, arg2, gamma in scheduler_args:
                for p6 in lr_langevin:
                    for p7 in weight_gradient_penalty:
                        for p8 in size_batch:
                            for p9 in device_cuda:
                                for p11 in fake_update_num:
                                    for p12 in wrmup_every:
                                        for p13 in num_data_fake:
                                            for p14 in update_order:
                                                for p15 in fake_train_jumps:
                                                    for p16 in coeff_style: 
                                                        for p17 in model_update_num: 
                                                            o0 = '--length_epoch={}'.format(p0)
                                                            o1 = '--length_time={}'.format(p1)
                                                            o2 = '--length_langevin={}'.format(p2)
                                                            o4 = '--lr_energy={}'.format(lr)
                                                            o5 = '--lr_fake={}'.format(p5)
                                                            o6 = '--lr_langevin={}'.format(p6)
                                                            o7 = '--weight_gradient_penalty={}'.format(p7)
                                                            o8 = '--size_batch={}'.format(p8)
                                                            o9 = '--device_cuda={}'.format(p9)
                                                            o11 = '--fake_update_num={}'.format(p11)

                                                            o12 = '--fake_data_warmup_every={}'.format(p12)
                                                            o13 = "--num_data_fake={}".format(p13) 
            #real_and_fake                                  
                                                            o14 = '--scheduler_gamma={}'.format(gamma)

                                                            o15 = '--scheduler_type={}'.format(stype)
                                                            o16 = '--scheduler_arg_step={}'.format(step)
                                                            o17 = '--scheduler_arg_1={}'.format(arg1)
                                                            o18 = '--scheduler_arg_2={}'.format(arg2)

                                                            o19 = "--weighting_type={}".format(p14)
                                                            o20 = "--fake_train_jumps={}".format(p15)
                                                            o21 = "--coefficient_type={}".format(p16)
                                                            o22 = "--model_update_num={}".format(p17)

                                                            command = '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}  --init_from=gaussian --penalty_type=interpolate --ema_start=5000  --name_model=unet_improved2_attpool2_small --name_dataset=cifar10 --buffer_type=normal --refresh_rate=1 --num_data_real=-1'.format('python', program, o0, o1, o2, o4, o5, o6, o7, o8, o9, o11, o12, o13, o14, o15, o16, o17, o18, o19, o20, o21, o22)
                                                            
                                                            if comment != "":
                                                                command = '{} --comment {}'.format(command, comment)
                                                        
                                                            print(command)
                                                            os.system(command) 
