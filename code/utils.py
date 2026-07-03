""" Exponential Moving Average (EMA) of model updates

Hacked together by / Copyright 2020 Ross Wightman
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class SpectralNormalizer: #()
    def __init__(self, model, num_power_iter=4):
        
        self.all_conv_layers = []
        for n, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                self.all_conv_layers.append(layer)
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4


    def spectral_norm_parallel(self):
        """ This method computes spectral normalization for all conv layers in parallel. This method should be called
         after calling the forward method of all the conv layers in each iteration. """

        weights = {}   # a dictionary indexed by the shape of weights
        for l in self.all_conv_layers:
            weight = l.weight #l.weight_normalized
            weight_mat = weight.view(weight.size(0), -1)
            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []

            weights[weight_mat.shape].append(weight_mat)

        loss = 0
        for i in weights:
            weights[i] = torch.stack(weights[i], dim=0)
            with torch.no_grad():
                num_iter = self.num_power_iter
                if i not in self.sr_u:
                    num_w, row, col = weights[i].shape
                    self.sr_u[i] = F.normalize(torch.ones(num_w, row).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    self.sr_v[i] = F.normalize(torch.ones(num_w, col).normal_(0, 1).cuda(), dim=1, eps=1e-3)
                    # increase the number of iterations for the first time
                    num_iter = 10 * self.num_power_iter

                for j in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    self.sr_v[i] = F.normalize(torch.matmul(self.sr_u[i].unsqueeze(1), weights[i]).squeeze(1),
                                               dim=1, eps=1e-3)  # bx1xr * bxrxc --> bx1xc --> bxc
                    self.sr_u[i] = F.normalize(torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)).squeeze(2),
                                               dim=1, eps=1e-3)  # bxrxc * bxcx1 --> bxrx1  --> bxr

            sigma = torch.matmul(self.sr_u[i].unsqueeze(1), torch.matmul(weights[i], self.sr_v[i].unsqueeze(2)))
            loss += torch.sum(sigma)
        return loss

# def category_mean(dload_train, args):
#     import time
#     start = time.time()
#     if args.dataset == 'svhn':
#         size = [3, 32, 32]
#     else:
#         size = [3, 32, 32]
#     centers = t.zeros([args.n_classes, int(np.prod(size))])
#     covs = t.zeros([args.n_classes, int(np.prod(size)), int(np.prod(size))])

#     im_test, targ_test = [], []
#     for im, targ in dload_train:
#         im_test.append(im)
#         targ_test.append(targ)
#     im_test, targ_test = t.cat(im_test), t.cat(targ_test)

#     # conditionals = []
#     for i in range(args.n_classes):
#         imc = im_test[targ_test == i]
#         imc = imc.view(len(imc), -1)
#         mean = imc.mean(dim=0)
#         sub = imc - mean.unsqueeze(dim=0)
#         cov = sub.t() @ sub / len(imc)
#         centers[i] = mean
#         covs[i] = cov
#     print(time.time() - start)
#     t.save(centers, '%s_mean.pt' % args.dataset)
#     t.save(covs, '%s_cov.pt' % args.dataset)

# def init_random(args, bs):
#     global conditionals
#     n_ch = 3
#     size = [3, 32, 32]
#     im_sz = 32
#     new = t.zeros(bs, n_ch, im_sz, im_sz)
#     for i in range(bs):
#         index = np.random.randint(len(conditionals))
#         dist = conditionals[index]
#         new[i] = dist.sample().view(size)
#     return t.clamp(new, -1, 1).cpu()

# def init_from_centers(arg):
#     global conditionals
#     from torch.distributions.multivariate_normal import MultivariateNormal
#     bs = arg.buffer_size
#     if arg.dataset == 'tinyimagenet':
#         size = [3, 64, 64]
#     else:
#         size = [3, 32, 32]
#     v = t.__version__.split('.')[1]
#     centers = t.load('v%s/%s_mean.pt' % (v, arg.dataset))
#     covs = t.load('v%s/%s_cov.pt' % (v, arg.dataset))

#     buffer = []
#     cov = covs.to(arg.device)
#     # cov_m = cov + 1e-4 * t.eye(int(np.prod(size))).to(arg.device)
#     with torch.no_grad():
#         for i in range(arg.n_classes):
#             mean = centers[i].to(arg.device)
#             cov_m = cov[i] + 1e-4 * t.eye(int(np.prod(size))).to(arg.device)
#             dist = MultivariateNormal(mean, covariance_matrix=cov_m)
#             buffer.append(dist.sample((bs // arg.n_classes,)).view([bs // arg.n_classes] + size).cpu())
#             conditionals.append(dist)
#     return t.clamp(t.cat(buffer), -1, 1)