from __future__ import print_function
import os
import random
import numpy as np
import argparse
from config import cfg, get_data_dir, get_output_dir, AverageMeter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from SDAE import sdae_mnist
from custom_data import DCCPT_data

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

# Parse all the input argument
parser = argparse.ArgumentParser(description='PyTorch SDAE Training')
parser.add_argument('--batchsize', type=int, default=256, help='batch size used for pretraining')
parser.add_argument('--niter', type=int, default=50000, help='number of iterations used for pretraining')
parser.add_argument('--step', type=int, default=20000,
                    help='stepsize in terms of number of iterations for pretraining. lr is decreased by 10 after every stepsize.')
parser.add_argument('--lr', default=10, type=float, help='initial learning rate for pretraining')
parser.add_argument('--manualSeed', default=cfg.RNG_SEED, type=int, help='manual seed')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--level', default=0, type=int, help='index of the module to resume from')
parser.add_argument('--data', dest='db', type=str, default='mnist', help='name of the dataset')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--dim', type=int, help='dimension of embedding space', default=16)
parser.add_argument('--deviceID', type=int, help='deviceID', default=0)
parser.add_argument('--h5', dest='h5', help='to store as h5py file', default=False, type=bool)
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--id', type=int, help='identifying number for storing tensorboard logs')
parser.add_argument('--fake', action='store_true', help='do not train')

def main():
    global args

    args = parser.parse_args()
    datadir = get_data_dir(args.db)
    outputdir = get_output_dir(args.db)

    if args.tensorboard:
        # One should create folder for storing logs
        loggin_dir = os.path.join(outputdir, 'runs', 'pretraining')
        if not os.path.exists(loggin_dir):
            os.makedirs(loggin_dir)
        configure(os.path.join(loggin_dir, '%s' % (args.id)))

    use_cuda = torch.cuda.is_available()

    # Set the seed for reproducing the results
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    trainset = DCCPT_data(root=datadir, train=True, h5=args.h5)
    testset = DCCPT_data(root=datadir, train=False, h5=args.h5)

    nepoch = int(np.ceil(np.array(args.niter * args.batchsize, dtype=float) / len(trainset)))
    step = int(np.ceil(np.array(args.step * args.batchsize, dtype=float) / len(trainset)))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, **kwargs)

    pretrain(outputdir, {'nlayers':4, 'dropout':0.2, 'reluslope':0.0,
                       'nepoch':nepoch, 'lrate':[args.lr], 'wdecay':[0.0], 'step':step}, use_cuda, trainloader, testloader)

def pretrain(outputdir, params, use_cuda, trainloader, testloader):
    numlayers = params['nlayers']
    lr = params['lrate'][0]
    maxepoch = params['nepoch']
    stepsize = params['step']
    startlayer = 0

    # For simplicity, I have created placeholder for each datasets and model
    if args.db == 'mnist':
        net = sdae_mnist(dropout=params['dropout'], slope=params['reluslope'], dim=args.dim)
    else:
        print("db not supported: '{}'".format(args.db))
        raise

    if args.fake:
        for param in net.parameters():
            param.requires_grad = False

    # For the final FT stage of SDAE pretraining, the total epoch is twice that of previous stages.
    maxepoch = [maxepoch]*numlayers + [maxepoch*2]
    stepsize = [stepsize]*(numlayers+1)

    if args.resume:
        filename = outputdir+'/checkpoint_%d.pth.tar' % args.level
        if os.path.isfile(filename):
            print("==> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            net.load_state_dict(checkpoint['state_dict'])
            startlayer = args.level+1
        else:
            print("==> no checkpoint found at '{}'".format(filename))
            raise

    if use_cuda:
        net.cuda()

    for index in range(startlayer, numlayers+1):
        # Freezing previous layer weights
        if index < numlayers:
            for par in net.base[index].parameters():
                par.requires_grad = False
            if args.db == 'cmnist' or args.db == 'ccoil100' or args.db == 'cytf' or args.db == 'cyale':
                for par in net.bbase[index].parameters():
                    par.requires_grad = False
                for m in net.bbase[index].modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.training = False
        else:
            for par in net.base[numlayers-1].parameters():
                par.requires_grad = True
            if args.db == 'cmnist' or args.db == 'ccoil100' or args.db == 'cytf' or args.db == 'cyale':
                for par in net.bbase[numlayers-1].parameters():
                    par.requires_grad = True
                for m in net.bbase[numlayers-1].modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.training = True

        # setting up optimizer - the bias params should have twice the learning rate w.r.t. weights params
        bias_params = filter(lambda x: ('bias' in x[0]) and (x[1].requires_grad), net.named_parameters())
        bias_params = list(map(lambda x: x[1], bias_params))
        nonbias_params = filter(lambda x: ('bias' not in x[0]) and (x[1].requires_grad), net.named_parameters())
        nonbias_params = list(map(lambda x: x[1], nonbias_params))

        optimizer = optim.SGD([{'params': bias_params, 'lr': 2*lr}, {'params': nonbias_params}],
                              lr=lr, momentum=0.9, weight_decay=params['wdecay'][0], nesterov=True)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize[index], gamma=0.1)

        print('\nIndex: %d \t Maxepoch: %d'%(index, maxepoch[index]))

        save_checkpoint({'epoch': 0, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
                            index, args.fake, filename=outputdir)
        for epoch in range(maxepoch[index]):
            scheduler.step()
            train(trainloader, net, index, optimizer, epoch, use_cuda)
            test(testloader, net, index, epoch, use_cuda)
            # Save checkpoint
            save_checkpoint({'epoch': epoch+1, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
                            index, args.fake, filename=outputdir)


# Training
def train(trainloader, net, index, optimizer, epoch, use_cuda):
    losses = AverageMeter()

    print('\nIndex: %d \t Epoch: %d' %(index,epoch))

    net.train()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs = inputs.cuda()
        optimizer.zero_grad()
        inputs_Var = Variable(inputs)
        outputs = net(inputs_Var, index)
        # import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

        # record loss
        losses.update(outputs.data[0], inputs.size(0))

        outputs.backward()
        optimizer.step()

    print('train_loss_{}'.format(index), losses.avg, epoch)
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss_{}'.format(index), losses.avg, epoch)


# Testing
def test(testloader, net, index, epoch, use_cuda):
    losses = AverageMeter()

    net.eval()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs = inputs.cuda()
        inputs_Var = Variable(inputs, volatile=True)
        outputs = net(inputs_Var, index)

        # measure accuracy and record loss
        losses.update(outputs.data[0], inputs.size(0))

    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss_{}'.format(index), losses.avg, epoch)


# Saving checkpoint
def save_checkpoint(state, index, is_fake, filename):
    out_filename = filename+'/checkpoint_%d.pth.tar' % index
    if is_fake:
        out_filename += '.fake'
    torch.save(state, out_filename)

if __name__ == '__main__':
    main()
