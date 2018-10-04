from __future__ import print_function
import os
import random
import math
import numpy as np
import scipy.io as sio
import argparse
from config import cfg, get_data_dir, get_output_dir, AverageMeter

import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from extractSDAE import extract_sdae_mnist
from custom_data import DCCPT_data, DCCFT_data, DCCSampler
from DCCLoss import DCCWeightedELoss, DCCLoss
from DCCComputation import makeDCCinp, computeHyperParams, computeObj
from Disentanglement_Nets_Mnist import DecoderNet

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

# Parse all the input argument
parser = argparse.ArgumentParser(description='PyTorch DCC Finetuning')
parser.add_argument('--data', dest='db', type=str, default='mnist',
                    help='Name of the dataset. The name should match with the output folder name.')
parser.add_argument('--batchsize', type=int, default=cfg.PAIRS_PER_BATCH, help='batch size used for Finetuning')
parser.add_argument('--nepoch', type=int, default=500, help='maximum number of iterations used for Finetuning')
# By default M = 20 is used. For convolutional SDAE M=10 was used.
# Similarly, for different NW architecture different value for M may be required.
parser.add_argument('--M', type=int, default=20, help='inner number of epochs at which to change lambda')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--manualSeed', default=cfg.RNG_SEED, type=int, help='manual seed')
parser.add_argument('--net', dest='torchmodel', help='path to the pretrained weights file', default=None, type=str)
parser.add_argument('--net-pretraining', dest='torchmodel_pretraining', help='path to the pretrained weights file', default=None, type=str)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--level', default=0, type=int, help='epoch to resume from')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--deviceID', type=int, help='deviceID', default=0)
parser.add_argument('--h5', dest='h5', action='store_true', help='to store as h5py file')
parser.add_argument('--dim', type=int, help='dimension of embedding space', default=16)
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--id', type=int, help='identifying number for storing tensorboard logs')
parser.add_argument('--step', type=int, help='Step number for disentanglement')
parser.add_argument('--nsamples', type=int, default=None, help='Number of samples')


def main():
    global args, oldassignment

    args = parser.parse_args()
    datadir = get_data_dir(args.db)
    outputdir = get_output_dir(args.db)

    if args.tensorboard:
        # One should create folder for storing logs
        loggin_dir = os.path.join(outputdir, 'runs', 'DCC')
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

    reluslope = 0.0
    startepoch = 0
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}

    # setting up dataset specific objects
    trainset = DCCPT_data(root=datadir, train=True, h5=args.h5)
    testset = DCCPT_data(root=datadir, train=False, h5=args.h5)

    numeval = len(trainset) + len(testset)


    # For simplicity, I have created placeholder for each datasets and model
    if args.db == 'mnist':
        net_s = extract_sdae_mnist(slope=reluslope, dim=args.dim)
        net_z = extract_sdae_mnist(slope=reluslope, dim=args.dim)
    else:
        print("db not supported: '{}'".format(args.db))
        raise

    totalset = torch.utils.data.ConcatDataset([trainset, testset])

    # extracting training data from the pretrained.mat file
    data, labels, pairs, Z, sampweight = makeDCCinp(args)

    # computing and initializing the hyperparams
    _sigma1, _sigma2, _lambda, _delta, _delta1, _delta2, lmdb, lmdb_data = computeHyperParams(pairs, Z, args.step)
    oldassignment = np.zeros(len(pairs))
    stopping_threshold = int(math.ceil(cfg.STOPPING_CRITERION * float(len(pairs))))

    # Create dataset and random batch sampler for Finetuning stage
    trainset = DCCFT_data(pairs, data, sampweight)
    batch_sampler = DCCSampler(trainset, shuffle=True, batch_size=args.batchsize)

    # setting up data loader for training and testing phase
    trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler, **kwargs)
    testloader = torch.utils.data.DataLoader(totalset, batch_size=args.batchsize, shuffle=False, **kwargs)


    if args.step == 1:

        pretraining_filename = os.path.join(outputdir, args.torchmodel_pretraining)
        if os.path.isfile(pretraining_filename):
            print("==> loading params from pretraining checkpoint '{}'".format(pretraining_filename))
            pretraining_checkpoint = torch.load(pretraining_filename)
        else:
            print("==> no pretraining checkpoint found at '{}'".format(pretraining_filename))
            raise


        # setting up optimizer - the bias params should have twice the learning rate w.r.t. weights params
        bias_params = filter(lambda x: ('bias' in x[0]), net_s.named_parameters())
        bias_params = list(map(lambda x: x[1], bias_params))
        nonbias_params = filter(lambda x: ('bias' not in x[0]), net_s.named_parameters())
        nonbias_params = list(map(lambda x: x[1], nonbias_params))

        # copying model params from Pretrained (SDAE) weights file
        net_s.load_state_dict(pretraining_checkpoint['state_dict'])

        criterion_sc = DCCLoss(Z.shape[0], Z.shape[1], Z, size_average=True)
        optimizer_sc = optim.Adam([{'params': bias_params, 'lr': 2*args.lr},
                            {'params': nonbias_params},
                            {'params': criterion_sc.parameters(), 'lr': args.lr},
                            ], lr=args.lr, betas=(0.99, 0.999))
        criterion_rec = DCCWeightedELoss(size_average=True) # OLD


        if use_cuda:
            net_s.cuda()
            criterion_sc = criterion_sc.cuda()
            criterion_rec = criterion_rec.cuda()

        # this is needed for WARM START
        if args.resume:
            filename = outputdir+'/FTcheckpoint_%d.pth.tar' % args.level
            if os.path.isfile(filename):
                print("==> loading checkpoint '{}'".format(filename))
                checkpoint = torch.load(filename)
                net_s.load_state_dict(checkpoint['state_dict_s'])
                criterion_sc.load_state_dict(checkpoint['criterion_state_dict_sc'])
                startepoch = checkpoint['epoch']
                optimizer_sc.load_state_dict(checkpoint['optimizer_sc'])
                _sigma1 = checkpoint['sigma1']
                _sigma2 = checkpoint['sigma2']
                _lambda = checkpoint['lambda']
                _delta = checkpoint['delta']
                _delta1 = checkpoint['delta1']
                _delta2 = checkpoint['delta2']
            else:
                print("==> no checkpoint found at '{}'".format(filename))
                raise

        # This is the actual Algorithm
        flag = 0
        for epoch in range(startepoch, args.nepoch):
            print('sigma1', _sigma1, epoch)
            print('sigma2', _sigma2, epoch)
            print('lambda', _lambda, epoch)
            if args.tensorboard:
                log_value('sigma1', _sigma1, epoch)
                log_value('sigma2', _sigma2, epoch)
                log_value('lambda', _lambda, epoch)

            train_step_1(trainloader, net_s, optimizer_sc, criterion_rec, criterion_sc, epoch, use_cuda, _sigma1, _sigma2, _lambda)
            Z, U, change_in_assign, assignment = test(testloader, net_s, criterion_sc, epoch, use_cuda, _delta, pairs, numeval, flag)

            if flag:
                # As long as the change in label assignment < threshold, DCC continues to run.
                # Note: This condition is always met in the very first epoch after the flag is set.
                # This false criterion is overwritten by checking for the condition twice.
                if change_in_assign > stopping_threshold:
                    flag += 1

            if((epoch+1) % args.M == 0):
                _sigma1 = max(_delta1, _sigma1 / 2)
                _sigma2 = max(_delta2, _sigma2 / 2)
                if _sigma2 == _delta2 and flag == 0:
                    # Start checking for stopping criterion
                    flag = 1

            # Save checkpoint
            index = (epoch // args.M) * args.M
            save_checkpoint({'epoch': epoch+1,
                             'state_dict_s': net_s.state_dict(),
                             'criterion_state_dict_sc': criterion_sc.state_dict(),
                             'optimizer_sc': optimizer_sc.state_dict(),
                             'sigma1': _sigma1,
                             'sigma2': _sigma2,
                             'lambda': _lambda,
                             'delta': _delta,
                             'delta1': _delta1,
                             'delta2': _delta2,
                             }, index, filename=outputdir)

            sio.savemat(os.path.join(outputdir, 'features_s'), {'Z': Z, 'U': U, 'gtlabels': labels, 'w': pairs, 'cluster':assignment})

    elif args.step == 2:
        filename = os.path.join(outputdir, args.torchmodel)
        if os.path.isfile(filename):
            print("==> loading params from checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
        else:
            print("==> no checkpoint found at '{}'".format(filename))
            raise

        # copying model params of s encoder from step 1
        net_s.load_state_dict(checkpoint['state_dict_s'])

        # freezing net_s
        for param in net_s.parameters():
            param.requires_grad = False

        net_d = DecoderNet(1)
        criterion_d = nn.MSELoss()

        # setting up optimizer - the bias params should have twice the learning rate w.r.t. weights params
        bias_params = filter(lambda x: ('bias' in x[0]), net_z.named_parameters())
        bias_params = list(map(lambda x: x[1], bias_params))
        nonbias_params = filter(lambda x: ('bias' not in x[0]), net_z.named_parameters())
        nonbias_params = list(map(lambda x: x[1], nonbias_params))

        criterion_zc = DCCLoss(Z.shape[0], Z.shape[1], Z, size_average=True)
        optimizer_zc = optim.Adam([{'params': bias_params, 'lr': 2*args.lr},
                            {'params': nonbias_params},
                            {'params': criterion_zc.parameters(), 'lr': args.lr},
                            ], lr=args.lr, betas=(0.99, 0.999))
        optimizer_d = torch.optim.Adam(net_d.parameters(), lr=0.001)
        criterion_rec = DCCWeightedELoss(size_average=True)
        if use_cuda:
            net_d.cuda()
            net_s.cuda()
            net_z.cuda()
            criterion_zc = criterion_zc.cuda()
            criterion_d = criterion_d.cuda()
            criterion_rec = criterion_rec.cuda()

        flag = 0
        for epoch in range(startepoch, args.nepoch):
            print('sigma1', _sigma1, epoch)
            print('sigma2', _sigma2, epoch)
            print('lambda', _lambda, epoch)
            if args.tensorboard:
                log_value('sigma1', _sigma1, epoch)
                log_value('sigma2', _sigma2, epoch)
                log_value('lambda', _lambda, epoch)

            train_step_2(trainloader, net_s, net_z, net_d, optimizer_zc, optimizer_d, criterion_rec, criterion_zc, criterion_d, epoch, use_cuda, _sigma1, _sigma2, _lambda)
            Z, U, change_in_assign, assignment = test(testloader, net_z, criterion_zc, epoch, use_cuda, _delta, pairs, numeval, flag)


            if flag:
                # As long as the change in label assignment < threshold, DCC continues to run.
                # Note: This condition is always met in the very first epoch after the flag is set.
                # This false criterion is overwritten by checking for the condition twice.
                if change_in_assign > stopping_threshold:
                    flag += 1

            if((epoch+1) % args.M == 0):
                _sigma1 = max(_delta1, _sigma1 / 2)
                _sigma2 = max(_delta2, _sigma2 / 2)
                if _sigma2 == _delta2 and flag == 0:
                    # Start checking for stopping criterion
                    flag = 1

            # Save checkpoint
            index = (epoch // args.M) * args.M
            save_checkpoint({'epoch': epoch+1,
                             'state_dict_s': net_s.state_dict(),
                             'state_dict_z': net_z.state_dict(),
                             'state_dict_d': net_d.state_dict(),
                             'criterion_state_dict_zc': criterion_zc.state_dict(),
                             'optimizer_zc': optimizer_zc.state_dict(),
                             'sigma1': _sigma1,
                             'sigma2': _sigma2,
                             'lambda': _lambda,
                             'delta': _delta,
                             'delta1': _delta1,
                             'delta2': _delta2,
                             }, index, filename=outputdir)

        sio.savemat(os.path.join(outputdir, 'features_z'), {'Z': Z, 'U': U, 'gtlabels': labels, 'w': pairs, 'cluster':assignment})



    else:
        raise(ValueError("step not recognized!"))



# Training
def train_step_1(trainloader, net, optimizer, criterion1, criterion2, epoch, use_cuda, _sigma1, _sigma2, _lambda):
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()

    print('\n Epoch: %d' % epoch)

    net.train()

    for i, (inputs, pairweights, sampweights, pairs, index) in enumerate(trainloader):

        inputs = torch.squeeze(inputs,0)
        pairweights = torch.squeeze(pairweights)
        sampweights = torch.squeeze(sampweights)
        index = torch.squeeze(index)
        pairs = pairs.view(-1, 2)

        if use_cuda:
            inputs = inputs.cuda()
            pairweights = pairweights.cuda()
            sampweights = sampweights.cuda()
            index = index.cuda()
            pairs = pairs.cuda()

        optimizer.zero_grad()
        inputs, sampweights, pairweights = Variable(inputs), Variable(sampweights, requires_grad=False), \
                                               Variable(pairweights, requires_grad=False)

        enc, dec = net(inputs)
        loss1 = criterion1(inputs, dec, sampweights)
        loss2 = criterion2(enc, sampweights, pairweights, pairs, index, _sigma1, _sigma2, _lambda)
        loss = loss1 + loss2

        # record loss
        losses1.update(loss1.data[0], inputs.size(0))
        losses2.update(loss2.data[0], inputs.size(0))
        losses.update(loss.data[0], inputs.size(0))

        loss.backward()
        optimizer.step()

    print('dcc_loss', losses.avg, epoch)
    print('dcc_reconstruction_loss', losses1.avg, epoch)
    print('dcc_clustering_loss', losses2.avg, epoch)
    # log to TensorBoard
    if args.tensorboard:
        log_value('dcc_loss', losses.avg, epoch)
        log_value('dcc_reconstruction_loss', losses1.avg, epoch)
        log_value('dcc_clustering_loss', losses2.avg, epoch)


def train_step_2(trainloader, net_s, net_z, net_d, optimizer_zc, optimizer_d, criterion_rec, criterion_zc, criterion_d, epoch, use_cuda, _sigma1, _sigma2, _lambda):

    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses_d_rec = AverageMeter()
    losses_d = AverageMeter()

    print('\n Epoch: %d' % epoch)

    net_z.train()
    net_d.train()


    decoder_loss = 0.0
    adversarial_loss = 0.0

    for i, (inputs, pairweights, sampweights, pairs, index) in enumerate(trainloader):

        inputs = torch.squeeze(inputs,0)
        pairweights = torch.squeeze(pairweights)
        sampweights = torch.squeeze(sampweights)
        index = torch.squeeze(index)
        pairs = pairs.view(-1, 2)

        if use_cuda:
            inputs = inputs.cuda()
            pairweights = pairweights.cuda()
            sampweights = sampweights.cuda()
            index = index.cuda()
            pairs = pairs.cuda()

        inputs, sampweights, pairweights = Variable(inputs), Variable(sampweights, requires_grad=False), \
            Variable(pairweights, requires_grad=False)


        # train z encoder and decoder
        if i % 3 == 0:
            # zero the parameter gradients
            optimizer_d.zero_grad()
            optimizer_zc.zero_grad()
            # forward + backward + optimize

            outputs_s, _ = net_s(inputs)
            outputs_z, dec_z = net_z(inputs)

            loss1 = criterion_rec(inputs, dec_z, sampweights)
            loss2 = criterion_zc(outputs_z, sampweights, pairweights, pairs, index, _sigma1, _sigma2, _lambda)
            loss_zc = loss1 + loss2

            # record loss
            losses1.update(loss1.data[0], inputs.size(0))
            losses2.update(loss2.data[0], inputs.size(0))
            losses.update(loss_zc.data[0], inputs.size(0))

            decoder_input = torch.cat((outputs_s, outputs_z),1)

            outputs_d = net_d(decoder_input)
            #beta = 1.985 # change?
            beta = 1.99 # change?
            loss_d_rec = criterion_d(outputs_d, inputs)
            loss_d =  loss_d_rec - beta * loss_zc

            #record loss
            losses_d_rec.update(loss_d_rec.data[0], inputs.size(0))
            losses_d.update(loss_d.data[0], inputs.size(0))

            loss_d.backward()
            #loss_zc.backward()
            optimizer_d.step()
            optimizer_zc.step()
            decoder_loss += loss_d.data[0]

            print('dcc_reconstruction_loss', losses1.avg, epoch)
            print('dcc_clustering_loss', losses2.avg, epoch)
            print('dcc_loss', losses.avg, epoch)
            print('total_reconstruction_loss', losses_d_rec.avg, epoch)
            print('total_loss', losses_d.avg, epoch)
            # log to TensorBoard
            if args.tensorboard:
                log_value('dcc_reconstruction_loss', losses1.avg, epoch)
                log_value('dcc_clustering_loss', losses2.avg, epoch)
                log_value('dcc_loss', losses.avg, epoch)
                log_value('total_reconstruction_loss', losses_d_rec.avg, epoch)
                log_value('total_loss', losses_d.avg, epoch)

        # train adversarial clustering
        else:
            # zero the parameter gradients
            optimizer_zc.zero_grad()
            # forward + backward + optimize
            outputs_z, dec_z = net_z(inputs)

            loss1 = criterion_rec(inputs, dec_z, sampweights)
            loss2 = criterion_zc(outputs_z, sampweights, pairweights, pairs, index, _sigma1, _sigma2, _lambda)
            loss_zc = loss1 + loss2

            # record loss
            losses1.update(loss1.data[0], inputs.size(0))
            losses2.update(loss2.data[0], inputs.size(0))
            losses.update(loss_zc.data[0], inputs.size(0))

            loss_zc.backward()
            optimizer_zc.step()
            adversarial_loss += loss_zc.data[0]


        # print statistics
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] decoder loss: %.3f, adversarial loss: %.3f' %(epoch + 1, i + 1, decoder_loss / 500, adversarial_loss / 1500))
            decoder_loss = 0.0
            adversarial_loss = 0.0






# Testing

def test(testloader, net, criterion, epoch, use_cuda, _delta, pairs, numeval, flag):
    net.eval()

    original = []
    features = []
    labels = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs = inputs.cuda()
        inputs_Var = Variable(inputs, volatile=True)
        enc, dec = net(inputs_Var)
        features += list(enc.data.cpu().numpy())
        labels += list(targets)
        original += list(inputs.cpu().numpy())

    original, features, labels = np.asarray(original).astype(np.float32), np.asarray(features).astype(np.float32), \
        np.asarray(labels)

    U = criterion.U.data.cpu().numpy()

    change_in_assign = 0
    assignment = -np.ones(len(labels))
    # logs clustering measures only if sigma2 has reached the minimum (delta2)
    if flag:
        index, ari, ami, nmi, acc, n_components, assignment = computeObj(U, pairs, _delta, labels, numeval)

        # log to TensorBoard
        change_in_assign = np.abs(oldassignment - index).sum()
        print('ARI', ari, epoch)
        print('AMI', ami, epoch)
        print('NMI', nmi, epoch)
        print('ACC', acc, epoch)
        print('Numcomponents', n_components, epoch)
        print('labeldiff', change_in_assign, epoch)
        if args.tensorboard:
            log_value('ARI', ari, epoch)
            log_value('AMI', ami, epoch)
            log_value('NMI', nmi, epoch)
            log_value('ACC', acc, epoch)
            log_value('Numcomponents', n_components, epoch)
            log_value('labeldiff', change_in_assign, epoch)

        oldassignment[...] = index

    return features, U, change_in_assign, assignment

# Saving checkpoint
def save_checkpoint(state, index, filename):
    newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
    torch.save(state, newfilename)

if __name__ == '__main__':
    main()
