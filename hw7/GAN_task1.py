import os
import torch
import argparse
import os
import numpy as np
import random
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import _netG, _netD
import sys
sys.path.append('dataset/task_1/')
from evaluator import evaluation_model
from dataset import ICLEVRLoader, get_test_conditions, get_test_conditions_new
from util import save_image
import copy
from torch.utils.tensorboard import SummaryWriter


batchSize = 16
nz=110
num_classes=24
epochs = 200



if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    # load training data
    dataset_train = ICLEVRLoader()
    loader_train = DataLoader(dataset_train,batch_size=batchSize,shuffle=True,num_workers=1,drop_last=True)

    netG = _netG(ngpu=1, nz=110).to(device) #nz: size of the latent z vector
    netD = _netD(ngpu=1, num_classes=24).to(device)
    evaluation_model=evaluation_model()
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # loss functions
    dis_criterion = nn.BCELoss().to(device)
    aux_criterion = nn.BCELoss().to(device)

    real_label = 1
    fake_label = 0
    best_score = 0
    iter = 0
    for epoch in range(1,epochs+1):
        netG.train()
        netD.train()
        for i, (images,read_conditions) in enumerate(loader_train):
            iter += 1
            conditions = Variable(read_conditions)


            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            batch_size = len(images)

            predict_realfake, predict_classes = netD(images.to(device)) # return realfake, classes
            with torch.no_grad():
                dis_label = torch.FloatTensor(batchSize).to(device)
                dis_label.data.fill_(real_label)
            dis_errD_real = dis_criterion(predict_realfake, dis_label)
            aux_label = copy.deepcopy(conditions)
            aux_label = aux_label.to(device)
            aux_errD_real = aux_criterion(predict_classes, aux_label)
            errD_real = dis_errD_real + aux_errD_real
            errD_real.backward(retain_graph=True)

            # train with fake
            with torch.no_grad():
                noise = torch.FloatTensor(batch_size, nz, 1, 1)
                noise = Variable(noise)
                noise.data.resize_(batch_size, nz, 1, 1).normal_(0, 1)
                label = np.random.randint(0, num_classes, batch_size)
                noise_ = np.random.normal(0, 1, (batch_size, nz))
                class_onehot = copy.deepcopy(conditions)
                noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
                noise_ = (torch.from_numpy(noise_))
                noise.data.copy_(noise_.view(batch_size, nz, 1, 1))

                dis_label = torch.FloatTensor(batchSize).to(device)
                dis_label.data.resize_(batch_size).fill_(fake_label)
            fake = netG(noise.to(device))
            predict_realfake, predict_classes = netD(fake)
            dis_errD_fake = dis_criterion(predict_realfake, dis_label)
            aux_errD_fake = aux_criterion(predict_classes, aux_label)
            errD_fake = dis_errD_fake + aux_errD_fake
            errD_fake.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            with torch.no_grad():
                dis_label = torch.FloatTensor(batchSize).to(device)
                dis_label.data.resize_(batch_size).fill_(real_label)  # fake labels are real for generator cost
            predict_realfake, predict_classes = netD(fake)
            dis_errG = dis_criterion(predict_realfake, dis_label)
            aux_errG = aux_criterion(predict_classes, aux_label)
            errG = dis_errG + aux_errG
            errG.backward(retain_graph=True)
            optimizerG.step()

            # evaluate
            netG.eval()
            netD.eval()
            with torch.no_grad():
                test_conditions = get_test_conditions()
                noise = torch.FloatTensor(batchSize, nz-num_classes, 1, 1).to(device)
                c_resize = copy.deepcopy(test_conditions)
                c_resize.resize_( (batchSize, num_classes, 1, 1) )
                z = torch.cat((noise.to(device),c_resize.to(device)),dim=1)

            gen_imgs=netG(z)
            score = evaluation_model.eval(gen_imgs,test_conditions)

            if score > best_score :
                best_G = copy.deepcopy(netG.state_dict())
                best_D = copy.deepcopy(netD.state_dict())
                torch.save(best_G,'models/iter_'+ str(iter)+'_'+str(score)+'_'+'_G.pt')
                torch.save(best_D,'models/iter_'+ str(iter)+'_'+str(score)+'_'+'_D.pt')
                print('save: models/iter_'+ str(iter)+'_'+str(score)+'_'+'_G.pt')
                best_score = score

            writer.add_scalars("cGAN_task1_Train", {    "errD_fake":errD_fake.item(),\
                                                        "errD_real": errD_real.item(),\
                                                        "errG": errG.item(),\
                                                        "score": score  }, iter)
            writer.flush()
            if i % 10 == 0: print(i, errD_fake.item(),errD_real.item(),errG.data.item(),'score = ',score)
        




        # savefig
        save_image(gen_imgs, os.path.join('results', f'epoch{epoch}.png'), nrow=8, normalize=True)
