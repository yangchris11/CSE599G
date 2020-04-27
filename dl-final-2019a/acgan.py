import os
import sys 
import random
import argparse
import numpy as np 
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from tensorboardX import SummaryWriter

writer = SummaryWriter('all_runs/acgan_runs_test')
r_writer = SummaryWriter('all_runs/errD_real_runs_test')
w_writer = SummaryWriter('all_runs/errD_wrong_runs_test')
f_writer = SummaryWriter('all_runs/errD_fake_runs_test')


parser = argparse.ArgumentParser()
parser.add_argument('-m','--mode',required=True, help='train | test')
parser.add_argument('-dr','--dataroot', required=True, help='path to dataset')
parser.add_argument('-w','--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('-b','--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('-z','--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('-c','--nconds',type=int, default=24,help='size of the latent c vector')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=64, help='number of filters of generator')
parser.add_argument('--ndf', type=int, default=64, help='number of filters of discriminator')
parser.add_argument('--niter', type=int, default=45, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta for optimizer ')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='./pth', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--flip',action='store_true', help='flip training images')

parser.add_argument('--testing_file',type=str,default='./test_tags.txt',help='path of testing file')
parser.add_argument('--demo_file', type=str, default='./demo_tags.txt',help='pth of demo file')

opt = parser.parse_args()

############# Helper Utils #############

def resize(jpgfile):
    if jpgfile.size != 64*64*3: 
        jpgfile.resize((64, 64), Image.ANTIALIAS)
    return jpgfile

def load_imgs(dirname):
    img_list = [ os.path.join(dirname, filename) for filename in sorted(os.listdir(dirname), key=lambda img:int(img[0:-4])) ]
    imgs = [ np.array(resize(Image.open(filename)), dtype=np.float32) for filename in img_list ]
    for i in imgs:
        if i.size != 12288:
            print(i.size)
    imgs = np.stack(imgs)
    imgs = imgs / 255.
    return imgs

def load_tags(filename):
    tag_file = open(filename, 'r')
    tags = []
    for line in tag_file:
        words = line.strip().split(',')[1].split()
        tags.append([' '.join(words[:2]), ' '.join(words[2:4]), ' '.join(words[4:])])
    tags = np.array(tags)
    return tags

def demo_tags(demofilename):
    tag_file = open(demofilename, 'r')
    tags = []
    for line in tag_file:
        words = line.split()
        target = [' '.join(words[:2]), ' '.join(words[2:4]), ' '.join(words[4:])]
    
    tags = [ target for _ in range(64) ]
    
    return tags

def gen_fake_conds(real_conds, num_style, num_hair, num_eyes):
    batch_size = real_conds.size(0)
    style_idx = np.random.randint(num_style, size=(batch_size,))
    hair_idx = np.random.randint(num_hair, size=(batch_size,)) + num_style
    eyes_idx = np.random.randint(num_eyes, size=(batch_size,)) + num_style + num_hair

    fake_conds = np.zeros((batch_size, num_style + num_hair + num_eyes), dtype=np.float32)

    for i, (s_i, h_i, e_i) in enumerate(zip(style_idx, hair_idx, eyes_idx)):
        if real_conds[i][h_i] == 1. and real_conds[i][e_i] == 1. and real_conds[i][s_i] == 1: 
            h_i = (h_i + 1) % num_hair
        fake_conds[(i,i, i), (s_i, h_i, e_i)] = 1.
            
    fake_conds = torch.FloatTensor(fake_conds)

    return fake_conds

def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1: 
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)

############# Preparation #############

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nconds = int(opt.nconds)

############# Data Folder #############

img_dir = opt.dataroot + 'imgs'
tag_file = opt.dataroot + 'tags.csv'
val_tag_file = 'val_tags.txt'

############# Generator Model #############

class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.embedding = nn.Linear(nz+nconds,ngf*8*4*4,bias=False)
        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      3, 4, 2, 1, bias=False),
            nn.Tanh() 
        )
    def forward(self, noises, conds):
        inputs = torch.cat((noises, conds), dim=1)
        inputs = self.embedding(inputs)
        inputs = inputs.view(-1, ngf * 8, 4, 4)
        outputs = self.main(inputs)
        return outputs

############# Discriminator Model #############

class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8)
        )
        self.out = nn.Sequential(
            nn.Conv2d(ndf * 8 + nconds, ndf, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, 1, 4, 1),
            nn.Sigmoid()
        )
    def forward(self, imgs, conds):
        hiddens = self.main(imgs)
        conds = conds.view(*conds.size(), 1, 1)
        conds = conds.repeat(1, 1, 4, 4)
        hiddens = torch.cat((hiddens, conds), dim=1)
        outputs = self.out(hiddens)
        outputs = outputs.squeeze()
        return outputs

netG = _netG()
netG.apply(weights_init)
print(netG)

netD = _netD()
netD.apply(weights_init)
print(netD)




if opt.mode == 'train':

    imgs = load_imgs(img_dir)
    tags = load_tags(tag_file)

    imgs = np.concatenate((imgs,np.flip(imgs,axis=0)),axis=0)
    tags = np.tile(tags,(2,1))

    print('Finished loading training images...')
    print('Number of training images: ',len(tags))

    imgs = torch.FloatTensor(imgs).permute(0,3,1,2)

    style_dict = {}
    hair_dict = {}
    eyes_dict = {}
    for style_feat in np.unique(tags[:, 0]):
        style_dict[style_feat] = len(style_dict)
    for hair_feat in np.unique(tags[:, 1]): 
        hair_dict[hair_feat] = len(hair_dict)
    for eyes_feat in np.unique(tags[:, 2]): 
        eyes_dict[eyes_feat] = len(eyes_dict)

    num_style = len(style_dict)
    num_hair = len(hair_dict)
    num_eyes = len(eyes_dict)

    print(style_dict)
    print(hair_dict)
    print(eyes_dict)

    conditions = torch.zeros((imgs.size(0), num_style + num_hair + num_eyes))
    for i, tag in enumerate(tags):
        conditions[(i, i, i), (style_dict[tag[0]], hair_dict[tag[1]] + num_style, eyes_dict[tag[2]] + num_style + num_hair)] = 1.

    dataset = TensorDataset(imgs,conditions)
    dataloader = DataLoader(dataset,
                            batch_size=opt.batchSize,
                            shuffle=True,
                            num_workers=int(opt.workers))

    val_tags = load_tags(val_tag_file)
    val_conditions = torch.zeros((len(val_tags), num_style + num_hair + num_eyes ))
    for i, tag in enumerate(val_tags): 
        val_conditions[(i, i, i), (style_dict[tag[0]], hair_dict[tag[1]] + num_style, eyes_dict[tag[2]] + num_style + num_hair)] = 1.
    val_noise = torch.Tensor(len(val_tags), nz).uniform_(-1.0, 1.0)


    if opt.cuda:
        val_noise = Variable(val_noise).cuda()
        val_conditions = Variable(val_conditions).cuda()
        netG = netG.cuda()
        netD = netD.cuda()

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    criterion = nn.BCELoss()
    
    for epoch in range(opt.niter):

        netD.train()
        netG.train()

        for local_step, (real_x, real_conds) in enumerate(dataloader):

            global_step = epoch*len(dataloader) + local_step


            ############################
            # (1) Update Discriminator #
            ############################

            optimizerD.zero_grad()
            real_y = torch.ones((real_x.size(0),))
            fake_y = torch.zeros((real_x.size(0),))
            tr_noise = torch.Tensor(real_x.size(0), nz).uniform_(-1.0, 1.0)
            fake_conds = gen_fake_conds(real_conds, num_style ,num_hair, num_eyes)
            if opt.cuda:
                real_x = Variable(real_x).cuda()
                real_y = Variable(real_y).cuda()
                fake_y = Variable(fake_y).cuda()
                tr_noise = Variable(tr_noise).cuda()
                real_conds = Variable(real_conds).cuda()
                fake_conds = Variable(fake_conds).cuda()
            else:
                real_x = Variable(real_x)
                real_y = Variable(real_y)
                fake_y = Variable(fake_y)
                tr_noise = Variable(tr_noise)
                real_conds = Variable(real_conds)
                fake_conds = Variable(fake_conds)
         
            # Train with (real image, real tags)
            output = netD(real_x, real_conds)
            errD_real = criterion(output, real_y)
            # Train with (real_image, fake tags)
            output = netD(real_x, fake_conds)
            errD_wrong = criterion(output, fake_y)
            # Train with (fake_image, real tags)
            gen_x = netG(tr_noise, real_conds)
            output = netD(gen_x.detach(), real_conds)
            errD_fake = criterion(output, fake_y)
            
            errD = errD_real + errD_wrong + errD_fake
            errD.backward()
            optimizerD.step()

            r_writer.add_scalar('errD', errD_real, global_step)
            w_writer.add_scalar('errD', errD_wrong, global_step)
            f_writer.add_scalar('errD', errD_fake, global_step)

            #########################
            # (2) Update Generator  #
            #########################
            optimizerG.zero_grad()
            fake_y.fill_(1.)
            output = netD(gen_x, real_conds)
            errG = criterion(output, fake_y)
            errG.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                % (epoch, opt.niter, local_step, len(dataloader),
                    errD.data, errG.data))

            writer.add_scalar('Loss/D', errD, global_step)
            writer.add_scalar('Loss/G', errG, global_step)

        val_x = netG(val_noise, val_conditions)

        writer.add_image('conditioned_fake_samples', vutils.make_grid(val_x.data, nrow=8, normalize=True), epoch)

        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))


if opt.mode == 'test':
    
    test_tag_file = opt.testing_file
    demo_tag_file = opt.demo_file

    style_dict = {'long hair': 0,
                    'short hair': 1}
    hair_dict = {'aqua hair': 0, 
                'black hair': 1, 
                'blonde hair': 2, 
                'blue hair': 3, 
                'brown hair': 4, 
                'gray hair': 5, 
                'green hair': 6, 
                'orange hair': 7, 
                'pink hair': 8, 
                'purple hair': 9, 
                'red hair': 10, 
                'white hair': 11}
    eyes_dict = {'aqua eyes': 0, 
                'black eyes': 1, 
                'blue eyes': 2, 
                'brown eyes': 3, 
                'green eyes': 4, 
                'orange eyes': 5, 
                'pink eyes': 6, 
                'purple eyes': 7, 
                'red eyes': 8, 
                'yellow eyes': 9}

    num_style = len(style_dict)
    num_eyes = len(eyes_dict)
    num_hair = len(hair_dict)

    val_tags = load_tags(test_tag_file)
    # val_tags = demo_tags(demo_tag_file)

    val_conditions = torch.zeros((len(val_tags), num_style + num_hair + num_eyes ))
    for i, tag in enumerate(val_tags): 
        val_conditions[(i, i, i), (style_dict[tag[0]], hair_dict[tag[1]] + num_style, eyes_dict[tag[2]] + num_style + num_hair)] = 1.
    val_noise = torch.Tensor(len(val_tags), nz).uniform_(-1.0, 1.0)

    if opt.cuda:
        val_noise = Variable(val_noise).cuda()
        val_conditions = Variable(val_conditions).cuda()
        netG = netG.cuda()

        model_path = 'acgan_netG.pth'
        netG.load_state_dict(torch.load(model_path))

        val_x = netG(val_noise, val_conditions)
        vutils.save_image(val_x.data,
                            'cgan.png',
                            nrow=8,
                            normalize=True,
                            padding=10,
                            pad_value=1)

    else:
        val_noise = Variable(val_noise)
        val_conditions = Variable(val_conditions)
        netG = netG

        model_path = 'weights/acgan_netG_42.pth'
        netG.load_state_dict(torch.load(model_path, map_location='cpu'))

        val_x = netG(val_noise, val_conditions)
        vutils.save_image(val_x.data,
                            'cgan1.png',
                            nrow=8,
                            normalize=True,
                            padding=10,
                            pad_value=1)