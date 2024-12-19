# Modules available
import os
import glob
import argparse
import time 
from tqdm import tqdm

# Installed modules
import numpy as np
import scipy, scipy.ndimage, scipy.io
import matplotlib.pyplot as plt
import nibabel as nib

# # Manually defined modules
from losses_3d import *
from model_components_3d import *
# from training_related_files.losses_3d import *
# from training_related_filesmodel_components_3d import *


# Monai modules used:
import monai
from monai.transforms import Compose, LoadImaged, Resized, ScaleIntensityRanged, ToTensord, EnsureChannelFirstd
from monai.data.utils import list_data_collate
from monai.data import DataLoader, Dataset, decollate_batch
from monai.losses import DiceCELoss

# PyTorch:
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random


class decoder3D_spade_dropped1(nn.Module):
    def __init__(self, out_channels=1, features=[64, 128, 256, 512], final_activation = "tanh"):
        super(decoder3D_spade_dropped1, self).__init__()
        '''
        Test 2 will use spadeblock after conv transpose 2d but before skip connections+doubleconv
        '''
        final_activations = {"tanh": nn.Hardtanh(), "sigmoid": nn.Sigmoid()}
        assert final_activation == "tanh" or "sigmoid", f"final_activation not valid. Valid: {final_activations.keys()}, given: {final_activation}" # might need to upgrade pylance to 3.6
        self.get_masks2 = nn.ModuleList()
        
        self.convtrans1 = nn.ConvTranspose3d(512*2, 512, kernel_size=2, stride=2)
        self.spadeblock1 =  SpadeBlock3D(512)
        self.doubleconv1 = DoubleConv3D(512*2, 512)

        self.convtrans2 = nn.ConvTranspose3d(256*2, 256, kernel_size=2, stride=2)
        self.spadeblock2 =  SpadeBlock3D(256)
        self.doubleconv2 = DoubleConv3D(256*2, 256)
        
        self.convtrans3 = nn.ConvTranspose3d(128*2, 128, kernel_size=2, stride=2)
        self.spadeblock3 =  SpadeBlock3D(128)
        self.doubleconv3 = DoubleConv3D(128*2, 128)
        
        self.convtrans4 = nn.ConvTranspose3d(64*2, 64, kernel_size=2, stride=2)
        self.spadeblock4 =  SpadeBlock3D(64)
        self.doubleconv4 = DoubleConv3D(64*2, 64)
        
        self.convtrans5 = nn.ConvTranspose3d(32*2, 32, kernel_size=2, stride=2)
        self.spadeblock5 =  SpadeBlock3D(32)
        self.doubleconv5 = DoubleConv3D(32, 32)
        
        
        inchannels = 1
        self.get_masks2.append(nn.Sequential(nn.Conv3d(inchannels, features[0], 3, 1, 1), 
                                   nn.BatchNorm3d(features[0]), nn.LeakyReLU()))
        inchannels = features[0]
        for feature in features[1:]:
            self.get_masks2.append(nn.Sequential(nn.Conv3d(inchannels, feature, 3, 2, 1), 
                                   nn.BatchNorm3d(feature), nn.LeakyReLU()))
            inchannels = feature

        self.final_conv = nn.Sequential(
                            nn.Conv3d(features[0], features[0], 3, 1, 1, bias=False), 
                            nn.BatchNorm3d(features[0]), 
                            nn.ReLU(inplace=True), 
                            nn.Conv3d(features[0], out_channels, 3, 1, 1, bias=False), 
                            final_activations[final_activation])
    
    def forward(self, x, skip_connections, mask):
        downsampled_masks = []
        downmask2 = mask

        for downsample_block2 in self.get_masks2:
            downmask2 = downsample_block2(downmask2)
            downsampled_masks.append(downmask2)
        
        downsampled_masks = downsampled_masks[::-1] # reverse list s.t. in correct order for spade blocks

        x = self.convtrans1(x)
        x = self.spadeblock1(x, downsampled_masks[0])
        skip_connection = skip_connections[0]
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.doubleconv1(concat_skip)
        
        x = self.convtrans2(x)
        x = self.spadeblock2(x, downsampled_masks[1])
        skip_connection = skip_connections[1]
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.doubleconv2(concat_skip)    

        x = self.convtrans3(x)
        x = self.spadeblock3(x, downsampled_masks[2])
        skip_connection = skip_connections[2]
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.doubleconv3(concat_skip)
        
        x = self.convtrans4(x)
        x = self.spadeblock4(x, downsampled_masks[3])
        skip_connection = skip_connections[3]
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.doubleconv4(concat_skip)
        
        x = self.convtrans5(x)
        x = self.spadeblock5(x, downsampled_masks[4])
        x = self.doubleconv5(x)
        
        return self.final_conv(x) 

class decoder3D_spade_dropped2(nn.Module):
    def __init__(self, out_channels=1, features=[64, 128, 256, 512], final_activation = "tanh"):
        super(decoder3D_spade_dropped2, self).__init__()
        '''
        Test 2 will use spadeblock after conv transpose 2d but before skip connections+doubleconv
        '''
        final_activations = {"tanh": nn.Hardtanh(), "sigmoid": nn.Sigmoid()}
        assert final_activation == "tanh" or "sigmoid", f"final_activation not valid. Valid: {final_activations.keys()}, given: {final_activation}" # might need to upgrade pylance to 3.6
        self.get_masks2 = nn.ModuleList()
        
        self.convtrans1 = nn.ConvTranspose3d(512*2, 512, kernel_size=2, stride=2)
        self.spadeblock1 =  SpadeBlock3D(512)
        self.doubleconv1 = DoubleConv3D(512*2, 512)

        self.convtrans2 = nn.ConvTranspose3d(256*2, 256, kernel_size=2, stride=2)
        self.spadeblock2 =  SpadeBlock3D(256)
        self.doubleconv2 = DoubleConv3D(256*2, 256)
        
        self.convtrans3 = nn.ConvTranspose3d(128*2, 128, kernel_size=2, stride=2)
        self.spadeblock3 =  SpadeBlock3D(128)
        self.doubleconv3 = DoubleConv3D(128*2, 128)
        
        self.convtrans4 = nn.ConvTranspose3d(64*2, 64, kernel_size=2, stride=2)
        self.spadeblock4 =  SpadeBlock3D(64)
        self.doubleconv4 = DoubleConv3D(64, 64)
        
        self.convtrans5 = nn.ConvTranspose3d(32*2, 32, kernel_size=2, stride=2)
        self.spadeblock5 =  SpadeBlock3D(32)
        self.doubleconv5 = DoubleConv3D(32, 32)
        
        
        inchannels = 1
        self.get_masks2.append(nn.Sequential(nn.Conv3d(inchannels, features[0], 3, 1, 1), 
                                   nn.BatchNorm3d(features[0]), nn.LeakyReLU()))
        inchannels = features[0]
        for feature in features[1:]:
            self.get_masks2.append(nn.Sequential(nn.Conv3d(inchannels, feature, 3, 2, 1), 
                                   nn.BatchNorm3d(feature), nn.LeakyReLU()))
            inchannels = feature

        self.final_conv = nn.Sequential(
                            nn.Conv3d(features[0], features[0], 3, 1, 1, bias=False), 
                            nn.BatchNorm3d(features[0]), 
                            nn.ReLU(inplace=True), 
                            nn.Conv3d(features[0], out_channels, 3, 1, 1, bias=False), 
                            final_activations[final_activation])
    
    def forward(self, x, skip_connections, mask):
        downsampled_masks = []
        downmask2 = mask

        for downsample_block2 in self.get_masks2:
            downmask2 = downsample_block2(downmask2)
            downsampled_masks.append(downmask2)
        
        downsampled_masks = downsampled_masks[::-1] # reverse list s.t. in correct order for spade blocks

        x = self.convtrans1(x)
        x = self.spadeblock1(x, downsampled_masks[0])
        skip_connection = skip_connections[0]
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.doubleconv1(concat_skip)
        
        x = self.convtrans2(x)
        x = self.spadeblock2(x, downsampled_masks[1])
        skip_connection = skip_connections[1]
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.doubleconv2(concat_skip)      

        x = self.convtrans3(x)
        x = self.spadeblock3(x, downsampled_masks[2])
        skip_connection = skip_connections[2]
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.doubleconv3(concat_skip)
        
        x = self.convtrans4(x)
        x = self.spadeblock4(x, downsampled_masks[3])
        x = self.doubleconv4(x)
        
        x = self.convtrans5(x)
        x = self.spadeblock5(x, downsampled_masks[4])
        x = self.doubleconv5(x)
        
        return self.final_conv(x) 
class decoder3D_spade_dropped3(nn.Module):
    def __init__(self, out_channels=1, features=[64, 128, 256, 512], final_activation = "tanh"):
        super(decoder3D_spade_dropped3, self).__init__()
        '''
        Test 2 will use spadeblock after conv transpose 2d but before skip connections+doubleconv
        '''
        final_activations = {"tanh": nn.Hardtanh(), "sigmoid": nn.Sigmoid()}
        assert final_activation == "tanh" or "sigmoid", f"final_activation not valid. Valid: {final_activations.keys()}, given: {final_activation}" # might need to upgrade pylance to 3.6
        self.get_masks2 = nn.ModuleList()
        
        self.convtrans1 = nn.ConvTranspose3d(512*2, 512, kernel_size=2, stride=2)
        self.spadeblock1 =  SpadeBlock3D(512)
        self.doubleconv1 = DoubleConv3D(512*2, 512)

        self.convtrans2 = nn.ConvTranspose3d(256*2, 256, kernel_size=2, stride=2)
        self.spadeblock2 =  SpadeBlock3D(256)
        self.doubleconv2 = DoubleConv3D(256*2, 256)
        
        self.convtrans3 = nn.ConvTranspose3d(128*2, 128, kernel_size=2, stride=2)
        self.spadeblock3 =  SpadeBlock3D(128)
        self.doubleconv3 = DoubleConv3D(128, 128)
        
        self.convtrans4 = nn.ConvTranspose3d(64*2, 64, kernel_size=2, stride=2)
        self.spadeblock4 =  SpadeBlock3D(64)
        self.doubleconv4 = DoubleConv3D(64, 64)
        
        self.convtrans5 = nn.ConvTranspose3d(32*2, 32, kernel_size=2, stride=2)
        self.spadeblock5 =  SpadeBlock3D(32)
        self.doubleconv5 = DoubleConv3D(32, 32)
        
        
        inchannels = 1
        self.get_masks2.append(nn.Sequential(nn.Conv3d(inchannels, features[0], 3, 1, 1), 
                                   nn.BatchNorm3d(features[0]), nn.LeakyReLU()))
        inchannels = features[0]
        for feature in features[1:]:
            self.get_masks2.append(nn.Sequential(nn.Conv3d(inchannels, feature, 3, 2, 1), 
                                   nn.BatchNorm3d(feature), nn.LeakyReLU()))
            inchannels = feature

        self.final_conv = nn.Sequential(
                            nn.Conv3d(features[0], features[0], 3, 1, 1, bias=False), 
                            nn.BatchNorm3d(features[0]), 
                            nn.ReLU(inplace=True), 
                            nn.Conv3d(features[0], out_channels, 3, 1, 1, bias=False), 
                            final_activations[final_activation])
    
    def forward(self, x, skip_connections, mask):
        downsampled_masks = []
        downmask2 = mask

        for downsample_block2 in self.get_masks2:
            downmask2 = downsample_block2(downmask2)
            downsampled_masks.append(downmask2)
        
        downsampled_masks = downsampled_masks[::-1] # reverse list s.t. in correct order for spade blocks

        x = self.convtrans1(x)
        x = self.spadeblock1(x, downsampled_masks[0])
        skip_connection = skip_connections[0]
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.doubleconv1(concat_skip)
        
        x = self.convtrans2(x)
        x = self.spadeblock2(x, downsampled_masks[1])
        skip_connection = skip_connections[1]
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.doubleconv2(concat_skip)     

        x = self.convtrans3(x)
        x = self.spadeblock3(x, downsampled_masks[2])
        x = self.doubleconv3(x)
        
        x = self.convtrans4(x)
        x = self.spadeblock4(x, downsampled_masks[3])
        x = self.doubleconv4(x)
        
        x = self.convtrans5(x)
        x = self.spadeblock5(x, downsampled_masks[4])
        x = self.doubleconv5(x)
        
        return self.final_conv(x) 
class decoder3D_spade_dropped4(nn.Module):
    def __init__(self, out_channels=1, features=[32, 64, 128, 256, 512], final_activation = "tanh"):
        super(decoder3D_spade_dropped4, self).__init__()
        '''
        Test 2 will use spadeblock after conv transpose 2d but before skip connections+doubleconv
        '''
        final_activations = {"tanh": nn.Hardtanh(), "sigmoid": nn.Sigmoid()}
        assert final_activation == "tanh" or "sigmoid", f"final_activation not valid. Valid: {final_activations.keys()}, given: {final_activation}" # might need to upgrade pylance to 3.6
        self.get_masks2 = nn.ModuleList()
        
        self.convtrans1 = nn.ConvTranspose3d(512*2, 512, kernel_size=2, stride=2)
        self.spadeblock1 =  SpadeBlock3D(512)
        self.doubleconv1 = DoubleConv3D(512*2, 512)

        self.convtrans2 = nn.ConvTranspose3d(256*2, 256, kernel_size=2, stride=2)
        self.spadeblock2 =  SpadeBlock3D(256)
        self.doubleconv2 = DoubleConv3D(256, 256)
        
        self.convtrans3 = nn.ConvTranspose3d(128*2, 128, kernel_size=2, stride=2)
        self.spadeblock3 =  SpadeBlock3D(128)
        self.doubleconv3 = DoubleConv3D(128, 128)
        
        self.convtrans4 = nn.ConvTranspose3d(64*2, 64, kernel_size=2, stride=2)
        self.spadeblock4 =  SpadeBlock3D(64)
        self.doubleconv4 = DoubleConv3D(64, 64)
        
        self.convtrans5 = nn.ConvTranspose3d(32*2, 32, kernel_size=2, stride=2)
        self.spadeblock5 =  SpadeBlock3D(32)
        self.doubleconv5 = DoubleConv3D(32, 32)
        
        
        inchannels = 1
        self.get_masks2.append(nn.Sequential(nn.Conv3d(inchannels, features[0], 3, 1, 1), 
                                   nn.BatchNorm3d(features[0]), nn.LeakyReLU()))
        inchannels = features[0]
        for feature in features[1:]:
            self.get_masks2.append(nn.Sequential(nn.Conv3d(inchannels, feature, 3, 2, 1), 
                                   nn.BatchNorm3d(feature), nn.LeakyReLU()))
            inchannels = feature

        self.final_conv = nn.Sequential(
                            nn.Conv3d(features[0], features[0], 3, 1, 1, bias=False), 
                            nn.BatchNorm3d(features[0]), 
                            nn.ReLU(inplace=True), 
                            nn.Conv3d(features[0], out_channels, 3, 1, 1, bias=False), 
                            final_activations[final_activation])
    
    def forward(self, x, skip_connections, mask):
        downsampled_masks = []
        downmask2 = mask

        for downsample_block2 in self.get_masks2:
            downmask2 = downsample_block2(downmask2)
            downsampled_masks.append(downmask2)
        
        downsampled_masks = downsampled_masks[::-1] # reverse list s.t. in correct order for spade blocks

        x = self.convtrans1(x)
        x = self.spadeblock1(x, downsampled_masks[0])
        skip_connection = skip_connections[0]
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.doubleconv1(concat_skip)
        
        x = self.convtrans2(x)
        x = self.spadeblock2(x, downsampled_masks[1])
        x = self.doubleconv2(x)

        x = self.convtrans3(x)
        x = self.spadeblock3(x, downsampled_masks[2])
        x = self.doubleconv3(x)
        
        x = self.convtrans4(x)
        x = self.spadeblock4(x, downsampled_masks[3])
        x = self.doubleconv4(x)

        x = self.convtrans5(x)
        x = self.spadeblock5(x, downsampled_masks[4])
        x = self.doubleconv5(x)
        return self.final_conv(x)     
class decoder3D_spade_dropped5(nn.Module):
    def __init__(self, out_channels=1, features=[32, 64, 128, 256, 512], final_activation = "tanh"):
        super(decoder3D_spade_dropped5, self).__init__()
        '''
        Test 2 will use spadeblock after conv transpose 2d but before skip connections+doubleconv
        '''
        final_activations = {"tanh": nn.Hardtanh(), "sigmoid": nn.Sigmoid()}
        assert final_activation == "tanh" or "sigmoid", f"final_activation not valid. Valid: {final_activations.keys()}, given: {final_activation}" # might need to upgrade pylance to 3.6
        self.get_masks2 = nn.ModuleList()
        
        self.convtrans1 = nn.ConvTranspose3d(512*2, 512, kernel_size=2, stride=2)
        self.spadeblock1 =  SpadeBlock3D(512)
        self.doubleconv1 = DoubleConv3D(512, 512)

        self.convtrans2 = nn.ConvTranspose3d(256*2, 256, kernel_size=2, stride=2)
        self.spadeblock2 =  SpadeBlock3D(256)
        self.doubleconv2 = DoubleConv3D(256, 256)
        
        self.convtrans3 = nn.ConvTranspose3d(128*2, 128, kernel_size=2, stride=2)
        self.spadeblock3 =  SpadeBlock3D(128)
        self.doubleconv3 = DoubleConv3D(128, 128)
        
        self.convtrans4 = nn.ConvTranspose3d(64*2, 64, kernel_size=2, stride=2)
        self.spadeblock4 =  SpadeBlock3D(64)
        self.doubleconv4 = DoubleConv3D(64, 64)
        
        self.convtrans5 = nn.ConvTranspose3d(32*2, 32, kernel_size=2, stride=2)
        self.spadeblock5 =  SpadeBlock3D(32)
        self.doubleconv5 = DoubleConv3D(32, 32)
        
        
        inchannels = 1
        self.get_masks2.append(nn.Sequential(nn.Conv3d(inchannels, features[0], 3, 1, 1), 
                                   nn.BatchNorm3d(features[0]), nn.LeakyReLU()))
        inchannels = features[0]
        for feature in features[1:]:
            self.get_masks2.append(nn.Sequential(nn.Conv3d(inchannels, feature, 3, 2, 1), 
                                   nn.BatchNorm3d(feature), nn.LeakyReLU()))
            inchannels = feature

        self.final_conv = nn.Sequential(
                            nn.Conv3d(features[0], features[0], 3, 1, 1, bias=False), 
                            nn.BatchNorm3d(features[0]), 
                            nn.ReLU(inplace=True), 
                            nn.Conv3d(features[0], out_channels, 3, 1, 1, bias=False), 
                            final_activations[final_activation])
    
    def forward(self, x, skip_connections, mask):
        downsampled_masks = []
        downmask2 = mask

        for downsample_block2 in self.get_masks2:
            downmask2 = downsample_block2(downmask2)
            downsampled_masks.append(downmask2)
        
        downsampled_masks = downsampled_masks[::-1] # reverse list s.t. in correct order for spade blocks

        x = self.convtrans1(x)
        x = self.spadeblock1(x, downsampled_masks[0])
        x = self.doubleconv1(x)
        
        x = self.convtrans2(x)
        x = self.spadeblock2(x, downsampled_masks[1])
        x = self.doubleconv2(x)    

        x = self.convtrans3(x)
        x = self.spadeblock3(x, downsampled_masks[2])
        x = self.doubleconv3(x)
        
        x = self.convtrans4(x)
        x = self.spadeblock4(x, downsampled_masks[3])
        x = self.doubleconv4(x)
        
        x = self.convtrans5(x)
        x = self.spadeblock5(x, downsampled_masks[4])
        x = self.doubleconv5(x)
        
        return self.final_conv(x) 
    
parser = argparse.ArgumentParser()
parser.add_argument( '-e', '--expt-name')  
parser.add_argument( '-sd', '--savedirectory')  
parser.add_argument( '-ep', '--epochs', type=int)  
parser.add_argument( '-s', '--seed', type=int) 
parser.add_argument( '-b', '--batchsize', type=int) 
parser.add_argument( '-r', '--root-dir' )
parser.add_argument( '-hf', '--healthyfilepath' )
parser.add_argument( '-df', '--diseasefilepath' )
parser.add_argument( '-dt', '--decodertype' )
parser.add_argument( '-wfg', '--weightfakegen', type=float)
parser.add_argument( '-wr', '--weightrecon', type=float)
parser.add_argument( '-ws', '--weightseg', type=float)
parser.add_argument( '-wc', '--weightcritic', type=float)
parser.add_argument( '-ds', '--dropskip', type=int)

args = parser.parse_args() 

expt_name = args.expt_name
save_dir = args.savedirectory
numepochs = args.epochs
seed_number = args.seed
batchsize = args.batchsize


root_dir = args.root_dir
healthy_filepath = args.healthyfilepath
disease_filepath = args.diseasefilepath
decodertype = args.decodertype
wfakegen = args.weightfakegen
wrecon = args.weightrecon
wseg = args.weightseg
wcritic = args.weightcritic
dropskip = args.dropskip

random.seed(seed_number)
np.random.seed(seed_number)
torch.manual_seed(seed_number)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Discriminator_decomp3D_64_smallerlatent(nn.Module):
    def __init__(self, features_in=1024, features_d=64):
        super(Discriminator_decomp3D_64_smallerlatent, self).__init__()      
        
        self.block1 =  nn.Sequential(nn.Conv3d(features_in, features_d, kernel_size=4, stride=2, padding=1),nn.LeakyReLU(0.2))
#         self.block2 =  nn.Sequential(nn.Conv3d(features_d, features_d, kernel_size=4, stride=2, padding=1),nn.LeakyReLU(0.2))
        self.block3 =  nn.Sequential(nn.Conv3d(features_d, features_d, kernel_size=4, stride=2, padding=1),nn.LeakyReLU(0.2))
        self.block4 =  nn.Sequential(nn.Conv3d(features_d, features_d, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.2))
        self.block5 =  nn.Conv3d(features_d, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

##**************************************************************************************
##  Obtaining PET, CT, Segmentation files and partitioning into train/val splits
##**************************************************************************************
fhealthy = open(healthy_filepath, "r")
fdisease = open(disease_filepath, "r")

healthy_files = fhealthy.readlines()
disease_files = fdisease.readlines()

fhealthy.close()
fdisease.close()

subset_healthy = healthy_files[:]
subset_disease = disease_files[:]

healthy = []
for example in subset_healthy:
    path = example.strip().split('dataset/')[-1]
    petf = f"{root_dir}{path}pet_bladder_crop_img.nii.gz"
    ctf = f"{root_dir}{path}ct_bladder_crop_img.nii.gz"
    maskf = f"{root_dir}{path}mask_bladder_crop_img.nii.gz"
    healthy.append({"pet": petf, "ct": ctf, "label": maskf})
    
disease = []
for example in subset_disease:
    path = example.strip().split('dataset/')[-1]
    petf = f"{root_dir}{path}pet_bladder_crop_img.nii.gz"
    ctf = f"{root_dir}{path}ct_bladder_crop_img.nii.gz"
    maskf = f"{root_dir}{path}mask_bladder_crop_img.nii.gz"
    disease.append({"pet": petf, "ct": ctf, "label": maskf})

draft_transforms = Compose(
    [LoadImaged(keys=["pet", 'ct', "label"]),
        EnsureChannelFirstd(keys=["pet", 'ct', "label"]),
        ScaleIntensityRanged(
            keys=["ct"], a_min=-100, a_max=250,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        ScaleIntensityRanged(
            keys=["pet"], a_min=0, a_max=15,
            b_min=0.0, b_max=1.0, clip=True,
        ),     
        Resized(keys=["pet", "ct", "label"], spatial_size=(64, 64, 64), mode=("nearest")),
        ToTensord(keys=["pet", "ct", "label"])]
)
train_files = healthy[:566]+disease[:244]
val_files = healthy[566:637]+disease[244:275]
test_files = healthy[637:]+disease[275:]

# BASELINE MODEL:
# train_ds = Dataset(data=train_files, transform=draft_transforms)
# val_ds = Dataset(data=val_files, transform=draft_transforms)
# # test_ds = Dataset(data=test_files, transform=draft_transforms)

# train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=True, num_workers=0)
# val_loader = DataLoader(val_ds, batch_size=batchsize, shuffle=False, num_workers=0)
# # test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

# # PSEUDO-HEALTHY MODEL:
train_ds_healthy = Dataset(data=healthy[:566], transform=draft_transforms)
train_ds_disease = Dataset(data=disease[:244], transform=draft_transforms)
# train_ds_healthy = Dataset(data=healthy[:12], transform=draft_transforms)
# train_ds_disease = Dataset(data=disease[:12], transform=draft_transforms)

# val_ds_healthy = Dataset(data=healthy[120:135], transform=draft_transforms)
# val_ds_disease = Dataset(data=disease[120:135], transform=draft_transforms)
val_ds = Dataset(data=healthy[566:637] + disease[244:275], transform=draft_transforms)
# val_ds = Dataset(data=healthy[120:125] + disease[120:125], transform=draft_transforms)

train_loader_healthy = DataLoader(train_ds_healthy, batch_size=batchsize, shuffle=True, num_workers=0)
train_loader_disease = DataLoader(train_ds_disease, batch_size=batchsize, shuffle=True, num_workers=0)

# val_loader_healthy = DataLoader(val_ds_healthy, batch_size=1, shuffle=False, num_workers=0)
# val_loader_disease = DataLoader(val_ds_disease, batch_size=1, shuffle=False, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=batchsize*2, shuffle=False, num_workers=0)

##**************************************************************************************
##  Initializing model parameters:
##**************************************************************************************
torch.manual_seed(seed_number)
device = "cuda:0" # if torch.cuda.is_available() else "cpu"
lr = 1e-3

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
feature_list = [32, 64, 128, 256, 512]
encoder3d = encoder_earlysplit(features=feature_list).to(device)  
decoder_segment = decoder3D(out_channels=2, features= feature_list, final_activation='sigmoid').to(device)

if dropskip == 1:
     decoder3d =  decoder3D_spade_dropped1(out_channels=1, features=feature_list, final_activation="sigmoid").to(device)
        
elif dropskip == 2:
     decoder3d =  decoder3D_spade_dropped2(out_channels=1, features=feature_list, final_activation="sigmoid").to(device)
        
elif dropskip == 3:
     decoder3d =  decoder3D_spade_dropped3(out_channels=1, features=feature_list, final_activation="sigmoid").to(device)
    
elif dropskip == 4:
     decoder3d =  decoder3D_spade_dropped4(out_channels=1, features=feature_list, final_activation="sigmoid").to(device)
        
elif dropskip == 5:
     decoder3d =  decoder3D_spade_dropped5(out_channels=1, features=feature_list, final_activation="sigmoid").to(device)
        
critic = critic_disentanglement3D_smallerlatent().to(device)    
    
# initializate optimizer
opt_gen = optim.Adam(list(encoder3d.parameters()) + list(decoder_segment.parameters()) + list(decoder3d.parameters()), lr=lr)
opt_critic = optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))

dice_ce_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
l2_loss_fn = nn.MSELoss()
l1_loss_fn = nn.L1Loss()


##**************************************************************************************
##  Training:
##**************************************************************************************
val_interval = 2
best_metric = 1e6
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
best_metric_epoch = -1
epoch_loss_values = []
training_losses = []
validation_losses = []
metric_values = []
critic_losses = []
critic_avg_losses = []

earlystop_counter = 0
earlystop_patience = 10

writer = SummaryWriter(f"runs/{expt_name}")
for epoch in tqdm(range(numepochs)):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{numepochs}")
    epoch_loss = 0
    epoch_loss_seg = 0
    epoch_loss_reconfull = 0
    epoch_loss_reconl1 = 0
    epoch_loss_reconl2 = 0
    epoch_loss_critic = 0
    epoch_loss_genfake = 0
    
    step = 0
    encoder3d.train()
    decoder_segment.train()
    decoder3d.train()
    critic.train()

    for healthybatch, unhealthybatch in zip(train_loader_healthy,  train_loader_disease):
        step += 1
        batch_imgs = torch.cat((healthybatch['pet'], unhealthybatch['pet']))
        batch_masks = torch.cat((healthybatch['label'], unhealthybatch['label']))

        idx = torch.randperm(batch_imgs.shape[0])
        train_imgs = batch_imgs[idx].view(batch_imgs.size())
        train_masks = batch_masks[idx].view(batch_masks.size())
        
        (normal, normal_fskips), (abnormal, abnormal_fskips) = encoder3d(train_imgs.to(device))  
        mask_prediction = decoder_segment(abnormal, abnormal_fskips)

        mask_channel = mask_prediction[:, 1, :, :, :] #.detach().cpu()
        mask_argmax = torch.argmax(mask_prediction, dim=1)[:, :, :, :]
        mod_mask = torch.clone(mask_channel)
        mod_mask[mask_argmax==0] = 0          
        full_recon = decoder3d(normal, normal_fskips, mod_mask.unsqueeze(1).to(device))  
        
        l1_loss_value = l1_loss_fn(full_recon, train_imgs.to(device)) 
        l2_loss_value = l2_loss_fn(full_recon, train_imgs.to(device))    
        loss_recon_full = l1_loss_value + l2_loss_value
        
        writer.add_scalar("L1Loss-Training", l1_loss_value, epoch*step) 
        writer.add_scalar("L2Loss-Training", l2_loss_value, epoch*step) 
        writer.add_scalar("ReconstructionLoss-Training", loss_recon_full, epoch*step) 
        
        loss_seg = dice_ce_loss_fn(mask_prediction, train_masks.to(device))
        writer.add_scalar("SegmentationLoss-Training", loss_seg, epoch*step) 
        
        (normal, _), _ = encoder3d(healthybatch['pet'].to(device))
        (disease_normal, _), _ = encoder3d(unhealthybatch['pet'].to(device))  
        
        critic_avg_loss = 0
        for _ in range(CRITIC_ITERATIONS):
            critic_healthy = critic(normal).reshape(-1)
            critic_disease = critic(disease_normal).reshape(-1)
            gp = gradient_penalty_3d_v2(critic, normal, disease_normal, device=device)
            loss_critic = (-(torch.mean(critic_healthy) - torch.mean(critic_disease)) + LAMBDA_GP * gp)*wcritic
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            critic_avg_loss += loss_critic.item()
            critic_losses.append(loss_critic.detach().cpu())
        
        critic_avg_loss /=  CRITIC_ITERATIONS
        critic_avg_losses.append(critic_avg_loss)
        writer.add_scalar("CriticLoss-Training", critic_avg_loss, epoch*step) 
        
        gen_fake = critic(disease_normal).reshape(-1)
        loss_fake_gen = -torch.mean(gen_fake)
        writer.add_scalar("LossFakeGen-Training", loss_fake_gen, epoch*step) 

        loss_gen = loss_seg*wseg + loss_recon_full*wrecon + loss_fake_gen*wfakegen 
        writer.add_scalar("LossGen-Training", loss_gen, epoch*step) 
   
        epoch_loss += loss_gen.item() 

        epoch_loss_seg += loss_seg.item()
        epoch_loss_reconfull += loss_recon_full.item()
        epoch_loss_reconl1 += l1_loss_value.item()
        epoch_loss_reconl2 += l2_loss_value.item()
        
        epoch_loss_critic += critic_avg_loss
        epoch_loss_genfake += loss_fake_gen.item()
                
        encoder3d.zero_grad()
        decoder3d.zero_grad()
        decoder_segment.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
    epoch_loss /= step
    epoch_loss_seg /= step
    epoch_loss_reconfull /= step
    epoch_loss_reconl1 /= step
    epoch_loss_reconl2 /= step
    epoch_loss_critic /= step
    epoch_loss_genfake /= step
    
    
    epoch_loss_values.append(epoch_loss)
    writer.add_scalar("Epoch-LossGen-Training", epoch_loss, epoch) 
    writer.add_scalar("Epoch-Loss_Segmentation-Training", epoch_loss_seg, epoch) 
    writer.add_scalar("Epoch-LossReconstruction-Training", epoch_loss_reconfull, epoch) 
    writer.add_scalar("Epoch-LossReconL1-Training", epoch_loss_reconl1, epoch) 
    writer.add_scalar("Epoch-LossReconL2-Training", epoch_loss_reconl2, epoch) 
    writer.add_scalar("Epoch-LossCritic-Training", epoch_loss_critic, epoch) 
    writer.add_scalar("Epoch-LossGenFake-Training", epoch_loss_genfake, epoch) 
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")


    if (epoch + 1) % val_interval == 0:
        encoder3d.eval()
        decoder3d.eval()
        decoder_segment.eval()
        
        dices = []
        val_epoch_loss_seg = 0
        val_epoch_loss_reconfull = 0
        val_epoch_loss_reconl1 = 0
        val_epoch_loss_reconl2 = 0
        validation_loss = 0
        val_step = 0
        with torch.no_grad():
            metric_sum = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_step += 1
                val_inputs, val_labels = (
                    val_data["pet"],
                    val_data["label"],
                )
                
                (normal, normal_fskips), (abnormal, abnormal_fskips)= encoder3d(val_inputs.to(device))
                mask_prediction = decoder_segment(abnormal, abnormal_fskips)
                mask_channel = mask_prediction[:, 1, :, :, :] #.detach().cpu()
                mask_argmax = torch.argmax(mask_prediction, dim=1)[:, :, :, :]
                mod_mask = torch.clone(mask_channel)
                mod_mask[mask_argmax==0] = 0          
                full_recon = decoder3d(normal, normal_fskips, mod_mask.unsqueeze(1).to(device))  
                
                binary_segmentation_prediction = torch.argmax(mask_prediction, dim=1).detach().cpu()
                # calculate the dice between GT mask and mask prediction:
                y_true =  torch.flatten(val_labels[:, 0, :, :, :])
                y_pred =  torch.flatten(binary_segmentation_prediction)
                
                smooth = 1
                intersection = torch.sum(y_true*y_pred)
                dice_value = (2.*intersection+smooth)/(torch.sum(y_true) + torch.sum(y_pred) + smooth)
                dices.append(dice_value.item())
                
                # Calculating validation loss:
                loss_seg = dice_ce_loss_fn(mask_prediction, val_labels.to(device))
                l1_loss_value = l1_loss_fn(full_recon, val_inputs.to(device)) 
                l2_loss_value = l2_loss_fn(full_recon, val_inputs.to(device))    
                loss_recon_full = l1_loss_value + l2_loss_value
                
                val_epoch_loss_seg += loss_seg.item()
                val_epoch_loss_reconfull += loss_recon_full.item()
                val_epoch_loss_reconl1 += l1_loss_value.item()
                val_epoch_loss_reconl2 += l2_loss_value.item()
            
            metric = np.mean(dices)
            metric_values.append(metric)
            writer.add_scalar("Dice Metric-Validation", metric, epoch)
            
            val_epoch_loss_seg /= val_step
            val_epoch_loss_reconfull /= val_step
            val_epoch_loss_reconl1 /= val_step
            val_epoch_loss_reconl2 /= val_step
            
            writer.add_scalar("SegmentationLoss-Validation", val_epoch_loss_seg, epoch)
            writer.add_scalar("ReconstructionLoss-Validation", val_epoch_loss_reconfull, epoch)
            writer.add_scalar("L1Loss-Validation", val_epoch_loss_reconl1, epoch)  
            writer.add_scalar("L2Loss-Validation", val_epoch_loss_reconl2, epoch)  

            if best_metric > val_epoch_loss_seg:
                best_metric = val_epoch_loss_seg
                best_metric_epoch = epoch + 1
                corresponding_dice = metric
                earlystop_counter = 0
                torch.save(encoder3d.state_dict(), os.path.join(save_dir , expt_name + "_best_encoder3d.pth"))
                torch.save(decoder_segment.state_dict(), os.path.join(save_dir , expt_name + "_best_decoder_segment.pth"))   
                torch.save(decoder3d.state_dict(), os.path.join(save_dir , expt_name + "_best_decoder3d.pth"))   
                torch.save(critic.state_dict(), os.path.join(save_dir , expt_name + "_best_critic.pth"))   
                torch.save(opt_gen.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_opt_gen.pth"))  
                torch.save(opt_critic.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_opt_critic.pth")) 
                print("saved new best metric model")
    
            print(
                f"current epoch: {epoch + 1} current mean comboloss: {val_epoch_loss_seg:.4f}, corresponding dice: {metric:.4f}"
                f"\nbest comboloss value: {best_metric:.4f}, corresponding dice: {corresponding_dice:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
            
    if (epoch + 1) % 10 == 0:
        torch.save(encoder3d.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_encoder3d.pth"))
        torch.save(decoder_segment.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_decoder_segment.pth"))
        torch.save(decoder3d.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_decoder3d.pth"))   
        torch.save(critic.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_critic.pth"))  
        torch.save(opt_gen.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_opt_gen.pth"))  
        torch.save(opt_critic.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_opt_critic.pth")) 
        print(f"saved last model at epoch {epoch}")
        writer.add_hparams({"lr":lr, "Batchsize": batchsize, "NumEpochs": numepochs},\
                           {"Best ComboLoss-Validation" : best_metric, "Best Corresponding Dice": corresponding_dice, "Best metric epoch": best_metric_epoch, 'epoch': epoch})
            
torch.save(encoder3d.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_encoder3d.pth"))
torch.save(decoder_segment.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_decoder_segment.pth"))
torch.save(decoder3d.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_decoder3d.pth"))   
torch.save(critic.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_critic.pth"))  
print(f"saved last model at epoch {epoch}")
