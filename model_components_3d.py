import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchmetrics import PearsonCorrCoef

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock3D, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, 1, 1), 
            nn.BatchNorm3d(in_channels),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels, in_channels, 3, 1, 1), 
            nn.BatchNorm3d(in_channels),
        )
        
    def forward(self, x):
        output = self.resblock(x)
        output += x
        return F.leaky_relu(output)

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False), 
        nn.BatchNorm3d(out_channels), 
        nn.ReLU(inplace=True), 
        nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False), 
        nn.BatchNorm3d(out_channels), 
        nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.conv(x)
    
class DoubleConv3D_v2(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DoubleConv3D_v2, self).__init__()
        self.conv = nn.Sequential(
        nn.Conv3d(in_channels, mid_channels, 3, 1, 1, bias=False), 
        nn.BatchNorm3d(mid_channels), 
        nn.ReLU(inplace=True), 
        nn.Conv3d(mid_channels, out_channels, 3, 1, 1, bias=False), 
        nn.BatchNorm3d(out_channels), 
        nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.conv(x)

# https://towardsdatascience.com/implementing-spade-using-fastai-6ad86b94030a
class SpadeBlock3D(nn.Module):
    def __init__(self, in_channels):
        super(SpadeBlock3D, self).__init__()
        self.batchnorm = nn.BatchNorm3d(in_channels, affine=False)
        
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, in_channels, 3, 1, 1), 
                                   nn.BatchNorm3d(in_channels), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(in_channels, in_channels, 3, 1, 1), 
                                   nn.BatchNorm3d(in_channels), nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv3d(in_channels, in_channels, 3, 1, 1), 
                                   nn.BatchNorm3d(in_channels), nn.LeakyReLU())       
        
    def forward(self, x, mask):
        mask = self.conv1(mask)
        gamma = self.conv2(mask)
        beta = self.conv3(mask)

        return  self.batchnorm(x)*(1 + gamma) + beta

class SpadeResBlock3D(nn.Module):
    def __init__(self, num_channels):
        super(SpadeResBlock3D, self).__init__()
        self.spade1 = SpadeBlock3D(num_channels)
        self.spade2 = SpadeBlock3D(num_channels)
        
        self.conv1 = nn.Sequential(nn.LeakyReLU(), nn.Conv3d(num_channels, num_channels, 3, 1, 1))
        self.conv2 = nn.Sequential(nn.LeakyReLU(), nn.Conv3d(num_channels, num_channels, 3, 1, 1))
    
    def forward(self, x, mask):
        skip_x = x # original features
        x = self.conv1(self.spade1(skip_x, mask))
        x = self.conv2(self.spade2(x, mask))
        return x + skip_x  

class encoder_earlysplit(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256]):
        super(encoder_earlysplit, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        mid_channels = 32
        
        self.downs1 = nn.ModuleList()
        self.downs2 = nn.ModuleList()
        
        self.initial_conv = DoubleConv3D_v2(in_channels, mid_channels, features[0])
        self.conv1 = DoubleConv3D_v2(features[0], features[0], features[0]) 
        self.conv2 = DoubleConv3D_v2(features[0], features[0], features[0])
      
        in_channels = features[0]
        mid_channels = features[0]
        
        ## Encoding portion of UNet:
        for feature in features[1:]: 
            self.downs1.append(DoubleConv3D_v2(in_channels, mid_channels, feature))
            self.downs2.append(DoubleConv3D_v2(in_channels, mid_channels, feature))
            in_channels = feature
            mid_channels = feature
            
        self.bottleneck1 = DoubleConv3D_v2(features[-1], features[-1], features[-1]*2)
        self.bottleneck2 = DoubleConv3D_v2(features[-1], features[-1], features[-1]*2)
        
    def forward(self, x):
        x = self.initial_conv(x)
        healthy = self.conv1(x)
        disease = self.conv2(x)
        
        fskips_healthy = [healthy]        
        fskips_disease = [disease]        
        
        healthy = self.pool(healthy)
        disease = self.pool(disease)
 
        for down1, down2 in zip(self.downs1, self.downs2):   
            healthy = down1(healthy)
            fskips_healthy.append(healthy)
            healthy = self.pool(healthy)
            
            disease = down2(disease)
            fskips_disease.append(disease)
            disease = self.pool(disease)


        healthy = self.bottleneck1(healthy)
        disease = self.bottleneck2(disease)
        
        fskips_healthy = fskips_healthy[::-1]      
        fskips_disease = fskips_disease[::-1]    

        return (healthy, fskips_healthy), (disease, fskips_disease) 

class decoder3D(nn.Module):
    def __init__(self, out_channels=1, features=[64, 128, 256, 512], final_activation = "tanh"):
        super(decoder3D, self).__init__()
        final_activations = {"tanh": nn.Hardtanh(), "sigmoid": nn.Sigmoid()}
        assert final_activation == "tanh" or "sigmoid", f"final_activation not valid. Valid: {final_activations.keys()}, given: {final_activation}" 
        self.ups = nn.ModuleList()
        
        # Decoding portion of UNet:
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv3D(feature*2, feature))

        self.final_conv = nn.Sequential(nn.Conv3d(features[0], out_channels, kernel_size=1), final_activations[final_activation])

    def forward(self, x, skip_connections):
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
                
            # concat x to skip connection:
            concat_skip = torch.cat((skip_connection, x), dim=1) # I believe this dim should be the same. 
                
            # perform double conv block on the concatenated result:
            x = self.ups[idx+1](concat_skip)
                
        return self.final_conv(x) 
    
class critic_disentanglement3D_smallerlatent(nn.Module):
    def __init__(self, features_in=1024, features_d=64):
        super(critic_disentanglement3D_smallerlatent, self).__init__()      
        
        self.block1 =  nn.Sequential(nn.Conv3d(features_in, features_d, kernel_size=4, stride=2, padding=1),nn.LeakyReLU(0.2))
        self.block2 =  nn.Sequential(nn.Conv3d(features_d, features_d, kernel_size=3, stride=1, padding=1),nn.LeakyReLU(0.2))
        self.block3 =  nn.Conv3d(features_d, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
    
class decoder3D_spade_dropped3(nn.Module):
    def __init__(self, out_channels=1, features=[64, 128, 256, 512], final_activation = "tanh"):
        super(decoder3D_spade_dropped3, self).__init__()

        final_activations = {"tanh": nn.Hardtanh(), "sigmoid": nn.Sigmoid()}
        assert final_activation == "tanh" or "sigmoid", f"final_activation not valid. Valid: {final_activations.keys()}, given: {final_activation}" 
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

class decoder3D_spade_dropped5(nn.Module):
    def __init__(self, out_channels=1, features=[32, 64, 128, 256, 512], final_activation = "tanh"):
        super(decoder3D_spade_dropped5, self).__init__()

        final_activations = {"tanh": nn.Hardtanh(), "sigmoid": nn.Sigmoid()}
        assert final_activation == "tanh" or "sigmoid", f"final_activation not valid. Valid: {final_activations.keys()}, given: {final_activation}" 
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
    
class decoder3D_spade_fullskips(nn.Module):
    def __init__(self, out_channels=1, features=[64, 128, 256, 512], final_activation = "tanh"):
        super(decoder3D_spade_fullskips, self).__init__()
        
        final_activations = {"tanh": nn.Hardtanh(), "sigmoid": nn.Sigmoid()}
        assert final_activation == "tanh" or "sigmoid", f"final_activation not valid. Valid: {final_activations.keys()}, given: {final_activation}" 
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
        self.doubleconv5 = DoubleConv3D(32*2, 32)
        
        
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
        skip_connection = skip_connections[4]
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.doubleconv5(concat_skip)
        
        return self.final_conv(x) 
    