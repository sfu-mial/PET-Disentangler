# Modules available
import os
import argparse
from tqdm import tqdm

# Installed modules
import numpy as np

# # Manually defined modules
from losses_3d import *
from model_components_3d import *

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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument( '-e', '--expt-name')  
parser.add_argument( '-sd', '--savedirectory')  
parser.add_argument( '-ep', '--epochs', type=int)  
parser.add_argument( '-s', '--seed', type=int) 
parser.add_argument( '-b', '--batchsize', type=int) 
parser.add_argument( '-r', '--root-dir' )
parser.add_argument( '-hf', '--healthyfilepath' )
parser.add_argument( '-df', '--diseasefilepath' )
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

wfakegen = args.weightfakegen
wrecon = args.weightrecon
wseg = args.weightseg
wcritic = args.weightcritic
dropskip = args.dropskip

##**************************************************************************************
##  Obtaining PET, CT, Segmentation files and partitioning into train/val splits
##**************************************************************************************
fhealthy = open(healthy_filepath, "r")
fdisease = open(disease_filepath, "r")

healthy_files = fhealthy.readlines()
disease_files = fdisease.readlines()

fhealthy.close()
fdisease.close()

subset_healthy = healthy_files[:150]
subset_disease = disease_files[:150]

healthy = []
for example in subset_healthy:
    path = example.strip().split('Documents/')[-1]
    petf = f"{root_dir}{path}pet_crop_img.nii.gz"
    ctf = f"{root_dir}{path}ct_crop_img.nii.gz"
    maskf = f"{root_dir}{path}mask_crop_img.nii.gz"
    healthy.append({"pet": petf, "ct": ctf, "label": maskf})
    
disease = []
for example in subset_disease:
    path = example.strip().split('Documents/')[-1]
    petf = f"{root_dir}{path}pet_crop_img.nii.gz"
    ctf = f"{root_dir}{path}ct_crop_img.nii.gz"
    maskf = f"{root_dir}{path}mask_crop_img.nii.gz"
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

train_ds_healthy = Dataset(data=healthy[:120], transform=draft_transforms)
train_ds_disease = Dataset(data=disease[:120], transform=draft_transforms)
val_ds = Dataset(data=healthy[120:135] + disease[120:135], transform=draft_transforms)

train_loader_healthy = DataLoader(train_ds_healthy, batch_size=batchsize, shuffle=True, num_workers=0)
train_loader_disease = DataLoader(train_ds_disease, batch_size=batchsize, shuffle=True, num_workers=0)
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
  
if dropskip == 0:
     decoder3d =  decoder3D_spade_fullskips(out_channels=1, features=feature_list, final_activation="sigmoid").to(device)
        
elif dropskip == 3:
     decoder3d =  decoder3D_spade_dropped3(out_channels=1, features=feature_list, final_activation="sigmoid").to(device)
    
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
metric_values = []
critic_losses = []
critic_avg_losses = []

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

        mask_channel = mask_prediction[:, 1, :, :, :] 
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
            gp = gradient_penalty_3d(critic, normal, disease_normal, device=device)
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
                torch.save(encoder3d.state_dict(), os.path.join(save_dir , expt_name + "_best_encoder3d.pth"))
                torch.save(decoder_segment.state_dict(), os.path.join(save_dir , expt_name + "_best_decoder_segment.pth"))   
                torch.save(decoder3d.state_dict(), os.path.join(save_dir , expt_name + "_best_decoder3d.pth"))   
                torch.save(critic.state_dict(), os.path.join(save_dir , expt_name + "_best_critic.pth"))   
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
        
        print(f"saved last model at epoch {epoch}")
        writer.add_hparams({"lr":lr, "Batchsize": batchsize, "NumEpochs": numepochs},\
                           {"Best ComboLoss-Validation" : best_metric, "Best Corresponding Dice": corresponding_dice, "Best metric epoch": best_metric_epoch, 'epoch': epoch})
            
torch.save(encoder3d.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_encoder3d.pth"))
torch.save(decoder_segment.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_decoder_segment.pth"))
torch.save(decoder3d.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_decoder3d.pth"))   
torch.save(critic.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_critic.pth"))  
print(f"saved last model at epoch {epoch}")
