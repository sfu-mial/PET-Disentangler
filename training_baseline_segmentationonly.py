# Available modules
import os
import argparse
from tqdm import tqdm

# Installed modules
import numpy as np

# Manually defined modules
from model_components_3d import *

# Monai modules used:
import monai
from monai.transforms import Compose, LoadImaged, Resized, ScaleIntensityRanged, ToTensord, EnsureChannelFirstd
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss

# PyTorch:
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
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

args = parser.parse_args() 
expt_name = args.expt_name
save_dir = args.savedirectory
numepochs = args.epochs
seed_number = args.seed
batchsize = args.batchsize
root_dir = args.root_dir
healthy_filepath = args.healthyfilepath
disease_filepath = args.diseasefilepath

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

feature_list = [32, 64, 128, 256, 512]
encoder3d = encoder_earlysplit_v1(features=feature_list).to(device)  
decoder_segment = decoder3D(out_channels=2, features= feature_list, final_activation='sigmoid').to(device)

# initializate optimizer
opt_gen = optim.Adam(list(encoder3d.parameters()) + list(decoder_segment.parameters()), lr=lr)

dice_ce_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
l2_loss_fn = nn.MSELoss()
l1_loss_fn = nn.L1Loss()

##**************************************************************************************
##  Training:
##**************************************************************************************
val_interval = 2
best_metric = 1e6
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

writer = SummaryWriter(f"runs/{expt_name}")

for epoch in tqdm(range(numepochs)):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{numepochs}")
    epoch_loss = 0
    epoch_loss_seg = 0

    step = 0
    encoder3d.train()
    decoder_segment.train()

    for healthybatch, unhealthybatch in zip(train_loader_healthy,  train_loader_disease):
        step += 1
        batch_imgs = torch.cat((healthybatch['pet'], unhealthybatch['pet']))
        batch_masks = torch.cat((healthybatch['label'], unhealthybatch['label']))

        idx = torch.randperm(batch_imgs.shape[0])
        train_imgs = batch_imgs[idx].view(batch_imgs.size())
        train_masks = batch_masks[idx].view(batch_masks.size())

        (normal, normal_fskips), (abnormal, abnormal_fskips) = encoder3d(train_imgs.to(device))  
        mask_prediction = decoder_segment(abnormal, abnormal_fskips)

        loss_seg = dice_ce_loss_fn(mask_prediction, train_masks.to(device))
        writer.add_scalar("SegmentationLoss-Training", loss_seg, epoch*step) 

        loss_gen = loss_seg
        writer.add_scalar("LossGen-Training", loss_gen, epoch*step) 

        epoch_loss += loss_gen.item() 
        epoch_loss_seg += loss_seg.item()

        encoder3d.zero_grad()
        decoder_segment.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    epoch_loss /= step
    epoch_loss_seg /= step
    epoch_loss_values.append(epoch_loss)
    writer.add_scalar("Epoch-LossGen-Training", epoch_loss, epoch) 
    writer.add_scalar("Epoch-Loss_Segmentation-Training", epoch_loss_seg, epoch) 
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")


    if (epoch + 1) % val_interval == 0:
        encoder3d.eval()
        decoder_segment.eval()

        dices = []
        val_epoch_loss_seg = 0

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

                val_epoch_loss_seg += loss_seg.item()

            metric = np.mean(dices)
            metric_values.append(metric)
            writer.add_scalar("Dice Metric-Validation", metric, epoch)
            val_epoch_loss_seg /= val_step
            writer.add_scalar("SegmentationLoss-Validation", val_epoch_loss_seg, epoch)

            if best_metric > val_epoch_loss_seg:
                best_metric = val_epoch_loss_seg
                best_metric_epoch = epoch + 1
                corresponding_dice = metric
                torch.save(encoder3d.state_dict(), os.path.join(save_dir , expt_name + "_best_encoder3d.pth"))
                torch.save(decoder_segment.state_dict(), os.path.join(save_dir , expt_name + "_best_decoder_segment.pth"))   
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean comboloss: {val_epoch_loss_seg:.4f}, corresponding dice: {metric:.4f}"
                f"\nbest comboloss value: {best_metric:.4f}, corresponding dice: {corresponding_dice:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
    if (epoch + 1) % 10 == 0:
        torch.save(encoder3d.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_encoder3d.pth"))
        torch.save(decoder_segment.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_decoder_segment.pth")) 
        print(f"saved last model at epoch {epoch}")
        writer.add_hparams({"lr":lr, "Batchsize": batchsize, "NumEpochs": numepochs},\
                           {"Best ComboLoss-Validation" : best_metric, "Best Corresponding Dice": corresponding_dice, "Best metric epoch": best_metric_epoch, 'epoch': epoch})

torch.save(encoder3d.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_encoder3d.pth"))
torch.save(decoder_segment.state_dict(), os.path.join(save_dir , expt_name + "_lastepoch_decoder_segment.pth"))
print(f"saved last model at epoch {epoch}")
 