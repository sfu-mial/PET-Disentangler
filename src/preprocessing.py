import os
import nibabel as nib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.patches as patches 
from totalsegmentator.python_api import totalsegmentator

root_dir = "../dataset/FDG-PET-CT-Lesions/"
# organ segmentations to obtain from TotalSegmentator:
list_classes = ["kidney_right", "kidney_left", "heart", "aorta", "liver", "urinary_bladder"]

patients = sorted(os.listdir(root_dir))
failed_examples = []

fhealthy_aorta = open("../dataset/TCIA_aorta_healthy.txt", "w")
fdisease_aorta = open("../dataset/TCIA_aorta_disease.txt", "w")
fhealthy_bladder = open("../dataset/TCIA_bladder_healthyfile.txt", "w")
fdisease_bladder = open("../dataset/TCIA_bladder_diseasefile.txt", "w")


for patient_i in tqdm(patients):
    patient_path = f"{root_dir}{patient_i}/"
    studies = sorted(os.listdir(patient_path))

    for study_i in studies:
        study_path = f"{patient_path}{study_i}/"
        # Generate the organ segmentations and save in subset_segmentations directory per study:
        totalsegmentator(f"{study_path}CTres.nii.gz", f"{study_path}subset_segmentations/", nr_thr_saving=1, fast=True, roi_subset=list_classes)
        
        ctres = nib.load(f"{patient_path}{study_i}/CTres.nii.gz")
        ct_img = ctres.get_fdata()
        pet = nib.load(f"{patient_path}{study_i}/SUV.nii.gz")
        pet_img = pet.get_fdata()     
        mask = nib.load(f"{patient_path}{study_i}/SEG.nii.gz")
        mask_img = mask.get_fdata()
 

        ## GENERATING UPPER TORSO CROPPED DATA:
        # Center cropping using aorta center coordinate:
        aorta_data = nib.load(aorta)
        aorta_img = aorta_data.get_fdata()
        rows, cols, depths = np.where(aorta_img>0)
        rows_m = int(np.median(rows))
        cols_m = int(np.median(cols))
        depths_m = int(np.median(depths))
        
        mask_fov_aorta = mask_img[rows_m-64:rows_m+64, cols_m-64:cols_m+64, depths_m-64:depths_m+64]
        mask_fov_sum_aorta = sum(mask_fov_aorta.flatten())

        # If mask pixels exist in mask_fov_aorta, then sum is >0 and these are disease examples:
        if mask_fov_sum_aorta > 0:
            fdisease_aorta.write(f"{patient_path}{study_i}/\n")
        else:
            fhealthy_aorta.write(f"{patient_path}{study_i}/\n")
            
        # Generate upper torso crops for each file:
        aorta_crop = aorta_img[rows_m-64:rows_m+64, cols_m-64:cols_m+64, depths_m-64:depths_m+64]
        ct_crop = ct_img[rows_m-64:rows_m+64, cols_m-64:cols_m+64, depths_m-64:depths_m+64]
        pet_crop = pet_img[rows_m-64:rows_m+64, cols_m-64:cols_m+64, depths_m-64:depths_m+64]
        mask_crop = mask_img[rows_m-64:rows_m+64, cols_m-64:cols_m+64, depths_m-64:depths_m+64]
        
        aorta_crop_img = nib.Nifti1Image(aorta_crop, None)
        ct_crop_img = nib.Nifti1Image(ct_crop, None)
        pet_crop_img = nib.Nifti1Image(pet_crop, None)
        mask_crop_img = nib.Nifti1Image(mask_crop, None)
        
        nib.save(aorta_crop_img, f"{patient_path}{study_i}/aorta_crop_img.nii.gz")
        nib.save(ct_crop_img, f"{patient_path}{study_i}/ct_crop_img.nii.gz")
        nib.save(pet_crop_img, f"{patient_path}{study_i}/pet_crop_img.nii.gz")
        nib.save(mask_crop_img, f"{patient_path}{study_i}/mask_crop_img.nii.gz")
            

        ## GENERATING UPPER TORSO CROPPED DATA:
        # Center cropping using bladder center coordinate:
        urinary_bladder = f"{segmentations_path}urinary_bladder.nii.gz"
        urinary_bladder_data = nib.load(urinary_bladder)
        urinary_bladder_img = urinary_bladder_data.get_fdata()
        
        rows, cols, depths = np.where(urinary_bladder_img>0)
        rows_m  = int(np.median(rows))
        cols_m  = int(np.median(cols))
        depths_m  = int(np.median(depths))
        
        mask_fov_bladder = mask_img[rows_m-64:rows_m+64, cols_m-64:cols_m+64, depths_m-17:depths_m+111]
        mask_fov_sum_bladder = sum(mask_fov_bladder.flatten())
        
        # If mask pixels exist in mask_fov_bladder, then sum is >0 and these are disease examples:
        if mask_fov_sum_bladder > 0:
            fdisease_bladder.append(f"{patient_path}{study_i}/\n")
        else:
            fhealthy_bladder.append(f"{patient_path}{study_i}/\n")
            
        # Generate lower torso crops for each file:  
        mask_crop = mask_img[rows_m-64:rows_m+64, cols_m-64:cols_m+64, depths_m-17:depths_m+111]
        ct_crop = ct_img[rows_m-64:rows_m+64, cols_m-64:cols_m+64, depths_m-17:depths_m+111]
        pet_crop = pet_img[rows_m-64:rows_m+64, cols_m-64:cols_m+64, depths_m-17:depths_m+111]
        bladder_crop = urinary_bladder_img[rows_m-64:rows_m+64, cols_m-64:cols_m+64, depths_m-17:depths_m+111]
        
        bladder_crop_img = nib.Nifti1Image(bladder_crop, None)
        ct_crop_img = nib.Nifti1Image(ct_crop, None)
        pet_crop_img = nib.Nifti1Image(pet_crop, None)
        mask_crop_img = nib.Nifti1Image(mask_crop, None)
        
        nib.save(bladder_crop_img, f"{target_dir}{patient_i}_{study_i}_bladder_crop_img.nii.gz")
        nib.save(ct_crop_img, f"{target_dir}{patient_i}_{study_i}_ct_bladder_crop_img.nii.gz")
        nib.save(pet_crop_img, f"{target_dir}{patient_i}_{study_i}_pet_bladder_crop_img.nii.gz")
        nib.save(mask_crop_img, f"{target_dir}{patient_i}_{study_i}_mask_bladder_crop_img.nii.gz")

fhealthy_aorta.close()
fdisease_aorta.close()
fhealthy_bladder.close() 
fdisease_bladder.close()