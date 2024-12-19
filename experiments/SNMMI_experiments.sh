#!/bin/bash
# -e: experiment name
# -sd: directory to save model checkpoints
# -ep: number of training epochs
# -s: number for random seed variable
# -b: batch size 
# -r: root directory in which dataset is located
# -hf: file indicating paths of healthy studies, created during preprocessing.py
# -df: file indicating paths of disease studies, created during preprocessing.py
# -wfg: weight of pseudohealthy loss
# -wr: weight of reconstruction loss
# -ws: weight of segmentation loss
# -ds: number of skip connections to drop

# training baseline segmentation only method:
python training_SNMMIbaseline_segmentationonly.py -e "baseline_unet_segmentation" -sd '../checkpoints' -ep 100 -s 0 -b 2 -r './' -hf 'TCIA_aorta_healthy.txt' -df 'TCIA_aorta_disease.txt'
# training PET-Disentangler full-skips variant by setting ds to 0, indicating no skip connections to be dropped:
python training_SNMMI_petdisentangler.py -e "petdisentangler_fullskips" -sd '../checkpoints' -ep 100 -s 0 -b 2 -r './' -hf 'TCIA_aorta_healthy.txt' -df 'TCIA_aorta_disease.txt' -wfg 0.001 -wr 10. -ws 1. -wc 0.01 -ds 0
# training PET-Disentangler no-skips variant by setting ds to 5, indicating all skip connections to be dropped:
python training_SNMMI_petdisentangler.py -e "petdisentangler_noskips" -sd '../checkpoints' -ep 100 -s 0 -b 2 -r './' -hf 'TCIA_aorta_healthy.txt' -df 'TCIA_aorta_disease.txt' -wfg 0.001 -wr 10. -ws 1. -wc 0.01 -ds 5
# training PET-Disentangler optimal variant by setting ds to 3, indicating last 3 skip connections to be dropped:
python training_SNMMI_petdisentangler.py -e "petdisentangler_optimizedskips" -sd '../checkpoints' -ep 100 -s 0 -b 2 -r './' -hf 'TCIA_aorta_healthy.txt' -df 'TCIA_aorta_disease.txt' -wfg 0.001 -wr 10. -ws 1. -wc 0.01 -ds 3