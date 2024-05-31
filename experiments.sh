#!/bin/bash

python training_baseline_segmentationonly.py -e "baseline_unet_segmentation" -sd './' -ep 100 -s 0 -b 2 -r './' -hf 'TCIA_aorta_healthy.txt' -df 'TCIA_aorta_disease.txt'

python training_petdisentangler.py -e "petdisentangler_fullskips" -sd './' -ep 100 -s 0 -b 2 -r './' -hf 'TCIA_aorta_healthy.txt' -df 'TCIA_aorta_disease.txt' -wfg 0.001 -wr 10. -ws 1. -wc 0.01 -ds 0

python training_petdisentangler.py -e "petdisentangler_noskips" -sd './' -ep 100 -s 0 -b 2 -r './' -hf 'TCIA_aorta_healthy.txt' -df 'TCIA_aorta_disease.txt' -wfg 0.001 -wr 10. -ws 1. -wc 0.01 -ds 5

python training_petdisentangler.py -e "petdisentangler_optimizedskips" -sd './' -ep 100 -s 0 -b 2 -r './' -hf 'TCIA_aorta_healthy.txt' -df 'TCIA_aorta_disease.txt' -wfg 0.001 -wr 10. -ws 1. -wc 0.01 -ds 3