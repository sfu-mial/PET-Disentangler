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
# -et: experiment type, one of "segonly", "segreconhealthy", "segreconhealthy"
# -ds: number of skip connections to drop

# training upper torso, segmentation only experiment by setting experiment type variable et to "segonly":
python training_fulldataset_uppertorso_ablation.py -e 'ablationstudy_segonly_run3' -sd '../checkpoints' -ep 300 -s 0 -b 2 -r './' -hf 'TCIA_aorta_healthy.txt' -df 'TCIA_aorta_disease.txt' -dt 'test2v2' -wfg 0.001 -wr 10. -ws 1. -wc 0.01 -et "segonly"
# training upper torso, segrecon experiment by setting experiment type variable et to "segrecon":
python training_fulldataset_uppertorso_ablation.py -e 'ablationstudy_segrecon_run3' -sd '../checkpoints' -ep 300 -s 0 -b 2 -r './' -hf 'TCIA_aorta_healthy.txt' -df 'TCIA_aorta_disease.txt' -dt 'test2v2' -wfg 0.001 -wr 10. -ws 100. -wc 0.01 -et "segrecon"
# training upper torso, segreconhealthy experiment by setting experiment type variable et to "segreconhealthy":
python training_fulldataset_uppertorso_ablation.py -e 'ablationstudy_segreconhealthy_run3' -sd '../checkpoints' -ep 300 -s 0 -b 2 -r './' -hf 'TCIA_aorta_healthy.txt' -df 'TCIA_aorta_disease.txt' -dt 'test2v2' -wfg 0.001 -wr 10. -ws 100. -wc 0.01 -et "segreconhealthy"
# training upper torso, PET-Disentangler with optimal configuration of dropping 3 skip connections:
python training_fulldataset_uppertorso_petdisentangler.py -e 'fullmodel_run3_drop3_evenmoresegweight' -sd '../checkpoints' -ep 300 -s 0 -b 2 -r './' -hf 'TCIA_aorta_healthy.txt' -df 'TCIA_aorta_disease.txt' -dt 'test2v2' -wfg 0.001 -wr 10. -ws 100. -wc 0.01 -ds 3