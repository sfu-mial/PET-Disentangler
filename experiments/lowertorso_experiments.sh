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

# training lower torso, segmentation only experiment by setting experiment type variable et to "segonly":
python training_fulldataset_lowertorso_ablation.py -e 'ablationstudy_bladder_segonly_run1' -sd '../checkpoints' -ep 300 -s 0 -b 2 -r './' -hf 'TCIA_bladder_healthyfile.txt' -df 'TCIA_bladder_diseasefile.txt' -dt 'test2v2' -wfg 0.001 -wr 10. -ws 100. -wc 0.01 -et "segonly"
# training lower torso, segrecon experiment by setting experiment type variable et to "segrecon":
python training_fulldataset_lowertorso_ablation.py -e 'ablationstudy_bladder_segrecon_run1' -sd '../checkpoints' -ep 300 -s 0 -b 2 -r './' -hf 'TCIA_bladder_healthyfile.txt' -df 'TCIA_bladder_diseasefile.txt' -dt 'test2v2' -wfg 0.001 -wr 10. -ws 100. -wc 0.01 -et "segrecon"
# training lower torso, segreconhealthy experiment by setting experiment type variable et to "segreconhealthy":
python training_fulldataset_lowertorso_ablation.py -e 'ablationstudy_bladder_segreconhealthy_run1' -sd '../checkpoints' -ep 300 -s 0 -b 2 -r './' -hf 'TCIA_bladder_healthyfile.txt' -df 'TCIA_bladder_diseasefile.txt' -dt 'test2v2' -wfg 0.001 -wr 10. -ws 100. -wc 0.01 -et "segreconhealthy"
python training_fulldataset_lowertorso_petdisentangler.py -e 'fullmodel_bladder_run1_drop3' -sd '../checkpoints' -ep 300 -s 0 -b 2 -r './' -hf 'TCIA_bladder_healthyfile.txt' -df 'TCIA_bladder_diseasefile.txt' -dt 'test2v2' -wfg 0.001 -wr 10. -ws 100. -wc 0.01 -ds 3