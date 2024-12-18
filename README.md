# PET-Disentangler
This repository contains the official implementation for PET-Disentangler, as seen in the paper [Disentangled PET Lesion Segmentation](https://arxiv.org/abs/2411.01758), that approaches lesion segmentation using an image disentanglement framework. 

## Abstract
PET imaging is an invaluable tool in clinical settings as it captures the functional activity of both healthy anatomy and cancerous lesions. Developing automatic lesion segmentation methods for PET images is crucial since manual lesion segmentation is laborious and prone to inter- and intra-observer variability. We propose PET-Disentangler, a 3D disentanglement method that uses a 3D UNet-like encoder-decoder architecture to disentangle disease and normal healthy anatomical features with losses for segmentation, reconstruction, and healthy component plausibility. A critic network is used to encourage the healthy latent features to match the distribution of healthy samples and thus encourages these features to not contain any lesion-related features. Our quantitative results show that PET-Disentangler is less prone to incorrectly declaring healthy and high tracer uptake regions as cancerous lesions, since such uptake pattern would be assigned to the disentangled healthy component.

## Citation
If you find this work or implementation useful, please cite our paper: 
```
@misc{gatsak2024disentangledpetlesionsegmentation,
      title={Disentangled PET Lesion Segmentation}, 
      author={Tanya Gatsak and Kumar Abhishek and Hanene Ben Yedder and Saeid Asgari Taghanaki and Ghassan Hamarneh},
      year={2024},
      eprint={2411.01758},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2411.01758}, 
}
```
