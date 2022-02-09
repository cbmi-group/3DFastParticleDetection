# 3DFastParticleDetection
This repo is for the paper "FAST PARTICLE PICKING FOR CRYO-ELECTRON TOMOGRAPHY USING ONE-STAGE DETECTION" (ISBI 2022). We provide a one-stage detection model that locates and classifies particles in 3D tomograms. 

---

## Overview

### Data Preparation

SHREC dataset can be downloaded from [SHREC.net](https://www.shrec.net/cryo-et/2020/). We normalize the reconstruction file and cut it in half to fit the location file. The file is named as "reconstruction_norm.mrc". 

3D rotation was also used for training data augmentation, but had little improvement. 

It's also easy to build your own dataset. See dataset/SHREC3D.py as an example. The only thing you need to do is to split the total data volume into batches. 

### Model

![](https://github.com/cbmi-group/3DFastParticleDetection/fig/Fig1.PNG)

![](https://github.com/cbmi-group/3DFastParticleDetection/fig/Fig2.PNG)

---

## How to use

### Dependencies
  - python 3.7.3
  - torch>=1.7.0+cu110
  - torchvision>=0.5.0

### Quick start 
run train.py
```
python train.py --gpu_ids='0,1,2,3' --total_epoches=150 --batch_size=32
```
See more controls in train.py file. 

(Note that `--pretrained` can load a pretrained model, and LOAD/SAVE methods can be seen in 'util/utils.py')


## Contributing 
Code for this projects developped at CBMI Group (Computational Biology and Machine Intelligence Group).

CBMI at National Laboratory of Pattern Recognition, INSTITUTE OF AUTOMATION, CHINESE ACADEMY OF SCIENCES

Bug reports and pull requests are welcome on GitHub at https://github.com/cbmi-group/3DFastParticleDetection