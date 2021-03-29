# GAN-based Style Transformation to Improve Gesture-recognition Accuracy
## Requirement
- Python (>=3.6)
- Chainer (>=6.3.0)
- cupy
- numpy
## Config
Basically, all training settings are handled in the onfig file.
The config file has a dictionary hierarchy and is parsed according to core/utils/config.py.
Typical properties are described below.
### train strategy
|Property|Description|
| :---: | :--- |
|class_equal|Whether or not to include transformation between the same style-class.|
|n_gesture|The number of gesture classes.|
|generator.top / descriminator.top|The number of output channels of the first convolutional layer in the networks.|
|generator.use_sigmoid|Whether or not to apply sigmoid to the output of the generator.|
|display_interval|Interval iterations of print logs on display.|
|preview_interval|Interval iterations of saving preview data.|
|save_interval|Interval iterations of saving models.|
### test strategy
|Property|Description|
| :---: | :--- |
|ges_class|The gesture class of source data.|
|target_style|Target style ID (If the style is gesture, use the gesture ID; if it is user, use the user ID.)|
## Train
To train the networks:
```
python train.py {PARH_TO_CONFIG_FILE}
```
exsample:
```
python train.py configs/StarGAN_config.py
```

## Test
To transfer data with the trained network:
```
python test.py {PARH_TO_CONFIG_FILE}
```
exsample:
```
python test.py configs/StarGAN_config.py
```

## Citation
If you find this work useful for your research, please cite our [paper](https://dl.acm.org/doi/abs/10.1145/3432199):
```
@article{10.1145/3432199,
author = {Suzuki, Noeru and Watanabe, Yuki and Nakazawa, Atsushi},
title = {GAN-Based Style Transformation to Improve Gesture-Recognition Accuracy},
year = {2020},
issue_date = {December 2020},
volume = {4},
number = {4},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
month = dec,
articleno = {154},
numpages = {20},
}
```
