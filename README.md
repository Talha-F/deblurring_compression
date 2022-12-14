All modifications denoted by the suffix '_pruned' in file names.

Single image dataset loaded (the same one used throughout our paper and presentation). Link for the rest of the dataset is further below.
To run eval on our new model: 

python test_prune.py --is_eval --is_save

Deblurred image is in './result/MSSNet/'.

Inference time may not be accurate without a proper warm up (recommend at least 10 images).

If you want to run on the entire dataset (warning: large download), download dataset from below and place into dataset folder, and run:

python test_prune.py --is_eval --is_save --test_datalist ./datalist/datalist_gopro_testset.txt


### Dataset
`Train`  [[GOPRO_Large](https://seungjunnah.github.io/Datasets/gopro.html)]

`Test`  [[Google Drive](https://drive.google.com/file/d/12hV5HFTYT1CsYdbOtCr3Sw7xo1DopSeq/view?usp=sharing)] 

### Pre-trained original weights [[Google Drive](https://drive.google.com/file/d/1w8eFYRhevHDiz2TAJUcO9h9P5qbSrCQL/view?usp=sharing)]
Download and place into checkpoint folder if you want to run baseline eval. Then run:

python test.py --is_eval --is_save
  
## Acknowledgment
This repo was adapted from the official repo for MSSNet:
https://github.com/kky7/MSSNet
  
## Original Paper
```bibtex
@inproceedings{Kim2022MSSNet,
author = {Kim, Kiyeon and Lee, Seungyong and Cho, Sunghyun},
title = {MSSNet: Multi-Scale-Stage Network for Single Image Deblurring},
booktitle = {Proc. of ECCVW (AIM)},
year = {2022}
}
```
