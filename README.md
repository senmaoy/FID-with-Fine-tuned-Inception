# Inception-Score-FID-on-CUB-and-OXford
As the original tensorflow code is out of data, it's a bit troublesome to run the evaluation code.

Hence, we provide a new version of code to caculate Inception Score and FID on CUB and OXford with original weight for fair of comparison.

remember to download inception_finetuned_models from StackGAN https://github.com/hanzhanggit/StackGAN-v2

requirement:

### Requirements
- python 3.8
- Tensorflow 2.7.0+cu113
- scikit-image

Usage:

1.Adapate the file path to the location of generated images and then

2.just run python inceptionscore_dir_cub.py