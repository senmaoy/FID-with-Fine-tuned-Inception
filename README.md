# Inception-Score-FID-on-CUB-and-OXford
As the original tensorflow code is out of data, it's a bit troublesome to run the evaluation code.

Hence, we provide a new version of code to caculate Inception Score and FID on CUB and OXford with original weight for fair of comparison.

### Note that the code will produce higher FID score than using weight without finetuning
For example, RAT-GAN scores 13.91 with this code. But using inception weight without finetuning , RAT-GAN scores 10.21.

This is because stackGAN fine-tuned the inception weights on CUB, which makes FID to be more aware of the difference of bird images. According to my experience, when the FID grows smaller, this code better distinguishes the image quality.

### First download inception_finetuned_models from [StackGAN](https://drive.google.com/file/d/0B3y_msrWZaXLMzNMNWhWdW0zVWs/view?resourcekey=0-gBxxw4fU6ikmNtkfFSQALw)


### Requirements
- python 3.8
- Tensorflow 2.7.0+cu113
- scikit-image
- pillow
### Usage:

1.Change the file path to the location of generated images

2.Just run python inceptionscore_dir_cub.py

**Reference**
- [Recurrent-Affine-Transformation-for-Text-to-image-Synthesis](https://arxiv.org/abs/2204.10482) [[code]](https://github.com/senmaoy/Recurrent-Affine-Transformation-for-Text-to-image-Synthesis.git)
