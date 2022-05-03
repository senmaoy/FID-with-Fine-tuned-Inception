# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from PIL import Image

from inception.slim import slim
from inception.slim import ops
import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow.compat.v1 as tf
from PIL import Image


import math
import os.path
import scipy.misc
from scipy import linalg

# import time
# import scipy.io as sio
# from datetime import datetime
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir',
                           '../inception_finetuned_models/flowers_valid299/model.ckpt',
                           """Path where to read model checkpoints.""")

tf.app.flags.DEFINE_string('image_folder', 

                            '/home/yesenmao/disk/dataset/flower/flower/jpg2/train',                            
                           """Path where to load the images """)
tf.app.flags.DEFINE_string('image_folder2', 
                            '/home/yesenmao/disk/dataset/flower/flower/jpg2/train',
                           """Path where to load the images """)
tf.app.flags.DEFINE_integer('num_classes', 20,      # 20 for flowers
                            """Number of classes """)
tf.app.flags.DEFINE_integer('splits', 10,
                            """Number of splits """)
tf.app.flags.DEFINE_integer('batch_size', 4, "batch size")
tf.app.flags.DEFINE_integer('gpu', 0, "The ID of GPU to use")
# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


fullpath = FLAGS.image_folder
fullpath2 = FLAGS.image_folder2
print(fullpath)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
def preprocess(img):
    # print('img', img.shape, img.max(), img.min())
    # img = Image.fromarray(img, 'RGB')
    #if len(img.size) == 2:
     #   img = np.resize(img, (img.size[0], img.size[1], 3))
    #img = scipy.misc.imresize(img, (299, 299, 3),
                              #interp='bilinear')
    img = img.resize((299, 299))
    img = np.array(img).astype(np.float32)
    # [0, 255] --> [0, 1] --> [-1, 1]
    img = img / 127.5 - 1.
    # print('img', img.shape, img.max(), img.min())
    return np.expand_dims(img, 0)


def get_inception_score(sess, images, pred_op):
    splits = FLAGS.splits
    # assert(type(images) == list)
    #assert(type(images[0]) == np.ndarray)
    assert(len(images[0].size) == 2)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    bs = FLAGS.batch_size
    preds = []
    num_examples = len(images)
    n_batches = int(math.floor(float(num_examples) / float(bs)))
    indices = list(np.arange(num_examples))
    np.random.shuffle(indices)
    for i in range(n_batches):
        inp = []
        # print('i*bs', i*bs)
        for j in range(bs):
            if (i*bs + j) == num_examples:
                break
            img = images[indices[i*bs + j]]
            # print('*****', img.shape)
            img = preprocess(img)
            inp.append(img)
        # print("%d of %d batches" % (i, n_batches))
        # inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        #  print('inp', inp.shape)
        pred = sess.run(pred_op, {'inputs:0': inp})
        preds.append(pred)
        # if i % 100 == 0:
        #     print('Batch ', i)
        #     print('inp', inp.shape, inp.max(), inp.min())
    preds = np.concatenate(preds, 0)
    scores = []
    mu = np.mean(preds, axis=0)
    sigma = np.cov(preds, rowvar=False)
    
    #print('mean:', "%.2f" % mu, 'std:', "%.2f" % sigma)
    return mu,sigma


def load_data(fullpath):
    print(fullpath)
    images = []
    for path, subdirs, files in os.walk(fullpath):
        for subdirname in subdirs:
            filepathname = os.path.join(path, subdirname)
            for path1, subdirs1, files1 in os.walk(filepathname):
                cnt = 0
                for name in files1:
                    if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                        filename = os.path.join(path1, name)
                        # print('filename', filename)
                        # print('path', path, '\nname', name)
                        # print('filename', filename)
                        
                        if os.path.isfile(filename):
                            #img = scipy.misc.imread(filename)
                            img = Image.open(filename).convert('RGB')
                            
                            images.append(img)
    print('images', len(images), images[0].size)
    return images


def inference(images, num_classes, for_training=False, restore_logits=True,
              scope=None):
    """Build Inception v3 model architecture.

    See here for reference: http://arxiv.org/abs/1512.00567

    Args:
    images: Images returned from inputs() or distorted_inputs().
    num_classes: number of classes
    for_training: If set to `True`, build the inference model for training.
      Kernels that operate differently for inference during training
      e.g. dropout, are appropriately configured.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: optional prefix string identifying the ImageNet tower.

    Returns:
    Logits. 2-D float Tensor.
    Auxiliary Logits. 2-D float Tensor of side-head. Used for training only.
    """
    # Parameters for BatchNorm.
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
    }
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
        with slim.arg_scope([slim.ops.conv2d],
                            stddev=0.1,
                            activation=tf.nn.relu,
                            batch_norm_params=batch_norm_params):
            logits, endpoints = slim.inception.inception_v3(
                images,
                dropout_keep_prob=0.8,
                num_classes=num_classes,
                is_training=for_training,
                restore_logits=restore_logits,
                scope=scope)

    # Grab the logits associated with the side head. Employed during training.
    auxiliary_logits = endpoints['mixed_8x8x2048b']

    return logits, auxiliary_logits


def main(unused_argv=None):
    """Evaluate model on Dataset for a number of steps."""
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)

        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % FLAGS.gpu):
                # Number of classes in the Dataset label set plus 1.
                # Label 0 is reserved for an (unused) background class.
                num_classes = FLAGS.num_classes + 1

                # Build a Graph that computes the logits predictions from the
                # inference model.
                inputs = tf.placeholder(
                    tf.float32, [FLAGS.batch_size, 299, 299, 3],
                    name='inputs')
                # print(inputs)

                logits, pred_op = inference(inputs, num_classes)
                
                #print(end_points)
                
                pred_op = ops.avg_pool(pred_op, pred_op.shape[1:3], padding='VALID', scope='pool')
                pred_op = ops.flatten(pred_op, scope='flatten')
               
                # Restore the moving average version of the
                # learned variables for eval.
                variable_averages = \
                    tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
                variables_to_restore = variable_averages.variables_to_restore()
                saver = tf.train.Saver(variables_to_restore)

                saver.restore(sess, FLAGS.checkpoint_dir)
                print('Restore the model from %s).' % FLAGS.checkpoint_dir)
                images = load_data(fullpath)
                images2 = load_data(fullpath2)
                m,s = get_inception_score(sess, images, pred_op)
                if not os.path.isfile('/home/yesenmao/disk/dataset/flower/IS.pkl'):
                
                    with open('/home/yesenmao/disk/dataset/flower/IS.pkl', 'wb') as f:
                        m2,s2 = get_inception_score(sess, images2, pred_op)
                        
                        pickle.dump([m2,s2], f, protocol=2)
                        print('Save to: ')
                else: 
                    with open('/home/yesenmao/disk/dataset/flower/IS.pkl', 'rb') as f:
                        x = pickle.load(f)
                        m2, s2 = x[0], x[1]                        
                        
                fid_value = calculate_frechet_distance(m, s, m2, s2)
                print(fid_value)




if __name__ == '__main__':
    tf.app.run()
