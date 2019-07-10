#!/usr/bin/env python
# Version 1: accepts a file/folder/tf_record file where stores test images, i.e., FLAGS.infile
# Author: Qian Yu
# Date: 05/13/2019

from __future__ import print_function
import sys
# sys.path.append('../../tensorflow/models/slim/') # add slim to PYTHONPATH
import tensorflow as tf

tf.app.flags.DEFINE_string('slim_dir', '/tmp/tfmodel/','Directory where models reside.')
tf.app.flags.DEFINE_string('eval_dir', '/tmp/tfmodel/','Directory where trained nn reside.')
tf.app.flags.DEFINE_string('dataset_name', '/tmp/tfmodel/','')
tf.app.flags.DEFINE_string('dataset_split_name', '/tmp/tfmodel/','')
tf.app.flags.DEFINE_string('dataset_dir', '/tmp/tfmodel/','')
tf.app.flags.DEFINE_integer('num_classes', 3, 'The number of classes.')
tf.app.flags.DEFINE_string('infile','/Users/simcenter/Codes/SimCenter/BIM.AI/data/images/raw/roof/roof_photos/new/gabled/', 'Image file, one image per line.')
tf.app.flags.DEFINE_boolean('tfrecord',False, 'Input file is formatted as TFRecord.')
tf.app.flags.DEFINE_string('outfile','./pred_demo.txt', 'Output file for prediction probabilities.')
tf.app.flags.DEFINE_string('model_name', 'inception_v3', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string('preprocessing_name', None, 'The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_string('checkpoint_path', '/Users/simcenter/Codes/SimCenter/BIM.AI/src/training/roof/tmp/roof-traindir/all/model.ckpt-119999','The directory where the model was written to or an absolute path to a checkpoint file.')
tf.app.flags.DEFINE_integer('eval_image_size', None, 'Eval image size.')
FLAGS = tf.app.flags.FLAGS

sys.path.append(FLAGS.slim_dir)
import numpy as np
import re
import os
from os import listdir
from os.path import isfile, join

from datasets import imagenet
from nets import inception
from nets import resnet_v1
from nets import inception_utils
from nets import resnet_utils
from preprocessing import inception_preprocessing
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

model_name_to_variables = {'inception_v3':'InceptionV3','inception_v4':'InceptionV4','resnet_v1_50':'resnet_v1_50','resnet_v1_152':'resnet_v1_152'}
categories = ['flat', 'gabled', 'hipped'] # change to roof categories

preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
eval_image_size = FLAGS.eval_image_size

#

if FLAGS.tfrecord:
  fls = tf.python_io.tf_record_iterator(path=FLAGS.infile)
else:
  if os.path.isdir(FLAGS.infile):
      # print("\nIt is a directory")
      fls = [f for f in listdir(FLAGS.infile) if isfile(join(FLAGS.infile, f))]
  elif os.path.isfile(FLAGS.infile):
      # print("\nIt is a normal file")
      fls = [FLAGS.infile]
  else:
      print("test dir is not defined correctly. quit.")
      exit()

model_variables = model_name_to_variables.get(FLAGS.model_name)
if model_variables is None:
  tf.logging.error("Unknown model_name provided `%s`." % FLAGS.model_name)
  sys.exit(-1)

if FLAGS.tfrecord:
  tf.logging.warn('Image name is not available in TFRecord file.')

if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
  checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
else:
  checkpoint_path = FLAGS.checkpoint_path


image_string = tf.placeholder(tf.string) # Entry to the computational graph, e.g. image_string = tf.gfile.FastGFile(image_file).read()

#image = tf.image.decode_image(image_string, channels=3)
image = tf.image.decode_jpeg(image_string, channels=3, try_recover_truncated=True, acceptable_fraction=0.3) ## To process corrupted image files

image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)

network_fn = nets_factory.get_network_fn(FLAGS.model_name, FLAGS.num_classes, is_training=False)

if FLAGS.eval_image_size is None:
  eval_image_size = network_fn.default_image_size

processed_image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

processed_images  = tf.expand_dims(processed_image, 0) # Or tf.reshape(processed_image, (1, eval_image_size, eval_image_size, 3))

logits, _ = network_fn(processed_images)

probabilities = tf.nn.softmax(logits)

init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_variables))

sess = tf.Session()
init_fn(sess)

fout = sys.stdout
if FLAGS.outfile is not None:
  fout = open(FLAGS.outfile, 'w')
h = ['image']
h.extend(['class%s' % i for i in range(FLAGS.num_classes)])
h.append('predicted_class')
print('\t'.join(h), file=fout)
print('--------------------------------------------------')


for fl in fls:
  image_name = None

  try:
    if FLAGS.tfrecord is False:
      if os.path.isdir(FLAGS.infile):
        filename = FLAGS.infile + '/' + fl
      else:
        filename = fl
      x = tf.gfile.FastGFile(filename, 'rb').read() # You can also use x = open(fl).read()
      image_name = os.path.basename(filename)
      print('Testing image "{}"\n'.format(image_name))

    else:
      example = tf.train.Example()
      example.ParseFromString(fl)

      # Note: The key of example.features.feature depends on how you generate n
      # n
      # ntfrecord.
      x = example.features.feature['image/encoded'].bytes_list.value[0] # retrieve image string
      # import pdb
      # pdb.set_trace()
      # image_name = 'TFRecord'
      image_name = example.features.feature['filename'].bytes_list.value[0]

    probs = sess.run(probabilities, feed_dict={image_string:x})
    #np_image, network_input, probs = sess.run([image, processed_image, probabilities], feed_dict={image_string:x})

  except Exception as e:
    tf.logging.warn('Cannot process image file %s' % fl)
    continue

  # import pdb
  # pdb.set_trace()
  probs = probs[0, 0:]
  a = [image_name]
  a.extend(probs)
  ind = np.argmax(probs)
  a.append(ind)
  print('\t'.join([str(e) for e in a]), file=fout)
  # class_id need to be consistent with definitions when creating tf_records
  # modify based on what you want to display.
  print('Predictions: {}  {} {}\n'.format(probs[0], probs[1], probs[2]))
  print('The building belongs to class {} -- {}'.format(categories[ind], probs[ind]))
  print('--------------------------------------------------')

sess.close()
fout.close()
