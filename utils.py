"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
import scipy.io.wavfile
import math
import tensorflow as tf
np.set_printoptions(threshold=np.inf)
FLAGS = tf.app.flags.FLAGS

def read_data(path):
  """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def preprocess(path, scale=3):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation

  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path, is_grayscale=True)
  #print(image[0:3])
  label_ = modcrop(image, scale)

  # Must be normalized
  
  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)
  input_ = input_[0:label_.shape[0]]
  
  for x in range(0,input_.shape[0]):
    if input_[x] >= 0:
      input_[x] = math.log10(input_[x]+1) / 5
      
    else:
      input_[x] = -1*math.log10(-1*(input_[x]-1)) /5
      
    if label_[x] >= 0:
      label_[x] = math.log10(label_[x]+1) / 5      
    else:
      label_[x] = -1*math.log10(-1*(label_[x]-1)) / 5
  print("Y")
  print(input_.shape)
  #print(label_.shape)
  return input_, label_

def prepare_data(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  if FLAGS.is_train:
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.wav"))
  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
    data = glob.glob(os.path.join(data_dir, "*.wav"))

  return data

def make_data(sess, data, label):
  """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
  if FLAGS.is_train:
    savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
  else:
    savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  rate, data = scipy.io.wavfile.read(path)
  
  if is_grayscale:
    return data.astype(np.float)
  else:
    return data.astype(np.float)

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    w = image.shape
    w = w - np.mod(w, scale)
    image1 = image[0:int(w)]
  return image

def input_setup(sess, config):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  # Load data path
  if config.is_train:
    data = prepare_data(sess, dataset="Train3")
  else:
    data = prepare_data(sess, dataset="Test")

  sub_input_sequence = []
  sub_label_sequence = []
  padding = abs(config.image_size - config.label_size) / 2 # 6

  if config.is_train:
    for i in range(len(data)):
      print(i)
      input_, label_ = preprocess(data[i], config.scale)

      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        w = input_.shape
        #print(w[0])
      #for x in range(0, h-config.image_size+1, config.stride):
      for x in range(0, 1):
        for y in range(0, w[0]-config.image_size+1, config.stride):
          sub_input = input_[y:y+config.image_size] # [33 x 33]
          sub_label = label_[int(y+padding):int(y+padding+config.label_size)] # [21 x 21]

          # Make channel value
          sub_input = sub_input.reshape([config.image_size, 1])  
          sub_label = sub_label.reshape([config.label_size, 1])

          sub_input_sequence.append(sub_input)
          sub_label_sequence.append(sub_label)

  else:
    print("X")
    input_, label_ = preprocess(data[0], config.scale)
    print(input_.shape)
    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      w = input_.shape
    print("W",w)
    # Numbers of sub-images in height and width of image are needed to compute merge operation.
    nx = ny = 0 
    #for x in range(0, h-config.image_size+1, config.stride):
    for x in range(0, 1):
      nx += 1; ny = 0
      for y in range(0, w[0]-config.image_size+1, config.image_size):
        ny += 1
        sub_input = input_[y:y+config.image_size] # [33 x 33]
        sub_label = label_[int(y+padding):int(y+padding+config.label_size)] # [21 x 21]
        
        sub_input = sub_input.reshape([config.image_size, 1])  
        sub_label = sub_label.reshape([config.label_size, 1])

        sub_input_sequence.append(sub_input)
        sub_label_sequence.append(sub_label)

  """
  len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
  (sub_input_sequence[0]).shape : (33, 33, 1)
  """
  # Make list to numpy array. With this transform
  
  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
  arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]
  print(arrdata.shape)

  make_data(sess, arrdata, arrlabel)

  if not config.is_train:
    return nx, ny
    
def imsave(image, path):
  return scipy.io.wavfile.write(path,48000,image)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  
  img = np.zeros((w*h*size[1], 1))
  #print(img.shape)
  #print(images.shape)
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    #print(i)
    #print(j)
    #print(image.shape)
    img[i*h:i*h+h, :] = image
  for x in range(0,w*h*size[1]):
    if img[x]>0:
      img[x] = math.pow(4*img[x],10) -1
    else:
      img[x] = -1*(math.pow(-4*(img[x]),10) ) +1
  print(img.shape)
  return img
