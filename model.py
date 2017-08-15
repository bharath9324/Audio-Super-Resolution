from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.inf)

class SRCNN(object):

  def __init__(self, 
               sess, 
               audio_size=500,
               label_size=500, 
               batch_size=128,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.audio_size = audio_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    self.audio = tf.placeholder(tf.float32, [None, self.audio_size, self.c_dim], name='audio')
    self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.c_dim], name='labels')
    
    self.weights = {
      'w1': tf.Variable(tf.random_normal([5, 1, 32], stddev=1e-4), name='w1'),
      'w2': tf.Variable(tf.random_normal([2, 32, 16], stddev=1e-4), name='w2'),
      'W_fc2': tf.Variable(tf.random_normal([16*256, 128], stddev=1e-4), name='w3')
      #'w3': tf.Variable(tf.random_normal([3, 64, 1], stddev=1e-4), name='w3')
    }
    self.biases = {
      'b1': tf.Variable(tf.zeros([32]), name='b1'),
      'b2': tf.Variable(tf.zeros([16]), name='b2'),
      'b_fc2': tf.Variable(tf.zeros([128]), name='b3')
         		  
  

  # Map the 1024 features to 10 classes, one for each digit
  



    }

    self.pred= self.model()

    # Loss function (MSE)
    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

    self.saver = tf.train.Saver()

  def train(self, config):
    if config.is_train:
      input_setup(self.sess, config)
    else:
      nx, ny = input_setup(self.sess, config)

    if config.is_train:     
      print("Here")
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
      print("Done")
    else:
      data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

    train_data, train_label = read_data(data_dir)
    #print(train_data)
    # Stochastic gradient descent with the standard backpropagation
    self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)

    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if config.is_train:
      print("Training...")

      for ep in range(config.epoch):
        # Run by batch audio

        batch_idxs = len(train_data) // config.batch_size
        for idx in range(0, batch_idxs):
          batch_audio = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]
          #print("Z")
          #print(batch_audio)
          #print(batch_labels)
          counter += 1
          _, err,pred = self.sess.run([self.train_op, self.loss, self.weights['w1']], feed_dict={self.audio: batch_audio, self.labels: batch_labels})
          #print(pred)
          if counter % 10 == 0:
            print("Weight %f", pred[1][0][0])
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err))

          if counter % 500 == 0:
            self.save(config.checkpoint_dir, counter)

    else:
      print("Testing...")
      print(train_data.shape)
      result = self.pred.eval({self.audio: train_data, self.labels: train_label})
      print(result.shape)
      result = merge(result, [nx, ny])
      result = result.squeeze()
      audio_path = os.path.join(os.getcwd(), config.sample_dir)
      audio_path = os.path.join(audio_path, "test_sound.wav")
      #print(result)
      print("YY")
      #print(train_label.shape)
      imsave(result, audio_path)

  def model(self):
    conv1 = tf.nn.relu(tf.nn.conv1d(self.audio, self.weights['w1'], stride=1, padding='SAME') + self.biases['b1'])
    #conv2 = tf.nn.relu(tf.nn.conv1d(conv1, self.weights['w2'], stride=1, padding='SAME') + self.biases['b2'])
    conv2 = tf.nn.conv1d(conv1, self.weights['w2'], stride=1, padding='SAME') + self.biases['b2']
    #conv3 = tf.nn.conv1d(conv2, self.weights['w3'], stride=1, padding='SAME') + self.biases['b3']
    conv2 = tf.reshape(conv2, [-1, 16*256])
    #keep_prob = tf.placeholder(tf.float32)
  	
    h_fc2 = tf.matmul(conv2, self.weights['W_fc2']) + self.biases['b_fc2']
    #h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    return h_fc2

  def save(self, checkpoint_dir, step):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
