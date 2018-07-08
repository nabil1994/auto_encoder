# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from keras.optimizers import adadelta
mnist = input_data.read_data_sets('MNIST_data', validation_size=0, one_hot= False)

hidden_units = 32
image_size = mnist.train.images.shape[1]
print(image_size)

# Input
inputs_ = tf.placeholder(tf.float32,[None,image_size],name='input_')
targets_ = tf.placeholder(tf.float32,[None,image_size],name='targets_')

# hidden
hidden_layer = tf.layers.dense(inputs_,image_size,activation=tf.nn.relu)

# Logits&outputs
logits_ = tf.layers.dense(hidden_layer, image_size, activation=None)
outputs_ = tf.sigmoid(logits_,name='outputs_')

# loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits_)
cost = tf.reduce_mean(loss)

# Optimizer
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.Session()
noise_factor = 0.5
epochs = 30
batch_size = 128
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range(mnist.train.num_examples//batch_size):
        batch = mnist.train.next_batch(batch_size)
        imgs = batch[0]
        # 加入噪声
        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)

        batch_cost, _ = sess.run([cost, optimizer],
                                 feed_dict={inputs_: noisy_imgs,
                                            targets_: batch[0]})

        print("Epoch: {}/{}...".format(e+1, epochs),
              "Training loss: {:.4f}".format(batch_cost))


fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
imgs = mnist.test.images[10:20]
# 加入噪声
noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
noisy_imgs = np.clip(noisy_imgs, 0.0, 1.0)

reconstructed = sess.run(outputs_, feed_dict={inputs_: imgs})

for images, row in zip([noisy_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)