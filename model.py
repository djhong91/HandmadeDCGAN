from utils import *
import os
from glob import glob
import tensorflow as tf
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import time

from ops import *
from utils import *

class DCGAN:
    def __init__(self, sess, input_height=64, input_width=64, output_height=64, output_width=64,
                 batch_size=128, z_dim=100, dataset_dir='faceimage', checkpoint_dir='checkpoint',
                 input_fname_pattern='*.jpg', sample_dir='samples',
                 gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, c_dim=3):
        """
        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          z_dim: (optional) Dimension of dim for Z. [100], 잡음의 feature 수
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess

        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim
        self.sample_num = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.dataset_dir = 'datasets/' + dataset_dir
        self.checkpoint_dir = checkpoint_dir
        self.input_fname_pattern = input_fname_pattern
        self.sample_dir = sample_dir

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        # data는 입력 파일 경로들이다. 아직 읽어들인 이미지가 아님
        data_path = os.path.join(self.dataset_dir, self.input_fname_pattern)
        self.data = glob(data_path)

        if len(self.data) == 0:
            raise Exception("[!] No data found in '" + data_path + "'")
        np.random.shuffle(self.data)
        imreadImg = imread(self.data[0])
        if len(imreadImg.shape) >= 3:
            self.c_dim = imread(self.data[0]).shape[-1]
        else:
            self.c_dim = 1

        self.grayscale = (self.c_dim == 1)

        if len(self.data) < self.batch_size:
            raise Exception("[!] batch_size가 전체 데이터 크기보다 큽니다.")

        self._build_model()

    def _build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
        inputs = self.inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(inputs, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        self.sampler = self.sampler(self.z)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.D_logits, labels=tf.ones_like(self.D)))

        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.D_logits_, labels=tf.zeros_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, args):
        d_optim = tf.train.AdamOptimizer(learning_rate=args.learning_rate, beta1=args.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate=args.learning_rate, beta1=args.beta1).minimize(self.g_loss, var_list=self.g_vars)

        self.sess.run(tf.global_variables_initializer())

        counter = 1
        start_time = time.time()
        is_loaded, checkpoint_counter = self.load(self.checkpoint_dir)

        if is_loaded:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        sample_z = np.random.uniform(-1, 1, size=(args.generate_test_images, self.z_dim))
        sample_files = self.data[0:args.generate_test_images]
        sample = [get_image(sample_file, grayscale=self.grayscale) for sample_file in sample_files]

        if self.grayscale:
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)

        for epoch in range(args.epoch):
            # carpedm20의 코드에는 이 부분에 데이터를 읽는 부분이 있지만 init에서 이미 읽었으므로 중복코드이다.
            np.random.shuffle(self.data)
            batch_idxs = len(self.data) // args.batch_size

            for idx in range(0, int(batch_idxs)):
                batch_files = self.data[idx*args.batch_size:(idx+1)*args.batch_size]
                batch = [get_image(batch_file, grayscale=self.grayscale) for batch_file in batch_files]

                if self.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [args.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                self.sess.run(d_optim, feed_dict={self.inputs: batch_images, self.z: batch_z})

                # Update G network
                self.sess.run(g_optim, feed_dict={self.z: batch_z})

                # d_loss가 확실히 0이 되지 않도록 G의 업데이트를 두 번할 것을 carpedm20이 제안함
                self.sess.run(g_optim, feed_dict={self.z: batch_z})

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})

                print("Epoch: [%3d/%3d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, args.epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))

            counter += 1

            if np.mod(counter, args.save_freq) == 1:
                samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],
                                                        feed_dict={self.z: sample_z, self.inputs: sample_inputs})
                save_images(samples, image_manifold_size(samples.shape[0]),
                            './{}/train_{:02d}.png'.format(args.sample_dir, epoch))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
            if np.mod(counter, args.save_freq) == 2:
                self.save(args.checkpoint_dir, counter)

    def test(self, args):
        counter = 1

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))    # 64
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))  # 128
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))  # 256
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))  # 512
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width                            # 64
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)         # 32
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)       # 16
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)       # 8
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)     # 4

            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim*8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim ], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def sampler(self, z):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width                            # 64
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)         # 32
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)       # 16
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)       # 8
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)     # 4

            h0 = tf.reshape(linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'), [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim ], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')
            return tf.nn.tanh(h4)

    def save(self, checkpoint_dir, step):
        model_name = "handmadeDCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_dir, self.input_height, self.input_width)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading Checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_dir, self.input_height, self.input_width)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

            counter = int(next(re.finditer("(\d+)(?!.*Wd)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [!] Failed to find a checkpoint")
            return False, 0