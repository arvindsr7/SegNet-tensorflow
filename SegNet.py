import json
import os

import tensorflow as tf
import numpy as np
import random
from layers_object import conv_layer, up_sampling, max_pool, initialization, \
    variable_with_weight_decay
from evaluation_object import cal_loss, cal_loss_gta5, cal_loss_gta5_v2, weighted_loss, weighted_loss_v2, normal_loss, per_class_acc, get_hist, print_hist_summary, train_op, MAX_VOTE, var_calculate, var_calculate_gta5
from inputs_object import get_filename_list, dataset_inputs, dataset_gta5_inputs, get_all_test_data, get_all_test_data_gta5
from drawings_object import draw_plots_bayes, draw_plots_bayes_external, writeImage
from scipy import misc
import time
import matplotlib.pyplot as plt
import cv2
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

class SegNet:
    def __init__(self, conf_file="config.json"):
        with open(conf_file) as f:
            self.config = json.load(f)

        self.num_classes = self.config["NUM_CLASSES"]
        self.use_vgg = self.config["USE_VGG"]

        if self.use_vgg is False:
            self.vgg_param_dict = None
            print("No VGG path in config, so learning from scratch")
        else:
            self.vgg16_npy_path = self.config["VGG_FILE"]
            self.vgg_param_dict = np.load(self.vgg16_npy_path, allow_pickle=True, encoding='latin1').item()
            print("VGG parameter loaded")

        self.train_file = self.config["TRAIN_FILE"]
        self.val_file = self.config["VAL_FILE"]
        self.test_file = self.config["TEST_FILE"]
        self.img_prefix = self.config["IMG_PREFIX"]
        self.label_prefix = self.config["LABEL_PREFIX"]
        self.bayes = self.config["BAYES"]
        self.opt = self.config["OPT"]
        self.saved_dir = self.config["SAVE_MODEL_DIR"]
        self.input_w = self.config["INPUT_WIDTH"]
        self.input_h = self.config["INPUT_HEIGHT"]
        self.input_c = self.config["INPUT_CHANNELS"]
        self.tb_logs = self.config["TB_LOGS"]
        self.batch_size = self.config["BATCH_SIZE"]

        self.train_loss, self.train_accuracy = [], []
        self.train_accuracy_iou, self.train_accuracy_sidewalk = [], []
        self.val_loss, self.val_acc = [], []
        self.val_acc_sidewalk, self.val_acc_iou = [], []

        self.model_version = 0  # used for saving the model
        self.saver = None
        self.images_tr, self.labels_tr = None, None
        self.images_val, self.labels_val = None, None

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            self.batch_size_pl = tf.placeholder(tf.int64, shape=[], name="batch_size")
            self.is_training_pl = tf.placeholder(tf.bool, name="is_training")
            self.with_dropout_pl = tf.placeholder(tf.bool, name="with_dropout")
            self.keep_prob_pl = tf.placeholder(tf.float32, shape=None, name="keep_rate")
            self.inputs_pl = tf.placeholder(tf.float32, [None, self.input_h, self.input_w, self.input_c])
            self.labels_pl = tf.placeholder(tf.int64, [None, self.input_h, self.input_w, 1])

            # Before enter the images into the architecture, we need to do Local Contrast Normalization
            # But it seems a bit complicated, so we use Local Response Normalization which implement in Tensorflow
            # Reference page:https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
            self.norm1 = tf.nn.lrn(self.inputs_pl, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')
            # first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
            self.conv1_1 = conv_layer(self.norm1, "conv1_1", [3, 3, 3, 64], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv1_2 = conv_layer(self.conv1_1, "conv1_2", [3, 3, 64, 64], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.pool1, self.pool1_index, self.shape_1 = max_pool(self.conv1_2, 'pool1')

            # Second box of convolution layer(4)
            self.conv2_1 = conv_layer(self.pool1, "conv2_1", [3, 3, 64, 128], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv2_2 = conv_layer(self.conv2_1, "conv2_2", [3, 3, 128, 128], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.pool2, self.pool2_index, self.shape_2 = max_pool(self.conv2_2, 'pool2')

            # Third box of convolution layer(7)
            self.conv3_1 = conv_layer(self.pool2, "conv3_1", [3, 3, 128, 256], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv3_2 = conv_layer(self.conv3_1, "conv3_2", [3, 3, 256, 256], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv3_3 = conv_layer(self.conv3_2, "conv3_3", [3, 3, 256, 256], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.pool3, self.pool3_index, self.shape_3 = max_pool(self.conv3_3, 'pool3')

            # Fourth box of convolution layer(10)
            if self.bayes:
                self.dropout1 = tf.layers.dropout(self.pool3, rate=(1 - self.keep_prob_pl),
                                                  training=self.with_dropout_pl, name="dropout1")
                self.conv4_1 = conv_layer(self.dropout1, "conv4_1", [3, 3, 256, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
            else:
                self.conv4_1 = conv_layer(self.pool3, "conv4_1", [3, 3, 256, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
            self.conv4_2 = conv_layer(self.conv4_1, "conv4_2", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv4_3 = conv_layer(self.conv4_2, "conv4_3", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.pool4, self.pool4_index, self.shape_4 = max_pool(self.conv4_3, 'pool4')

            # Fifth box of convolution layers(13)
            if self.bayes:
                self.dropout2 = tf.layers.dropout(self.pool4, rate=(1 - self.keep_prob_pl),
                                                  training=self.with_dropout_pl, name="dropout2")
                self.conv5_1 = conv_layer(self.dropout2, "conv5_1", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
            else:
                self.conv5_1 = conv_layer(self.pool4, "conv5_1", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
            self.conv5_2 = conv_layer(self.conv5_1, "conv5_2", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv5_3 = conv_layer(self.conv5_2, "conv5_3", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.pool5, self.pool5_index, self.shape_5 = max_pool(self.conv5_3, 'pool5')

            # ---------------------So Now the encoder process has been Finished--------------------------------------#
            # ------------------Then Let's start Decoder Process-----------------------------------------------------#

            # First box of deconvolution layers(3)
            if self.bayes:
                self.dropout3 = tf.layers.dropout(self.pool5, rate=(1 - self.keep_prob_pl),
                                                  training=self.with_dropout_pl, name="dropout3")
                self.deconv5_1 = up_sampling(self.dropout3, self.pool5_index, self.shape_5, self.batch_size_pl,
                                             name="unpool_5")
            else:
                self.deconv5_1 = up_sampling(self.pool5, self.pool5_index, self.shape_5, self.batch_size_pl,
                                             name="unpool_5")
            self.deconv5_2 = conv_layer(self.deconv5_1, "deconv5_2", [3, 3, 512, 512], self.is_training_pl)
            self.deconv5_3 = conv_layer(self.deconv5_2, "deconv5_3", [3, 3, 512, 512], self.is_training_pl)
            self.deconv5_4 = conv_layer(self.deconv5_3, "deconv5_4", [3, 3, 512, 512], self.is_training_pl)
            # Second box of deconvolution layers(6)
            if self.bayes:
                self.dropout4 = tf.layers.dropout(self.deconv5_4, rate=(1 - self.keep_prob_pl),
                                                  training=self.with_dropout_pl, name="dropout4")
                self.deconv4_1 = up_sampling(self.dropout4, self.pool4_index, self.shape_4, self.batch_size_pl,
                                             name="unpool_4")
            else:
                self.deconv4_1 = up_sampling(self.deconv5_4, self.pool4_index, self.shape_4, self.batch_size_pl,
                                             name="unpool_4")
            self.deconv4_2 = conv_layer(self.deconv4_1, "deconv4_2", [3, 3, 512, 512], self.is_training_pl)
            self.deconv4_3 = conv_layer(self.deconv4_2, "deconv4_3", [3, 3, 512, 512], self.is_training_pl)
            self.deconv4_4 = conv_layer(self.deconv4_3, "deconv4_4", [3, 3, 512, 256], self.is_training_pl)
            # Third box of deconvolution layers(9)
            if self.bayes:
                self.dropout5 = tf.layers.dropout(self.deconv4_4, rate=(1 - self.keep_prob_pl),
                                                  training=self.with_dropout_pl, name="dropout5")
                self.deconv3_1 = up_sampling(self.dropout5, self.pool3_index, self.shape_3, self.batch_size_pl,
                                             name="unpool_3")
            else:
                self.deconv3_1 = up_sampling(self.deconv4_4, self.pool3_index, self.shape_3, self.batch_size_pl,
                                             name="unpool_3")
            self.deconv3_2 = conv_layer(self.deconv3_1, "deconv3_2", [3, 3, 256, 256], self.is_training_pl)
            self.deconv3_3 = conv_layer(self.deconv3_2, "deconv3_3", [3, 3, 256, 256], self.is_training_pl)
            self.deconv3_4 = conv_layer(self.deconv3_3, "deconv3_4", [3, 3, 256, 128], self.is_training_pl)
            # Fourth box of deconvolution layers(11)
            if self.bayes:
                self.dropout6 = tf.layers.dropout(self.deconv3_4, rate=(1 - self.keep_prob_pl),
                                                  training=self.with_dropout_pl, name="dropout6")
                self.deconv2_1 = up_sampling(self.dropout6, self.pool2_index, self.shape_2, self.batch_size_pl,
                                             name="unpool_2")
            else:
                self.deconv2_1 = up_sampling(self.deconv3_4, self.pool2_index, self.shape_2, self.batch_size_pl,
                                             name="unpool_2")
            self.deconv2_2 = conv_layer(self.deconv2_1, "deconv2_2", [3, 3, 128, 128], self.is_training_pl)
            self.deconv2_3 = conv_layer(self.deconv2_2, "deconv2_3", [3, 3, 128, 64], self.is_training_pl)
            # Fifth box of deconvolution layers(13)
            self.deconv1_1 = up_sampling(self.deconv2_3, self.pool1_index, self.shape_1, self.batch_size_pl,
                                         name="unpool_1")
            self.deconv1_2 = conv_layer(self.deconv1_1, "deconv1_2", [3, 3, 64, 64], self.is_training_pl)
            self.deconv1_3 = conv_layer(self.deconv1_2, "deconv1_3", [3, 3, 64, 64], self.is_training_pl)

            with tf.variable_scope('conv_classifier') as scope:
                self.kernel = variable_with_weight_decay('weights', initializer=initialization(1, 64),
                                                         shape=[1, 1, 64, self.num_classes], wd=False)
                self.conv = tf.nn.conv2d(self.deconv1_3, self.kernel, [1, 1, 1, 1], padding='SAME')
                self.biases = variable_with_weight_decay('biases', tf.constant_initializer(0.0),
                                                         shape=[self.num_classes], wd=False)
                self.logits = tf.nn.bias_add(self.conv, self.biases, name=scope.name)

    def restore(self, res_file):
        from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        print_tensors_in_checkpoint_file(res_file, None, False)
        with self.graph.as_default():
            for v in tf.global_variables():
                print(v)
            if self.saver is None:
                self.saver = tf.train.Saver(tf.global_variables())
            self.saver.restore(self.sess, res_file)

           
    def train_gta5_v2(self, max_steps=30001, batch_size=3):
        """
        Modifications from train_gta5:
            - only classes relevant to sidewalks used
            - weight for sidewalk increased 
            - new accuracy measures that amplified detection used 
        """
        # For train the bayes, the FLAG_OPT SHOULD BE SGD, BUT FOR TRAIN THE NORMAL SEGNET,
        # THE FLAG_OPT SHOULD BE ADAM!!!

        image_filename, label_filename = get_filename_list(self.train_file, self.config)
        val_image_filename, val_label_filename = get_filename_list(self.val_file, self.config)
        
        with self.graph.as_default():
            if self.images_tr is None:
                self.images_tr, self.labels_tr = dataset_gta5_inputs(image_filename, label_filename, batch_size, self.config)
                self.images_val, self.labels_val = dataset_gta5_inputs(val_image_filename, val_label_filename, batch_size,
                                                                  self.config)

            loss, accuracy, accuracy_sidewalk, accuracy_iou, prediction = cal_loss_gta5_v2(logits=self.logits, labels=self.labels_pl, number_class=self.num_classes)
            train, global_step = train_op(total_loss=loss, opt=self.opt)

            summary_op = tf.summary.merge_all()

            with self.sess.as_default():
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                # The queue runners basic reference:
                # https://www.tensorflow.org/versions/r0.12/how_tos/threading_and_queues
                train_writer = tf.summary.FileWriter(self.tb_logs, self.sess.graph)
                for step in range(max_steps):
                    
                    image_batch, label_batch = self.sess.run([self.images_tr, self.labels_tr])
                    
                    feed_dict = {self.inputs_pl: image_batch,
                                 self.labels_pl: label_batch,
                                 self.is_training_pl: True,
                                 self.keep_prob_pl: 0.5,
                                 self.with_dropout_pl: True,
                                 self.batch_size_pl: batch_size}

                    _, _loss, _accuracy, _accuracy_sidewalk, _accuracy_iou, summary = self.sess.run([train, loss, accuracy, accuracy_sidewalk, accuracy_iou, summary_op], feed_dict=feed_dict)
                    self.train_loss.append(_loss)
                    self.train_accuracy.append(_accuracy)
                    self.train_accuracy_sidewalk.append(_accuracy_sidewalk)
                    self.train_accuracy_iou.append(_accuracy_iou)

                    print("Iteration {}/{}: Train Loss{:6.3f}, Train Accu Total {:6.3f}, Train Accu Sidewalk {:6.3f}, Train Accu IOU {:6.3f}".format(step, max_steps, self.train_loss[-1],self.train_accuracy[-1], self.train_accuracy_sidewalk[-1], self.train_accuracy_iou[-1]))

                    if step % 100 == 0:
                        conv_classifier = self.sess.run(self.logits, feed_dict=feed_dict)
                        print('per_class accuracy by logits in training time',
                              per_class_acc(conv_classifier, label_batch, self.num_classes))
                        # per_class_acc is a function from utils
                        train_writer.add_summary(summary, step)

                    if step % 1000 == 0:
                        print("start validating.......")
                        _val_loss = []
                        _val_acc = []
                        _val_acc_sidewalk = []
                        _val_acc_iou = []
                        hist = np.zeros((self.num_classes, self.num_classes))
                        for test_step in range(int(20)):
                            fetches_valid = [loss, accuracy, accuracy_sidewalk, accuracy_iou, self.logits]
                            image_batch_val, label_batch_val = self.sess.run([self.images_val, self.labels_val])
                            feed_dict_valid = {self.inputs_pl: image_batch_val,
                                               self.labels_pl: label_batch_val,
                                               self.is_training_pl: True,
                                               self.keep_prob_pl: 1.0,
                                               self.with_dropout_pl: False,
                                               self.batch_size_pl: batch_size}
                            # since we still using mini-batch, so in the batch norm we set phase_train to be
                            # true, and because we didin't run the trainop process, so it will not update
                            # the weight!
                            _loss, _acc, _acc_sidewalk, _acc_iou, _val_pred = self.sess.run(fetches_valid, feed_dict_valid)
                            _val_loss.append(_loss)
                            _val_acc.append(_acc)
                            _val_acc_sidewalk.append(_acc_sidewalk)
                            _val_acc_iou.append(_acc_iou)
                            hist += get_hist(_val_pred, label_batch_val)

                        print_hist_summary(hist)

                        self.val_loss.append(np.mean(_val_loss))
                        self.val_acc.append(np.mean(_val_acc))
                        self.val_acc_sidewalk.append(np.mean(_val_acc_sidewalk))
                        self.val_acc_iou.append(np.mean(_val_acc_iou))

                        print( \
                            "Iteration {}/{}: Train Loss {:6.3f}, Train Acc {:6.3f}, Val Loss {:6.3f}, Val Acc {:6.3f}  \
                            Val Acc Sidewalk{:6.3f}  Val Acc IOU {:6.3f}".format(step, max_steps, self.train_loss[-1], \
                            self.train_accuracy[-1], self.val_loss[-1], self.val_acc[-1], self.val_acc_sidewalk[-1], \
                            self.val_acc_iou[-1]))
                    
                    if step % 5000 == 0:
                        print("Save checkpoint at step =", step)
                        self.save()

                coord.request_stop()
                coord.join(threads)
                
    def train_gta5(self, max_steps=30001, batch_size=3):
        # For train the bayes, the FLAG_OPT SHOULD BE SGD, BUT FOR TRAIN THE NORMAL SEGNET,
        # THE FLAG_OPT SHOULD BE ADAM!!!

        image_filename, label_filename = get_filename_list(self.train_file, self.config)
        val_image_filename, val_label_filename = get_filename_list(self.val_file, self.config)
        
        with self.graph.as_default():
            if self.images_tr is None:
                self.images_tr, self.labels_tr = dataset_gta5_inputs(image_filename, label_filename, batch_size, self.config)
                self.images_val, self.labels_val = dataset_gta5_inputs(val_image_filename, val_label_filename, batch_size,
                                                                  self.config)

            loss, accuracy, prediction = cal_loss_gta5(logits=self.logits, labels=self.labels_pl,
                                                     number_class=self.num_classes)
            train, global_step = train_op(total_loss=loss, opt=self.opt)

            summary_op = tf.summary.merge_all()

            with self.sess.as_default():
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                # The queue runners basic reference:
                # https://www.tensorflow.org/versions/r0.12/how_tos/threading_and_queues
                train_writer = tf.summary.FileWriter(self.tb_logs, self.sess.graph)
                for step in range(max_steps):
                    
                    image_batch, label_batch = self.sess.run([self.images_tr, self.labels_tr])
                    
                    feed_dict = {self.inputs_pl: image_batch,
                                 self.labels_pl: label_batch,
                                 self.is_training_pl: True,
                                 self.keep_prob_pl: 0.5,
                                 self.with_dropout_pl: True,
                                 self.batch_size_pl: batch_size}

                    _, _loss, _accuracy, summary = self.sess.run([train, loss, accuracy, summary_op],
                                                                 feed_dict=feed_dict)
                    self.train_loss.append(_loss)
                    self.train_accuracy.append(_accuracy)
                    print("Iteration {}/{}: Train Loss{:6.3f}, Train Accu {:6.3f}".format(step, max_steps, self.train_loss[-1],
                                                                                       self.train_accuracy[-1]))

                    if step % 100 == 0:
                        conv_classifier = self.sess.run(self.logits, feed_dict=feed_dict)
                        print('per_class accuracy by logits in training time',
                              per_class_acc(conv_classifier, label_batch, self.num_classes))
                        # per_class_acc is a function from utils
                        train_writer.add_summary(summary, step)

                    if step % 1000 == 0:
                        print("start validating.......")
                        _val_loss = []
                        _val_acc = []
                        hist = np.zeros((self.num_classes, self.num_classes))
                        for test_step in range(int(20)):
                            fetches_valid = [loss, accuracy, self.logits]
                            image_batch_val, label_batch_val = self.sess.run([self.images_val, self.labels_val])
                            feed_dict_valid = {self.inputs_pl: image_batch_val,
                                               self.labels_pl: label_batch_val,
                                               self.is_training_pl: True,
                                               self.keep_prob_pl: 1.0,
                                               self.with_dropout_pl: False,
                                               self.batch_size_pl: batch_size}
                            # since we still using mini-batch, so in the batch norm we set phase_train to be
                            # true, and because we didin't run the trainop process, so it will not update
                            # the weight!
                            _loss, _acc, _val_pred = self.sess.run(fetches_valid, feed_dict_valid)
                            _val_loss.append(_loss)
                            _val_acc.append(_acc)
                            hist += get_hist(_val_pred, label_batch_val)

                        print_hist_summary(hist)

                        self.val_loss.append(np.mean(_val_loss))
                        self.val_acc.append(np.mean(_val_acc))

                        print(
                            "Iteration {}/{}: Train Loss {:6.3f}, Train Acc {:6.3f}, Val Loss {:6.3f}, Val Acc {:6.3f}".format(
                                step, max_steps, self.train_loss[-1], self.train_accuracy[-1], self.val_loss[-1],
                                self.val_acc[-1]))

                coord.request_stop()
                coord.join(threads)
                
    def train(self, max_steps=30001, batch_size=3):
        # For train the bayes, the FLAG_OPT SHOULD BE SGD, BUT FOR TRAIN THE NORMAL SEGNET,
        # THE FLAG_OPT SHOULD BE ADAM!!!

        image_filename, label_filename = get_filename_list(self.train_file, self.config)
        val_image_filename, val_label_filename = get_filename_list(self.val_file, self.config)
        
        with self.graph.as_default():
            if self.images_tr is None:
                self.images_tr, self.labels_tr = dataset_inputs(image_filename, label_filename, batch_size, self.config)
                self.images_val, self.labels_val = dataset_inputs(val_image_filename, val_label_filename, batch_size,
                                                                  self.config)

            loss, accuracy, prediction = cal_loss(logits=self.logits, labels=self.labels_pl,
                                                     number_class=self.num_classes)
            train, global_step = train_op(total_loss=loss, opt=self.opt)

            summary_op = tf.summary.merge_all()

            with self.sess.as_default():
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                # The queue runners basic reference:
                # https://www.tensorflow.org/versions/r0.12/how_tos/threading_and_queues
                train_writer = tf.summary.FileWriter(self.tb_logs, self.sess.graph)
                for step in range(max_steps):
                    image_batch, label_batch = self.sess.run([self.images_tr, self.labels_tr])
                    print(np.unique(label_batch))
                    feed_dict = {self.inputs_pl: image_batch,
                                 self.labels_pl: label_batch,
                                 self.is_training_pl: True,
                                 self.keep_prob_pl: 0.5,
                                 self.with_dropout_pl: True,
                                 self.batch_size_pl: batch_size}

                    _, _loss, _accuracy, summary = self.sess.run([train, loss, accuracy, summary_op],
                                                                 feed_dict=feed_dict)
                    self.train_loss.append(_loss)
                    self.train_accuracy.append(_accuracy)
                    print("Iteration {}: Train Loss{:6.3f}, Train Accu {:6.3f}".format(step, self.train_loss[-1],
                                                                                       self.train_accuracy[-1]))

                    if step % 100 == 0:
                        conv_classifier = self.sess.run(self.logits, feed_dict=feed_dict)
                        print('per_class accuracy by logits in training time',
                              per_class_acc(conv_classifier, label_batch, self.num_classes))
                        # per_class_acc is a function from utils
                        train_writer.add_summary(summary, step)

                    if step % 1000 == 0:
                        print("start validating.......")
                        _val_loss = []
                        _val_acc = []
                        hist = np.zeros((self.num_classes, self.num_classes))
                        for test_step in range(int(20)):
                            fetches_valid = [loss, accuracy, self.logits]
                            image_batch_val, label_batch_val = self.sess.run([self.images_val, self.labels_val])
                            feed_dict_valid = {self.inputs_pl: image_batch_val,
                                               self.labels_pl: label_batch_val,
                                               self.is_training_pl: True,
                                               self.keep_prob_pl: 1.0,
                                               self.with_dropout_pl: False,
                                               self.batch_size_pl: batch_size}
                            # since we still using mini-batch, so in the batch norm we set phase_train to be
                            # true, and because we didin't run the trainop process, so it will not update
                            # the weight!
                            _loss, _acc, _val_pred = self.sess.run(fetches_valid, feed_dict_valid)
                            _val_loss.append(_loss)
                            _val_acc.append(_acc)
                            hist += get_hist(_val_pred, label_batch_val)

                        print_hist_summary(hist)

                        self.val_loss.append(np.mean(_val_loss))
                        self.val_acc.append(np.mean(_val_acc))

                        print(
                            "Iteration {}: Train Loss {:6.3f}, Train Acc {:6.3f}, Val Loss {:6.3f}, Val Acc {:6.3f}".format(
                                step, self.train_loss[-1], self.train_accuracy[-1], self.val_loss[-1],
                                self.val_acc[-1]))

                coord.request_stop()
                coord.join(threads)

                
    def visual_results(self, dataset_type = "TEST", images_index = 3, FLAG_MAX_VOTE = False):
        
        image_w = self.config["INPUT_WIDTH"]
        image_h = self.config["INPUT_HEIGHT"]
        image_c = self.config["INPUT_CHANNELS"]
        train_dir = self.config["SAVE_MODEL_DIR"]
        FLAG_BAYES = self.config["BAYES"]

        with self.sess as sess:
            
            # Restore saved session
            saver = tf.train.Saver()
            saver.restore(sess, train_dir)
            
            _, _, prediction = cal_loss(self.logits, self.labels_pl, 2)
            prob = tf.nn.softmax(self.logits,dim = -1)
            
            if (dataset_type=='TRAIN'):
                test_type_path = self.config["TRAIN_FILE"]
                if type(images_index) == list:
                    indexes = images_index
                else:
                    indexes = random.sample(range(367),images_index)
                #indexes = [0,75,150,225,300]
            elif (dataset_type=='VAL'):
                test_type_path = self.config["VAL_FILE"]
                if type(images_index) == list:
                    indexes = images_index
                else:
                    indexes = random.sample(range(101),images_index)
                #indexes = [0,25,50,75,100]
            elif (dataset_type=='TEST'):
                test_type_path = self.config["TEST_FILE"]
                if type(images_index) == list:
                    indexes = images_index
                else:
                    indexes = random.sample(range(233),images_index)
                #indexes = [0,50,100,150,200]

            # Load images
            image_filename,label_filename = get_filename_list(test_type_path, self.config)
            images, labels = get_all_test_data(image_filename,label_filename)

            # Keep images subset of length images_index
            images = [images[i] for i in indexes]
            labels = [labels[i] for i in indexes]
            
            num_sample_generate = 30
            pred_tot = []
            var_tot = []
            
            for image_batch, label_batch in zip(images,labels):
                
                image_batch = np.reshape(image_batch,[1,image_h,image_w,image_c])
                label_batch = np.reshape(label_batch,[1,image_h,image_w,1])
                
                if FLAG_BAYES is False:
                    fetches = [prediction]
                    feed_dict = {self.inputs_pl: image_batch, 
                                 self.labels_pl: label_batch, 
                                 self.is_training_pl: False, 
                                 self.keep_prob_pl: 0.5,
                                 self.batch_size_pl: 1}
                    pred = sess.run(fetches = fetches, feed_dict = feed_dict)
                    pred = np.reshape(pred,[image_h,image_w])
                    var_one = []
                else:
                    feed_dict = {self.inputs_pl: image_batch, 
                                 self.labels_pl: label_batch, 
                                 self.is_training_pl: False, 
                                 self.keep_prob_pl: 0.5,
                                 self.with_dropout_pl: True,
                                 self.batch_size_pl: 1}
                    prob_iter_tot = []
                    pred_iter_tot = []
                    for iter_step in range(num_sample_generate):
                        prob_iter_step = sess.run(fetches = [prob], feed_dict = feed_dict) 
                        prob_iter_tot.append(prob_iter_step)
                        pred_iter_tot.append(np.reshape(np.argmax(prob_iter_step,axis = -1),[-1]))
                        
                    if FLAG_MAX_VOTE is True:
                        prob_variance,pred = MAX_VOTE(pred_iter_tot,prob_iter_tot,self.config["NUM_CLASSES"])
                        #acc_per = np.mean(np.equal(pred,np.reshape(label_batch,[-1])))
                        var_one = var_calculate(pred,prob_variance)
                        pred = np.reshape(pred,[image_h,image_w])
                    else:
                        prob_mean = np.nanmean(prob_iter_tot,axis = 0)
                        prob_variance = np.var(prob_iter_tot, axis = 0)
                        pred = np.reshape(np.argmax(prob_mean,axis = -1),[-1]) #pred is the predicted label with the mean of generated samples
                        #THIS TIME I DIDN'T INCLUDE TAU
                        var_one = var_calculate(pred,prob_variance)
                        pred = np.reshape(pred,[image_h,image_w])
                        

                pred_tot.append(pred)
                var_tot.append(var_one)
            
            draw_plots_bayes(images, labels, pred_tot, var_tot)

    def print_checkpoint(self, all_tensors=True, tensor_name=''):
        print_tensors_in_checkpoint_file(file_name=self.config["SAVE_MODEL_DIR"], tensor_name=tensor_name, all_tensors=all_tensors)
        
  
    def visual_results_gta5_v2(self, dataset_type = "TEST", images_index = 3, FLAG_MAX_VOTE = False):
        
        image_w = self.config["INPUT_WIDTH"]
        image_h = self.config["INPUT_HEIGHT"]
        image_c = self.config["INPUT_CHANNELS"]
        train_dir = self.config["SAVE_MODEL_DIR"]
        FLAG_BAYES = self.config["BAYES"]

        with self.sess as sess:
            name_to_var_map = {var.op.name: var for var in tf.global_variables()}
            #print(name_to_var_map)
            name_to_var_map["conv1_1/biases"] = name_to_var_map["conv1_1/biases_1"]
            name_to_var_map["conv1_2/biases"] = name_to_var_map["conv1_2/biases_1"]
            name_to_var_map["conv2_1/biases"] = name_to_var_map["conv2_1/biases_1"]
            name_to_var_map["conv2_2/biases"] = name_to_var_map["conv2_2/biases_1"]
            name_to_var_map["conv3_1/biases"] = name_to_var_map["conv3_1/biases_1"]
            name_to_var_map["conv3_2/biases"] = name_to_var_map["conv3_2/biases_1"]
            name_to_var_map["conv3_3/biases"] = name_to_var_map["conv3_3/biases_1"]
            name_to_var_map["conv4_1/biases"] = name_to_var_map["conv4_1/biases_1"]
            name_to_var_map["conv4_2/biases"] = name_to_var_map["conv4_2/biases_1"]
            name_to_var_map["conv4_3/biases"] = name_to_var_map["conv4_3/biases_1"]
            name_to_var_map["conv5_1/biases"] = name_to_var_map["conv5_1/biases_1"]
            name_to_var_map["conv5_2/biases"] = name_to_var_map["conv5_2/biases_1"]
            name_to_var_map["conv5_3/biases"] = name_to_var_map["conv5_3/biases_1"]
            del name_to_var_map["conv1_1/biases_1"]
            del name_to_var_map["conv1_2/biases_1"]
            del name_to_var_map["conv2_1/biases_1"]
            del name_to_var_map["conv2_2/biases_1"]
            del name_to_var_map["conv3_1/biases_1"]
            del name_to_var_map["conv3_2/biases_1"]
            del name_to_var_map["conv3_3/biases_1"]
            del name_to_var_map["conv4_1/biases_1"]
            del name_to_var_map["conv4_2/biases_1"]
            del name_to_var_map["conv4_3/biases_1"]
            del name_to_var_map["conv5_1/biases_1"]
            del name_to_var_map["conv5_2/biases_1"]
            del name_to_var_map["conv5_3/biases_1"]
            
            # Restore saved session
            saver = tf.train.Saver(name_to_var_map)
            saver.restore(sess, train_dir)
            
            _, _, prediction = cal_loss_gta5_v2(self.logits, self.labels_pl, 2)
            prob = tf.nn.softmax(self.logits,dim = -1)
            
            if (dataset_type=='TRAIN'):
                test_type_path = self.config["TRAIN_FILE"]
                if type(images_index) == list:
                    indexes = images_index
                else:
                    indexes = random.sample(range(367),images_index)
                #indexes = [0,75,150,225,300]
            elif (dataset_type=='VAL'):
                test_type_path = self.config["VAL_FILE"]
                if type(images_index) == list:
                    indexes = images_index
                else:
                    indexes = random.sample(range(101),images_index)
                #indexes = [0,25,50,75,100]
            elif (dataset_type=='TEST'):
                test_type_path = self.config["TEST_FILE"]
                if type(images_index) == list:
                    indexes = images_index
                else:
                    indexes = random.sample(range(233),images_index)
                #indexes = [0,50,100,150,200]

            # Load images
            image_filename,label_filename = get_filename_list(test_type_path, self.config)
            images, labels = get_all_test_data_gta5(image_filename,label_filename)

            # Keep images subset of length images_index
            images = [images[i] for i in indexes]
            labels = [labels[i] for i in indexes]
            
            num_sample_generate = 30
            pred_tot = []
            var_tot = []
            
            for image_batch, label_batch in zip(images,labels):
                
                image_batch = np.reshape(image_batch,[1,image_h,image_w,image_c])
                label_batch = np.reshape(label_batch,[1,image_h,image_w,1])
                
                if FLAG_BAYES is False:
                    fetches = [prediction]
                    feed_dict = {self.inputs_pl: image_batch, 
                                 self.labels_pl: label_batch, 
                                 self.is_training_pl: False, 
                                 self.keep_prob_pl: 0.5,
                                 self.batch_size_pl: 1}
                    pred = sess.run(fetches = fetches, feed_dict = feed_dict)
                    pred = np.reshape(pred,[image_h,image_w])
                    var_one = []
                else:
                    feed_dict = {self.inputs_pl: image_batch, 
                                 self.labels_pl: label_batch, 
                                 self.is_training_pl: False, 
                                 self.keep_prob_pl: 0.5,
                                 self.with_dropout_pl: True,
                                 self.batch_size_pl: 1}
                    prob_iter_tot = []
                    pred_iter_tot = []
                    for iter_step in range(num_sample_generate):
                        prob_iter_step = sess.run(fetches = [prob], feed_dict = feed_dict) 
                        prob_iter_tot.append(prob_iter_step)
                        pred_iter_tot.append(np.reshape(np.argmax(prob_iter_step,axis = -1),[-1]))
                        
                    if FLAG_MAX_VOTE is True:
                        prob_variance,pred = MAX_VOTE(pred_iter_tot,prob_iter_tot,self.config["NUM_CLASSES"])
                        #acc_per = np.mean(np.equal(pred,np.reshape(label_batch,[-1])))
                        var_one = var_calculate_gta5(pred,prob_variance)
                        pred = np.reshape(pred,[image_h,image_w])
                    else:
                        prob_mean = np.nanmean(prob_iter_tot,axis = 0)
                        prob_variance = np.var(prob_iter_tot, axis = 0)
                        pred = np.reshape(np.argmax(prob_mean,axis = -1),[-1]) #pred is the predicted label with the mean of generated samples
                        #THIS TIME I DIDN'T INCLUDE TAU
                        var_one = var_calculate_gta5(pred,prob_variance)
                        pred = np.reshape(pred,[image_h,image_w])
                        

                pred_tot.append(pred)
                var_tot.append(var_one)
            
            draw_plots_bayes(images, labels, pred_tot, var_tot)
    
    
    
    def visual_results_gta5(self, n_class=5, dataset_type = "TEST", images_index = 3, FLAG_MAX_VOTE = False):
        
        image_w = self.config["INPUT_WIDTH"]
        image_h = self.config["INPUT_HEIGHT"]
        image_c = self.config["INPUT_CHANNELS"]
        train_dir = self.config["SAVE_MODEL_DIR"]
        FLAG_BAYES = self.config["BAYES"]

        with self.sess as sess:
            name_to_var_map = {var.op.name: var for var in tf.global_variables()}
            #print(name_to_var_map)
            name_to_var_map["conv1_1/biases"] = name_to_var_map["conv1_1/biases_1"]
            name_to_var_map["conv1_2/biases"] = name_to_var_map["conv1_2/biases_1"]
            name_to_var_map["conv2_1/biases"] = name_to_var_map["conv2_1/biases_1"]
            name_to_var_map["conv2_2/biases"] = name_to_var_map["conv2_2/biases_1"]
            name_to_var_map["conv3_1/biases"] = name_to_var_map["conv3_1/biases_1"]
            name_to_var_map["conv3_2/biases"] = name_to_var_map["conv3_2/biases_1"]
            name_to_var_map["conv3_3/biases"] = name_to_var_map["conv3_3/biases_1"]
            name_to_var_map["conv4_1/biases"] = name_to_var_map["conv4_1/biases_1"]
            name_to_var_map["conv4_2/biases"] = name_to_var_map["conv4_2/biases_1"]
            name_to_var_map["conv4_3/biases"] = name_to_var_map["conv4_3/biases_1"]
            name_to_var_map["conv5_1/biases"] = name_to_var_map["conv5_1/biases_1"]
            name_to_var_map["conv5_2/biases"] = name_to_var_map["conv5_2/biases_1"]
            name_to_var_map["conv5_3/biases"] = name_to_var_map["conv5_3/biases_1"]
            del name_to_var_map["conv1_1/biases_1"]
            del name_to_var_map["conv1_2/biases_1"]
            del name_to_var_map["conv2_1/biases_1"]
            del name_to_var_map["conv2_2/biases_1"]
            del name_to_var_map["conv3_1/biases_1"]
            del name_to_var_map["conv3_2/biases_1"]
            del name_to_var_map["conv3_3/biases_1"]
            del name_to_var_map["conv4_1/biases_1"]
            del name_to_var_map["conv4_2/biases_1"]
            del name_to_var_map["conv4_3/biases_1"]
            del name_to_var_map["conv5_1/biases_1"]
            del name_to_var_map["conv5_2/biases_1"]
            del name_to_var_map["conv5_3/biases_1"]
            
            # Restore saved session
            saver = tf.train.Saver() #name_to_var_map)
            saver.restore(sess, train_dir)
            
            _, _, prediction = cal_loss_gta5(self.logits, self.labels_pl, n_class)
            prob = tf.nn.softmax(self.logits,dim = -1)
            
            if (dataset_type=='TRAIN'):
                test_type_path = self.config["TRAIN_FILE"]
                if type(images_index) == list:
                    indexes = images_index
                else:
                    indexes = random.sample(range(367),images_index)
                #indexes = [0,75,150,225,300]
            elif (dataset_type=='VAL'):
                test_type_path = self.config["VAL_FILE"]
                if type(images_index) == list:
                    indexes = images_index
                else:
                    indexes = random.sample(range(101),images_index)
                #indexes = [0,25,50,75,100]
            elif (dataset_type=='TEST'):
                test_type_path = self.config["TEST_FILE"]
                if type(images_index) == list:
                    indexes = images_index
                else:
                    indexes = random.sample(range(233),images_index)
                #indexes = [0,50,100,150,200]

            # Load images
            image_filename,label_filename = get_filename_list(test_type_path, self.config)
            images, labels = get_all_test_data_gta5(image_filename,label_filename)

            # Keep images subset of length images_index
            images = [images[i] for i in indexes]
#             labels = [labels[i] for i in indexes]
            labels = [np.ones((image_h,image_w)) for i in indexes]
            
            num_sample_generate = 30
            pred_tot = []
            var_tot = []

            for image_batch, label_batch in zip(images,labels):
                
                image_batch = np.reshape(image_batch,[1,image_h,image_w,image_c])
                label_batch = np.reshape(label_batch,[1,image_h,image_w,1])
#                 label_batch = np.zeros((1,image_h,image_w,1))
    
                if FLAG_BAYES is False:
                    fetches = [prediction]
                    feed_dict = {self.inputs_pl: image_batch, 
                                 self.labels_pl: label_batch, 
                                 self.is_training_pl: False, 
                                 self.keep_prob_pl: 0.5,
                                 self.batch_size_pl: 1}
                    pred = sess.run(fetches = fetches, feed_dict = feed_dict)
                    pred = np.reshape(pred,[image_h,image_w])
                    var_one = []
                else:
                    feed_dict = {self.inputs_pl: image_batch, 
                                 self.labels_pl: label_batch, 
                                 self.is_training_pl: False, 
                                 self.keep_prob_pl: 0.5,
                                 self.with_dropout_pl: True,
                                 self.batch_size_pl: 1}
                    prob_iter_tot = []
                    pred_iter_tot = []
                    for iter_step in range(num_sample_generate):
                        prob_iter_step = sess.run(fetches = [prob], feed_dict = feed_dict) 
                        prob_iter_tot.append(prob_iter_step)
                        pred_iter_tot.append(np.reshape(np.argmax(prob_iter_step,axis = -1),[-1]))
                        
                    if FLAG_MAX_VOTE is True:
                        prob_variance,pred = MAX_VOTE(pred_iter_tot,prob_iter_tot,self.config["NUM_CLASSES"])
                        #acc_per = np.mean(np.equal(pred,np.reshape(label_batch,[-1])))
                        var_one = var_calculate_gta5(pred,prob_variance)
                        pred = np.reshape(pred,[image_h,image_w])
                    else:
                        prob_mean = np.nanmean(prob_iter_tot,axis = 0)
                        prob_variance = np.var(prob_iter_tot, axis = 0)
                        pred = np.reshape(np.argmax(prob_mean,axis = -1),[-1]) #pred is the predicted label with the mean of generated samples
                        #THIS TIME I DIDN'T INCLUDE TAU
                        var_one = var_calculate_gta5(pred,prob_variance)
                        pred = np.reshape(pred,[image_h,image_w])
                        

                pred_tot.append(pred)
                var_tot.append(var_one)
            
            draw_plots_bayes(images, labels, pred_tot, var_tot)
#             for image in pred_tot:
#                 print(image.shape)
#                 writeImage(image)
    
    def create_demo(self, images):
        
        image_w = self.config["INPUT_WIDTH"]
        image_h = self.config["INPUT_HEIGHT"]
        image_c = self.config["INPUT_CHANNELS"]
        train_dir = self.config["SAVE_MODEL_DIR"]
        FLAG_BAYES = self.config["BAYES"]
        n_class = self.config["NUM_CLASSES"]
        
        images = [misc.imresize(image, (image_h, image_w)) for image in images]
        
        with self.sess as sess:
            name_to_var_map = {var.op.name: var for var in tf.global_variables()}
            #print(name_to_var_map)
            name_to_var_map["conv1_1/biases"] = name_to_var_map["conv1_1/biases_1"]
            name_to_var_map["conv1_2/biases"] = name_to_var_map["conv1_2/biases_1"]
            name_to_var_map["conv2_1/biases"] = name_to_var_map["conv2_1/biases_1"]
            name_to_var_map["conv2_2/biases"] = name_to_var_map["conv2_2/biases_1"]
            name_to_var_map["conv3_1/biases"] = name_to_var_map["conv3_1/biases_1"]
            name_to_var_map["conv3_2/biases"] = name_to_var_map["conv3_2/biases_1"]
            name_to_var_map["conv3_3/biases"] = name_to_var_map["conv3_3/biases_1"]
            name_to_var_map["conv4_1/biases"] = name_to_var_map["conv4_1/biases_1"]
            name_to_var_map["conv4_2/biases"] = name_to_var_map["conv4_2/biases_1"]
            name_to_var_map["conv4_3/biases"] = name_to_var_map["conv4_3/biases_1"]
            name_to_var_map["conv5_1/biases"] = name_to_var_map["conv5_1/biases_1"]
            name_to_var_map["conv5_2/biases"] = name_to_var_map["conv5_2/biases_1"]
            name_to_var_map["conv5_3/biases"] = name_to_var_map["conv5_3/biases_1"]
            del name_to_var_map["conv1_1/biases_1"]
            del name_to_var_map["conv1_2/biases_1"]
            del name_to_var_map["conv2_1/biases_1"]
            del name_to_var_map["conv2_2/biases_1"]
            del name_to_var_map["conv3_1/biases_1"]
            del name_to_var_map["conv3_2/biases_1"]
            del name_to_var_map["conv3_3/biases_1"]
            del name_to_var_map["conv4_1/biases_1"]
            del name_to_var_map["conv4_2/biases_1"]
            del name_to_var_map["conv4_3/biases_1"]
            del name_to_var_map["conv5_1/biases_1"]
            del name_to_var_map["conv5_2/biases_1"]
            del name_to_var_map["conv5_3/biases_1"]
            
            saver = tf.train.Saver() #name_to_var_map)
            saver.restore(sess, train_dir)
            
            _, _, prediction = cal_loss_gta5(self.logits, self.labels_pl, n_class)
            prob = tf.nn.softmax(self.logits,dim = -1)
            
            pred_tot = []
            var_tot = []
            
            labels = []
            for i in range(len(images)):
                
                labels.append(np.ones((images[i].shape[0], images[i].shape[1])))
            

            for image_batch, label_batch in zip(images,labels):

                image_batch = np.reshape(image_batch,[1, image_h, image_w, image_c])
                label_batch = np.reshape(label_batch,[1, image_h, image_w, 1])
                
                fetches = [prediction]
                feed_dict = {self.inputs_pl: image_batch, 
                             self.labels_pl: label_batch, 
                             self.is_training_pl: False, 
                             self.keep_prob_pl: 0.5,
                             self.with_dropout_pl: True,
                             self.batch_size_pl: 1}
                pred = sess.run(fetches = fetches, feed_dict = feed_dict)
                pred = np.reshape(pred,[image_h,image_w])
                
                pred_tot.append(pred)
            
            pred_tot = [writeImage(pred_img) for pred_img in pred_tot]
            return images, pred_tot

                
    def visual_results_external_image(self, images, FLAG_MAX_VOTE = False):
        
        #train_dir = "./saved_models/segnet_vgg_bayes/segnet_vgg_bayes_30000/model.ckpt-30000"
        #train_dir = "./saved_models/segnet_scratch/segnet_scratch_30000/model.ckpt-30000"
        
        
        i_width = 480
        i_height = 360
        images = [misc.imresize(image, (i_height, i_width)) for image in images]
        
        image_w = self.config["INPUT_WIDTH"]
        image_h = self.config["INPUT_HEIGHT"]
        image_c = self.config["INPUT_CHANNELS"]
        train_dir = self.config["SAVE_MODEL_DIR"]
        FLAG_BAYES = self.config["BAYES"]

        with self.sess as sess:
            
            # Restore saved session
            saver = tf.train.Saver()
           # saver.restore(sess, train_dir)
            
            _, _, prediction = cal_loss(logits=self.logits, 
                                           labels=self.labels_pl)
            prob = tf.nn.softmax(self.logits,dim = -1)
            
            num_sample_generate = 30
            pred_tot = []
            var_tot = []
            
            labels = []
            for i in range(len(images)):
                labels.append(np.array([[1 for x in range(480)] for y in range(360)]))
            
            
            inference_time = []
            start_time = time.time()
            
            for image_batch, label_batch in zip(images,labels):
            #for image_batch in zip(images):
                
                image_batch = np.reshape(image_batch,[1,image_h,image_w,image_c])
                label_batch = np.reshape(label_batch,[1,image_h,image_w,1])
                
                if FLAG_BAYES is False:
                    fetches = [prediction]
                    feed_dict = {self.inputs_pl: image_batch, 
                                 self.labels_pl: label_batch, 
                                 self.is_training_pl: False, 
                                 self.keep_prob_pl: 0.5,
                                 self.batch_size_pl: 1}
                    pred = sess.run(fetches = fetches, feed_dict = feed_dict)
                    pred = np.reshape(pred,[image_h,image_w])
                    var_one = []
                else:
                    feed_dict = {self.inputs_pl: image_batch, 
                                 self.labels_pl: label_batch, 
                                 self.is_training_pl: False, 
                                 self.keep_prob_pl: 0.5,
                                 self.with_dropout_pl: True,
                                 self.batch_size_pl: 1}
                    prob_iter_tot = []
                    pred_iter_tot = []
                    for iter_step in range(num_sample_generate):
                        prob_iter_step = sess.run(fetches = [prob], feed_dict = feed_dict)
                        prob_iter_tot.append(prob_iter_step)
                        pred_iter_tot.append(np.reshape(np.argmax(prob_iter_step,axis = -1),[-1]))
                        
                    if FLAG_MAX_VOTE is True:
                        prob_variance,pred = MAX_VOTE(pred_iter_tot,prob_iter_tot,self.config["NUM_CLASSES"])
                        #acc_per = np.mean(np.equal(pred,np.reshape(label_batch,[-1])))
                        var_one = var_calculate(pred,prob_variance)
                        pred = np.reshape(pred,[image_h,image_w])
                    else:
                        prob_mean = np.nanmean(prob_iter_tot,axis = 0)
                        prob_variance = np.var(prob_iter_tot, axis = 0)
                        pred = np.reshape(np.argmax(prob_mean,axis = -1),[-1]) #pred is the predicted label with the mean of generated samples
                        #THIS TIME I DIDN'T INCLUDE TAU
                        var_one = var_calculate(pred,prob_variance)
                        pred = np.reshape(pred,[image_h,image_w])
                        

                pred_tot.append(pred)
                var_tot.append(var_one)
                inference_time.append(time.time() - start_time)
                start_time = time.time()
            
            try:
                draw_plots_bayes_external(images, pred_tot, var_tot)
                return pred_tot, var_tot, inference_time
            except:
                return pred_tot, var_tot, inference_time

    def test_gta5_v2(self):

        image_filename, label_filename = get_filename_list(self.test_file, self.config)
        train_dir = self.config["SAVE_MODEL_DIR"]
        loss_weight = np.array([
            0.25, 
            5, 
            1, 
            1, 
            1, 
            1, 
            1, 
            1,
            1,
            1,
            1
            ])

        with self.graph.as_default():
            with self.sess as sess:
                saver = tf.train.Saver() #name_to_var_map) #
                saver.restore(sess, train_dir) #
                loss, accuracy, accuracy_sidewalk, accuracy_iou, prediction = weighted_loss_v2(self.logits, self.labels_pl, self.num_classes, frequency=loss_weight)
                prob = tf.nn.softmax(self.logits, dim=-1)
                prob = tf.reshape(prob, [self.input_h, self.input_w, self.num_classes])

                images, labels = get_all_test_data(image_filename, label_filename)

                NUM_SAMPLE = []
                for i in range(30):
                    NUM_SAMPLE.append(2 * i + 1)

                acc_final = []
                acc_final_sidewalk = []
                acc_final_iou = []
                iu_final = []
                iu_mean_final = []
                # uncomment the line below to only run for two times.
                # NUM_SAMPLE = [1, 30]
                NUM_SAMPLE = [1]
                for num_sample_generate in NUM_SAMPLE:

                    loss_tot = []
                    acc_tot = []
                    acc_tot_sidewalk = []
                    acc_tot_iou = []
                    pred_tot = []
                    var_tot = []
                    hist = np.zeros((self.num_classes, self.num_classes))
                    step = 0
                    for image_batch, label_batch in zip(images, labels):
                        image_batch = np.reshape(image_batch, [1, self.input_h, self.input_w, self.input_c])
                        label_batch = np.reshape(label_batch, [1, self.input_h, self.input_w, 1])
                        # comment the code below to apply the dropout for all the samples
                        if num_sample_generate == 1:
                            feed_dict = {self.inputs_pl: image_batch, self.labels_pl: label_batch,
                                         self.is_training_pl: False,
                                         self.keep_prob_pl: 0.5, self.with_dropout_pl: False,
                                         self.batch_size_pl: 1}
                        else:
                            feed_dict = {self.inputs_pl: image_batch, self.labels_pl: label_batch,
                                         self.is_training_pl: False,
                                         self.keep_prob_pl: 0.5, self.with_dropout_pl: True,
                                         self.batch_size_pl: 1}
                        # uncomment this code below to run the dropout for all the samples
                        # feed_dict = {test_data_tensor: image_batch, test_label_tensor:label_batch, phase_train: False, keep_prob:0.5, phase_train_dropout:True}
                        fetches = [loss, accuracy, accuracy_sidewalk, accuracy_iou, self.logits, prediction]
                        if self.bayes is False:
                            loss_per, acc_per, acc_per_sidewalk, acc_per_iou, logit, pred = sess.run(fetches=fetches, feed_dict=feed_dict)
                            var_one = []
                        else:
                            logit_iter_tot = []
                            loss_iter_tot = []
                            acc_iter_tot = []
                            acc_iter_tot_sidewalk = []
                            acc_iter_tot_iou = []
                            prob_iter_tot = []
                            logit_iter_temp = []
                            for iter_step in range(num_sample_generate):
                                loss_iter_step, acc_iter_step, acc_iter_step_sidewalk, acc_iter_step_iou, logit_iter_step, prob_iter_step = sess.run(
                                    fetches=[loss, accuracy, accuracy_sidewalk, accuracy_iou, self.logits, prob], feed_dict=feed_dict)
                                loss_iter_tot.append(loss_iter_step)
                                acc_iter_tot.append(acc_iter_step)
                                acc_iter_tot_sidewalk.append(acc_iter_step_sidewalk)
                                acc_iter_tot_iou.append(acc_iter_step_iou)
                                logit_iter_tot.append(logit_iter_step)
                                prob_iter_tot.append(prob_iter_step)
                                logit_iter_temp.append(
                                    np.reshape(logit_iter_step, [self.input_h, self.input_w, self.num_classes]))

                            loss_per = np.nanmean(loss_iter_tot)
                            acc_per = np.nanmean(acc_iter_tot)
                            acc_per_sidewalk = np.nanmean(acc_iter_tot_sidewalk)
                            acc_per_iou = np.nanmean(acc_iter_tot_iou)
                            logit = np.nanmean(logit_iter_tot, axis=0)
                            print(np.shape(prob_iter_tot))

                            prob_mean = np.nanmean(prob_iter_tot, axis=0)
                            prob_variance = np.var(prob_iter_tot, axis=0)
                            logit_variance = np.var(logit_iter_temp, axis=0)

                            # THIS TIME I DIDN'T INCLUDE TAU
                            pred = np.reshape(np.argmax(prob_mean, axis=-1), [-1])  # pred is the predicted label

                            var_sep = []  # var_sep is the corresponding variance if this pixel choose label k
                            length_cur = 0  # length_cur represent how many pixels has been read for one images
                            for row in np.reshape(prob_variance, [self.input_h * self.input_w, self.num_classes]):
                                temp = row[pred[length_cur]]
                                length_cur += 1
                                var_sep.append(temp)
                            var_one = np.reshape(var_sep, [self.input_h,
                                                           self.input_w])  # var_one is the corresponding variance in terms of the "optimal" label
                            pred = np.reshape(pred, [self.input_h, self.input_w])

                        loss_tot.append(loss_per)
                        acc_tot.append(acc_per)
                        acc_tot_sidewalk.append(acc_per_sidewalk)
                        acc_tot_iou.append(acc_per_iou)
                        pred_tot.append(pred)
                        var_tot.append(var_one)
                        print("Image Index {}: TEST Loss{:6.3f}, TEST Accu {:6.3f}, TEST Accu Sidewalk {:6.3f}, TEST Accu IOU{:6.3f}".format(step, loss_tot[-1], acc_tot[-1], acc_tot_sidewalk[-1], acc_tot_iou[-1]))
                        step = step + 1
                        per_class_acc(logit, label_batch, self.num_classes)
                        hist += get_hist(logit, label_batch)

                    acc_tot = np.diag(hist).sum() / hist.sum()
                    acc_tot_sidewalk = np.nanmean(np.array(acc_tot_sidewalk))
                    acc_tot_iou = np.nanmean(np.array(acc_tot_iou))
                    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

                    print("Total Accuracy for test image: ", acc_tot)
                    print("Sidewalk Accuracy for test image: ", acc_tot_sidewalk)
                    print("IOU Accuracy for test image: ", acc_tot_iou)
                    print("Total MoI for test images: ", iu)
                    print("mean MoI for test images: ", np.nanmean(iu))

                    acc_final.append(acc_tot)
                    acc_final_sidewalk.append(acc_tot_sidewalk)
                    acc_final_iou.append(acc_tot_iou)
                    iu_final.append(iu)
                    iu_mean_final.append(np.nanmean(iu))

            return acc_final, ac_final_sidewalk, acc_final_iou, iu_final, iu_mean_final, prob_variance, logit_variance, pred_tot, var_tot
                        
    def test_gta5(self):
        image_filename, label_filename = get_filename_list(self.test_file, self.config)
        train_dir = self.config["SAVE_MODEL_DIR"]
        with self.graph.as_default():
            with self.sess as sess:
                
                name_to_var_map = {var.op.name: var for var in tf.global_variables()}
                #print(name_to_var_map)
                name_to_var_map["conv1_1/biases"] = name_to_var_map["conv1_1/biases_1"]
                name_to_var_map["conv1_2/biases"] = name_to_var_map["conv1_2/biases_1"]
                name_to_var_map["conv2_1/biases"] = name_to_var_map["conv2_1/biases_1"]
                name_to_var_map["conv2_2/biases"] = name_to_var_map["conv2_2/biases_1"]
                name_to_var_map["conv3_1/biases"] = name_to_var_map["conv3_1/biases_1"]
                name_to_var_map["conv3_2/biases"] = name_to_var_map["conv3_2/biases_1"]
                name_to_var_map["conv3_3/biases"] = name_to_var_map["conv3_3/biases_1"]
                name_to_var_map["conv4_1/biases"] = name_to_var_map["conv4_1/biases_1"]
                name_to_var_map["conv4_2/biases"] = name_to_var_map["conv4_2/biases_1"]
                name_to_var_map["conv4_3/biases"] = name_to_var_map["conv4_3/biases_1"]
                name_to_var_map["conv5_1/biases"] = name_to_var_map["conv5_1/biases_1"]
                name_to_var_map["conv5_2/biases"] = name_to_var_map["conv5_2/biases_1"]
                name_to_var_map["conv5_3/biases"] = name_to_var_map["conv5_3/biases_1"]
                del name_to_var_map["conv1_1/biases_1"]
                del name_to_var_map["conv1_2/biases_1"]
                del name_to_var_map["conv2_1/biases_1"]
                del name_to_var_map["conv2_2/biases_1"]
                del name_to_var_map["conv3_1/biases_1"]
                del name_to_var_map["conv3_2/biases_1"]
                del name_to_var_map["conv3_3/biases_1"]
                del name_to_var_map["conv4_1/biases_1"]
                del name_to_var_map["conv4_2/biases_1"]
                del name_to_var_map["conv4_3/biases_1"]
                del name_to_var_map["conv5_1/biases_1"]
                del name_to_var_map["conv5_2/biases_1"]
                del name_to_var_map["conv5_3/biases_1"]
                
                saver = tf.train.Saver(name_to_var_map)
                saver.restore(sess, train_dir)
                loss, accuracy, prediction = cal_loss_gta5(self.logits, self.labels_pl, self.num_classes)
                prob = tf.nn.softmax(self.logits, dim=-1)
                prob = tf.reshape(prob, [self.input_h, self.input_w, self.num_classes])

                images, labels = get_all_test_data(image_filename, label_filename)

                NUM_SAMPLE = []
                for i in range(30):
                    NUM_SAMPLE.append(2 * i + 1)

                acc_final = []
                iu_final = []
                iu_mean_final = []
                acc_final_sidewalk = []
                # uncomment the line below to only run for two times.
                # NUM_SAMPLE = [1, 30]
                NUM_SAMPLE = [1]
                for num_sample_generate in NUM_SAMPLE:

                    loss_tot = []
                    acc_tot = []
                    pred_tot = []
                    var_tot = []
                    sidewalk_acc = [] # added
                    hist = np.zeros((self.num_classes, self.num_classes))
                    step = 0
                    for image_batch, label_batch in zip(images, labels):
                        image_batch = np.reshape(image_batch, [1, self.input_h, self.input_w, self.input_c])
                        label_batch = np.reshape(label_batch, [1, self.input_h, self.input_w, 1])
                        # comment the code below to apply the dropout for all the samples
                        if num_sample_generate == 1:
                            feed_dict = {self.inputs_pl: image_batch, self.labels_pl: label_batch,
                                         self.is_training_pl: False,
                                         self.keep_prob_pl: 0.5, self.with_dropout_pl: False,
                                         self.batch_size_pl: 1}
                        else:
                            feed_dict = {self.inputs_pl: image_batch, self.labels_pl: label_batch,
                                         self.is_training_pl: False,
                                         self.keep_prob_pl: 0.5, self.with_dropout_pl: True,
                                         self.batch_size_pl: 1}
                        # uncomment this code below to run the dropout for all the samples
                        # feed_dict = {test_data_tensor: image_batch, test_label_tensor:label_batch, phase_train: False, keep_prob:0.5, phase_train_dropout:True}
                        fetches = [loss, accuracy, self.logits, prediction]
                        if self.bayes is False:
                            loss_per, acc_per, logit, pred = sess.run(fetches=fetches, feed_dict=feed_dict)
                            var_one = []
                        else:
                            logit_iter_tot = []
                            loss_iter_tot = []
                            acc_iter_tot = []
                            prob_iter_tot = []
                            logit_iter_temp = []
                            for iter_step in range(num_sample_generate):
                                loss_iter_step, acc_iter_step, logit_iter_step, prob_iter_step = sess.run(
                                    fetches=[loss, accuracy, self.logits, prob], feed_dict=feed_dict)
                                loss_iter_tot.append(loss_iter_step)
                                acc_iter_tot.append(acc_iter_step)
                                logit_iter_tot.append(logit_iter_step)
                                prob_iter_tot.append(prob_iter_step)
                                logit_iter_temp.append(
                                    np.reshape(logit_iter_step, [self.input_h, self.input_w, self.num_classes]))

                            loss_per = np.nanmean(loss_iter_tot)
                            acc_per = np.nanmean(acc_iter_tot)
                            logit = np.nanmean(logit_iter_tot, axis=0)
                            #print(np.shape(prob_iter_tot))

                            prob_mean = np.nanmean(prob_iter_tot, axis=0)
                            prob_variance = np.var(prob_iter_tot, axis=0)
                            logit_variance = np.var(logit_iter_temp, axis=0)

                            # THIS TIME I DIDN'T INCLUDE TAU
                            pred = np.reshape(np.argmax(prob_mean, axis=-1), [-1])  # pred is the predicted label

                            var_sep = []  # var_sep is the corresponding variance if this pixel choose label k
                            length_cur = 0  # length_cur represent how many pixels has been read for one images
                            for row in np.reshape(prob_variance, [self.input_h * self.input_w, self.num_classes]):
                                temp = row[pred[length_cur]]
                                length_cur += 1
                                var_sep.append(temp)
                            var_one = np.reshape(var_sep, [self.input_h,
                                                           self.input_w])  # var_one is the corresponding variance in terms of the "optimal" label
                            pred = np.reshape(pred, [self.input_h, self.input_w])

                        loss_tot.append(loss_per)
                        acc_tot.append(acc_per)
                        pred_tot.append(pred)
                        var_tot.append(var_one)
                        print("Image Index {}: TEST Loss{:6.3f}, TEST Accu {:6.3f}".format(step, loss_tot[-1], acc_tot[-1]))
                        step = step + 1
                        #per_class_acc(logit, label_batch, self.num_classes) # commented this
                        sidewalk_acc.append(per_class_acc(logit, label_batch, self.num_classes))
                        hist += get_hist(logit, label_batch)

                    acc_tot = np.diag(hist).sum() / hist.sum()
                    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
                    acc_tot_sidewalk = np.mean(np.array(sidewalk_acc)) # added this

                    print("Total Accuracy for test image: ", acc_tot)
                    print("Sidewalk Accuracy for test image: ", acc_tot_sidewalk)
                    print("Total MoI for test images: ", iu)
                    print("mean MoI for test images: ", np.nanmean(iu))

                    acc_final.append(acc_tot)
                    acc_final_sidewalk.append(acc_tot_sidewalk)
                    iu_final.append(iu)
                    iu_mean_final.append(np.nanmean(iu))

            return acc_final, acc_final_sidewalk, iu_final, iu_mean_final, prob_variance, logit_variance, pred_tot, var_tot
            
    def test(self):
        image_filename, label_filename = get_filename_list(self.test_file, self.config)

        with self.graph.as_default():
            with self.sess as sess:
                loss, accuracy, prediction = normal_loss(self.logits, self.labels_pl, self.num_classes)
                prob = tf.nn.softmax(self.logits, dim=-1)
                prob = tf.reshape(prob, [self.input_h, self.input_w, self.num_classes])

                images, labels = get_all_test_data(image_filename, label_filename)

                NUM_SAMPLE = []
                for i in range(30):
                    NUM_SAMPLE.append(2 * i + 1)

                acc_final = []
                iu_final = []
                iu_mean_final = []
                # uncomment the line below to only run for two times.
                # NUM_SAMPLE = [1, 30]
                NUM_SAMPLE = [1]
                for num_sample_generate in NUM_SAMPLE:

                    loss_tot = []
                    acc_tot = []
                    pred_tot = []
                    var_tot = []
                    hist = np.zeros((self.num_classes, self.num_classes))
                    step = 0
                    for image_batch, label_batch in zip(images, labels):
                        image_batch = np.reshape(image_batch, [1, self.input_h, self.input_w, self.input_c])
                        label_batch = np.reshape(label_batch, [1, self.input_h, self.input_w, 1])
                        # comment the code below to apply the dropout for all the samples
                        if num_sample_generate == 1:
                            feed_dict = {self.inputs_pl: image_batch, self.labels_pl: label_batch,
                                         self.is_training_pl: False,
                                         self.keep_prob_pl: 0.5, self.with_dropout_pl: False,
                                         self.batch_size_pl: 1}
                        else:
                            feed_dict = {self.inputs_pl: image_batch, self.labels_pl: label_batch,
                                         self.is_training_pl: False,
                                         self.keep_prob_pl: 0.5, self.with_dropout_pl: True,
                                         self.batch_size_pl: 1}
                        # uncomment this code below to run the dropout for all the samples
                        # feed_dict = {test_data_tensor: image_batch, test_label_tensor:label_batch, phase_train: False, keep_prob:0.5, phase_train_dropout:True}
                        fetches = [loss, accuracy, self.logits, prediction]
                        if self.bayes is False:
                            loss_per, acc_per, logit, pred = sess.run(fetches=fetches, feed_dict=feed_dict)
                            var_one = []
                        else:
                            logit_iter_tot = []
                            loss_iter_tot = []
                            acc_iter_tot = []
                            prob_iter_tot = []
                            logit_iter_temp = []
                            for iter_step in range(num_sample_generate):
                                loss_iter_step, acc_iter_step, logit_iter_step, prob_iter_step = sess.run(
                                    fetches=[loss, accuracy, self.logits, prob], feed_dict=feed_dict)
                                loss_iter_tot.append(loss_iter_step)
                                acc_iter_tot.append(acc_iter_step)
                                logit_iter_tot.append(logit_iter_step)
                                prob_iter_tot.append(prob_iter_step)
                                logit_iter_temp.append(
                                    np.reshape(logit_iter_step, [self.input_h, self.input_w, self.num_classes]))

                            loss_per = np.nanmean(loss_iter_tot)
                            acc_per = np.nanmean(acc_iter_tot)
                            logit = np.nanmean(logit_iter_tot, axis=0)
                            print(np.shape(prob_iter_tot))

                            prob_mean = np.nanmean(prob_iter_tot, axis=0)
                            prob_variance = np.var(prob_iter_tot, axis=0)
                            logit_variance = np.var(logit_iter_temp, axis=0)

                            # THIS TIME I DIDN'T INCLUDE TAU
                            pred = np.reshape(np.argmax(prob_mean, axis=-1), [-1])  # pred is the predicted label

                            var_sep = []  # var_sep is the corresponding variance if this pixel choose label k
                            length_cur = 0  # length_cur represent how many pixels has been read for one images
                            for row in np.reshape(prob_variance, [self.input_h * self.input_w, self.num_classes]):
                                temp = row[pred[length_cur]]
                                length_cur += 1
                                var_sep.append(temp)
                            var_one = np.reshape(var_sep, [self.input_h,
                                                           self.input_w])  # var_one is the corresponding variance in terms of the "optimal" label
                            pred = np.reshape(pred, [self.input_h, self.input_w])

                        loss_tot.append(loss_per)
                        acc_tot.append(acc_per)
                        pred_tot.append(pred)
                        var_tot.append(var_one)
                        print("Image Index {}: TEST Loss{:6.3f}, TEST Accu {:6.3f}".format(step, loss_tot[-1], acc_tot[-1]))
                        step = step + 1
                        per_class_acc(logit, label_batch, self.num_classes)
                        hist += get_hist(logit, label_batch)

                    acc_tot = np.diag(hist).sum() / hist.sum()
                    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

                    print("Total Accuracy for test image: ", acc_tot)
                    print("Total MoI for test images: ", iu)
                    print("mean MoI for test images: ", np.nanmean(iu))

                    acc_final.append(acc_tot)
                    iu_final.append(iu)
                    iu_mean_final.append(np.nanmean(iu))

            return acc_final, iu_final, iu_mean_final, prob_variance, logit_variance, pred_tot, var_tot

    def save(self):
        np.save(os.path.join(self.saved_dir, "Data", "trainloss"), self.train_loss)
        np.save(os.path.join(self.saved_dir, "Data", "trainacc"), self.train_accuracy)
        np.save(os.path.join(self.saved_dir, "Data", "valloss"), self.val_loss)
        np.save(os.path.join(self.saved_dir, "Data", "valacc"), self.val_acc)
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver = tf.train.Saver()
                checkpoint_path = os.path.join(self.saved_dir, 'model.ckpt')
                self.saver.save(self.sess, checkpoint_path, global_step=self.model_version)
                self.model_version += 1

