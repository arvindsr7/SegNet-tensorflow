# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:12:45 2017
This file is utilized to calculate the loss, accuracy, per_class_accuracy, and MOI(mean of intersection)

"""
import tensorflow as tf
import numpy as np

def cal_loss(logits, labels, number_class):
    loss_weight = np.array([
        0.2595,
        0.1826,
        4.5640,
        0.1417,
        0.9051,
        0.3826,
        9.6446,
        1.8418,
        0.6823,
        6.2478,
        7.3614,
        1.0974
    ])
    # class 0 to 11, but the class 11 is ignored, so maybe the class 11 is background!

    labels = tf.to_int64( )
    loss, accuracy, prediction = weighted_loss(logits, labels, number_class=12, frequency=loss_weight)
    return loss, accuracy, prediction

def cal_loss_gta5_v2(logits, labels, number_class):
    """
    Modified loss functions with: 
        - only classes relevant to sidewalks used
        - weight for sidewalk increased 
        - new accuracy measure that amplified detection used 
    Classes:
        0 background
        1 sidewalk
        2 road
        3 veg
        4 building
        5 fence
        6 trafficsign
        7 firehydrant
        8 person
        9 bikes
        10 vehicles

        Computed fractions: 
        [
        0.23975128696351541, 
        0.2507147462990658, 
        0.16095488789992088, 
        0.07417991627042077, 
        0.2352810984060984, 
        0.012232230619511656, 
        0.0018319206780745242, 
        0.0007349405147719907, 
        0.017041525215062206, 
        0.0009230732839872778, 
        0.027867874511862293
        ]
    """
    # Make sure to edit loss weight in test_gta5_v2 as well if modified here. 
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

    labels = tf.to_int64(labels)
    loss, accuracy, accuracy_sidewalk, accuracy_iou, prediction = weighted_loss_v2(logits, labels, number_class=number_class, frequency=loss_weight)
    return loss, accuracy, accuracy_sidewalk, accuracy_iou, prediction

def cal_loss_gta5(logits, labels, number_class):
    loss_weight = np.array([
        0.5798636667907213, 
        0.2507147462990658, 
        0.017041525215062206, 
        0.0007349405147719907, 
        0.16095488789992088
    ])

    labels = tf.to_int64(labels)
    loss, accuracy, prediction = weighted_loss(logits, labels, number_class=number_class, frequency=loss_weight)
    return loss, accuracy, prediction

def weighted_loss_v2(logits, labels, number_class, frequency):
    """
    Modifief weighted_loss function with:
        - new accuracy measure that amplified detection used 
    NOTE: Sidewalk class is 1 
    """
    # parameters
    sidewalk_class = 1

    label_flatten = tf.reshape(labels, [-1])
    label_onehot = tf.one_hot(label_flatten, depth=number_class)
    logits_reshape = tf.reshape(logits, [-1, number_class])
    detected_flatten = tf.argmax(logits_reshape, -1)
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=label_onehot, logits=logits_reshape, pos_weight=frequency)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.summary.scalar('loss', cross_entropy_mean)

    # IOU sidewalk:
    #   intersection (detection and actual sidwalk pixels) / union 
    sidwalk_mask = tf.equal(label_flatten, sidewalk_class)
    detected_sidewalk = tf.to_float(tf.equal(detected_flatten, sidewalk_class))
    intersection_pixels = tf.reduce_sum(tf.to_float(tf.boolean_mask(detected_sidewalk, sidwalk_mask)))
    union_pixels = tf.reduce_sum(tf.to_float(sidwalk_mask)) + tf.reduce_sum(detected_sidewalk) - intersection_pixels
    accuracy_iou = intersection_pixels / union_pixels
    
    # accuracy sidewalk prediction: 
    #   Find total number of sidewalk pixels accurately predicted; divide it by total sidewalk pixel labels
    total_sidwalk_pixels = tf.reduce_sum(tf.to_float(sidwalk_mask))                                 
    accuracy_sidewalk = intersection_pixels / total_sidwalk_pixels

    # accuracy total:
    #   accuracy over all classes
    correct_prediction_total = tf.equal(detected_flatten, label_flatten)
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction_total))
    
    tf.summary.scalar('accuracy_total', accuracy)
    tf.summary.scalar('accuracy_sidewalk', accuracy_sidewalk)
    tf.summary.scalar('accuracy_iou', accuracy_iou)

    return cross_entropy_mean, accuracy, accuracy_sidewalk, accuracy_iou, tf.argmax(logits_reshape, -1)

def weighted_loss(logits, labels, number_class, frequency):
    """
    The reference paper is : https://arxiv.org/pdf/1411.4734.pdf 
    Median Frequency Balancing: alpha_c = median_freq/freq(c).
    median_freq is the median of these frequencies 
    freq(c) is the number of pixles of class c divided by the total number of pixels in images where c is present
    we weight each pixels by alpha_c
    Inputs: 
    logits is the output from the inference, which is the output of the decoder layers without softmax.
    labels: true label information 
    number_class: In the CamVid data set, it's 11 classes, or 12, because class 11 seems to be background? 
    frequency: is the frequency of each class
    Outputs:
    Loss
    Accuracy
    """
    label_flatten = tf.reshape(labels, [-1])
    label_onehot = tf.one_hot(label_flatten, depth=number_class)
    logits_reshape = tf.reshape(logits, [-1, number_class])
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=label_onehot, logits=logits_reshape,
                                                             pos_weight=frequency)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.summary.scalar('loss', cross_entropy_mean)
    correct_prediction = tf.equal(tf.argmax(logits_reshape, -1), label_flatten)
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
    tf.summary.scalar('accuracy', accuracy)

    return cross_entropy_mean, accuracy, tf.argmax(logits_reshape, -1)


def normal_loss(logits, labels, number_class):
    """
    Calculate the normal loss instead of median frequency balancing
    Inputs:
    logits, the output from decoder layers, without softmax, shape [Num_batch,height,width,Number_class]
    lables: the atual label information, shape [Num_batch,height,width,1]
    number_class:12
    Output:loss,and accuracy
    Using tf.nn.sparse_softmax_cross_entropy_with_logits assume that each pixel have and only have one specific
    label, instead of having a probability belongs to labels. Also assume that logits is not softmax, because it
    will conduct a softmax internal to be efficient, this is the reason that we don't do softmax in the inference 
    function!
    """
    label_flatten = tf.to_int64(tf.reshape(labels, [-1]))
    logits_reshape = tf.reshape(logits, [-1, number_class])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flatten, logits=logits_reshape,
                                                                   name='normal_cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.summary.scalar('loss', cross_entropy_mean)
    correct_prediction = tf.equal(tf.argmax(logits_reshape, -1), label_flatten)
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction))
    tf.summary.scalar('accuracy', accuracy)

    return cross_entropy_mean, accuracy, tf.argmax(logits_reshape, -1)


def per_class_acc(predictions, label_tensor, num_class):
    """
    This function is copied from "Implement slightly different segnet on tensorflow"
    """
    labels = label_tensor

    size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    acc_total = np.diag(hist).sum() / hist.sum()
    print('accuracy = %f' % np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('mean IU  = %f' % np.nanmean(iu))
    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("    class # %d accuracy = %f " % (ii, acc))
        # added from here
        if ii == 0:
            sidewalk_acc = acc
    return sidewalk_acc

def fast_hist(a, b, n):
    """
    This function is copied from "Implement slightly different segnet on tensorflow"
    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def get_hist(predictions, labels):
    """
    This function is copied from "Implement slightly different segnet on tensorflow"
    """
    num_class = predictions.shape[3]  # becomes 2 for aerial - correct
    batch_size = predictions.shape[0]
    hist = np.zeros((num_class, num_class))
    for i in range(batch_size):
        hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    return hist

def print_hist_summary(hist):
    """
    This function is copied from "Implement slightly different segnet on tensorflow"
    """
    acc_total = np.diag(hist).sum() / hist.sum()
    print('accuracy = %f' % np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('mean IU  = %f' % np.nanmean(iu))
    for ii in range(hist.shape[0]):
        if float(hist.sum(1)[ii]) == 0:
            acc = 0.0
        else:
            acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
        print("    class # %d accuracy = %f " % (ii, acc))

def train_op(total_loss, opt):
    """
    Input:
    total_loss: The loss 
    Learning_Rate: learning_rate for different optimization algorithm, for AdamOptimizer 0.001, for SGD 0.1
    global_step: global_step is used to track how many batches had been passed. In the training process, the intial
    value for global_step is 0 (tf.variable(0,trainable=False)), then after one batch of images passed, the loss is
    passed into the optimizer to update the weight, then the global step increased by one. Number of batches seen
    by the graph.. Reference: https://stackoverflow.com/questions/41166681/what-does-tensorflow-global-step-mean
    FLAG: utilized to denote which optimization method are we using, because for segnet, we can easily use Adam, but
    for segnet bayes, from the paper it says SGD will be more helpful to learn. 
    Output
    The train_op
    """
    global_step = tf.Variable(0, trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        if (opt == "ADAM"):
            optimizer = tf.train.AdamOptimizer(0.001, epsilon=0.0001)
            print("Running with Adam Optimizer with learning rate:", 0.001)
        elif (opt == "SGD"):
            base_learning_rate = 0.1
            learning_rate = tf.train.exponential_decay(base_learning_rate, global_step, decay_steps=1000, decay_rate=0.0005)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            print("Running with Gradient Descent Optimizer with learning rate", 0.1)
        else:
            raise ValueError("Optimizer is not recognized")

        grads = optimizer.compute_gradients(total_loss, var_list=tf.trainable_variables())
        training_op = optimizer.apply_gradients(grads, global_step=global_step)

    return training_op, global_step

def MAX_VOTE(pred,prob,NUM_CLASS):
    """
    logit: the shape should be [NUM_SAMPLES,Batch_size, image_h,image_w,NUM_CLASS]
    pred: the shape should be[NUM_SAMPLES,NUM_PIXELS]
    label: the real label information for each image
    prob: the probability, the shape should be [NUM_SAMPLES,image_h,image_w,NUM_CLASS]
    Output:
    logit: which will feed into the Normal loss function to calculate loss and also accuracy!
    """

    image_h = 360
    image_w = 480
    NUM_SAMPLES = np.shape(pred)[0]
    #transpose the prediction to be [NUM_PIXELS,NUM_SAMPLES]
    pred_tot = np.transpose(pred)
    prob_re = np.reshape(prob,[NUM_SAMPLES,image_h*image_w,NUM_CLASS])
    prediction = []
    variance_final = []
    step = 0
    for i in pred_tot:
        
        value = np.bincount(i,minlength = NUM_CLASS)
        value_max = np.argmax(value)
        #indices = [k for k,j in enumerate(i) if j == value_max]
        indices = np.where(i == value_max)[0]
        prediction.append(value_max)
        variance_final.append(np.var(prob_re[indices,step,:],axis = 0))
        step = step+1
        
     
    return variance_final,prediction
        
def var_calculate(pred,prob_variance):
    """
    Inputs: 
    pred: predicted label, shape is [NUM_PIXEL,1]
    prob_variance: the total variance for 12 classes wrt each pixel, prob_variance shape [image_h,image_w,12]
    Output:
    var_one: corresponding variance in terms of the "optimal" label
    """
        
    image_h = 360
    image_w = 480
    NUM_CLASS = np.shape(prob_variance)[-1]
    var_sep = [] #var_sep is the corresponding variance if this pixel choose label k
    length_cur = 0 #length_cur represent how many pixels has been read for one images
    for row in np.reshape(prob_variance,[image_h*image_w,NUM_CLASS]):
        temp = row[pred[length_cur]]
        length_cur += 1
        var_sep.append(temp)
    var_one = np.reshape(var_sep,[image_h,image_w]) #var_one is the corresponding variance in terms of the "optimal" label
    
    return var_one

def var_calculate_gta5(pred,prob_variance):
    """
    Inputs: 
    pred: predicted label, shape is [NUM_PIXEL,1]
    prob_variance: the total variance for 12 classes wrt each pixel, prob_variance shape [image_h,image_w,12]
    Output:
    var_one: corresponding variance in terms of the "optimal" label
    """
        
    image_h = 338
    image_w = 600
    NUM_CLASS = np.shape(prob_variance)[-1]
    var_sep = [] #var_sep is the corresponding variance if this pixel choose label k
    length_cur = 0 #length_cur represent how many pixels has been read for one images
    for row in np.reshape(prob_variance,[image_h*image_w,NUM_CLASS]):
        temp = row[pred[length_cur]]
        length_cur += 1
        var_sep.append(temp)
    var_one = np.reshape(var_sep,[image_h,image_w]) #var_one is the corresponding variance in terms of the "optimal" label
    
    return var_one