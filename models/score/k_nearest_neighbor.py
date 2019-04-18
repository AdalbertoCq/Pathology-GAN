import tensorflow as tf
import numpy as np
from models.score.utils import *


def k_nearest_neighbor(x, y, k):
    x_samples = x.shape.as_list()[0]
    y_samples = y.shape.as_list()[0]

    xx_d = euclidean_distance(x, x)
    yy_d = euclidean_distance(y, y)
    xy_d = euclidean_distance(x, y)

    labels = tf.concat([tf.ones((x_samples,1)), tf.zeros((y_samples,1))], axis=0)

    x_dist = tf.concat([xx_d, xy_d], axis=-1)
    y_dist = tf.concat([tf.transpose(xy_d), yy_d], axis=-1)
    total_dist = tf.concat([x_dist, y_dist], axis=0)
    '''
    x1x1   x1x2   ... x1x100   | x1y1   x1xy2   ... x1y200
    ...						   |   				  ...
    x100x1 x100x2 ... x100x100 | x100y1 x100xy2 ... x100y200
    ________________________________________________________
    y1x1   y1x2   ... y1x100   | y1y1   y1xy2   ... y1y200
    ...						   |  				  ...
    y100x1 y100x2 ... y100x100 | y100y1 y1xy2   ... y100y100
    ...						   |  				  ...
    y200x1 y200x2 ... y200x100 | y200y1 y200xy2 ... y200y200

    Diagonals of this tensor are the distance for the vector with itself.
    '''
    total_dist = tf.sqrt(tf.abs(total_dist))
    inf_eye = tf.eye(total_dist.shape.as_list()[0])*1e+7

    #All element positive now, no smallest elements functions.
    all_dist = tf.math.add(inf_eye, total_dist)
    neg_all_dist = tf.negative(all_dist)
    values, indices = tf.math.top_k(input=neg_all_dist, k=k, sorted=True)
    values = tf.negative(values)

    num_vectors, k_ = indices.shape.as_list()
    addition_labels = np.zeros((x_samples+y_samples, 1))

    for i in range(num_vectors):
        add = 0
        for j in range(k):
            add += labels[indices[i, j]]
        addition_labels[i] = add

    reference = (k/2.)* np.ones((x_samples+y_samples, 1))
    prediction = tf.cast(tf.greater(addition_labels, reference), tf.float32)
    
    true_positive = tf.reduce_sum(prediction*labels)
    false_positive = tf.reduce_sum(prediction*(1-labels))
    true_negative = tf.reduce_sum((1-prediction)*labels)
    false_negative = tf.reduce_sum((1-prediction)*(1-labels))
    
    precision = true_positive/(true_positive+false_positive)
    recall = true_positive/(true_positive+false_negative)
    
    accuracy_true = true_positive/(true_positive+false_negative)
    accuracy_false = true_negative/(true_negative+false_positive)
    
    matched = tf.cast(tf.equal(labels, prediction), tf.float32)
    accuracy_x = tf.reduce_mean(matched[:x_samples, :])
    accuracy_y = tf.reduce_mean(matched[x_samples:, :])
    accuracy = tf.reduce_mean(matched)

    return accuracy_x, accuracy_y, accuracy