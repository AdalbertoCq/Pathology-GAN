import tensorflow as tf
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def l2_loss(weight_penalty):
    weights = [var for var in tf.trainable_variables() if 'kernel' in var.name]
    return weight_penalty * tf.add_n([tf.nn.l2_loss(w) for w in weights])


def cross_entropy(logits, labels):
    cross_entropy_ = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(cross_entropy_)


# Deprecated.
def job_id():
    try:
        return os.environ['PBS_JOBID'].split('.')[0]
    except KeyError:
        return '0'


def template(vars_job_id=None):
    job_id_ = vars_job_id if vars_job_id else job_id()
    return os.path.join(os.getcwd(), 'run', job_id_)


def log(vars_job_id):
    return os.path.join(template(vars_job_id), 'tensorboard')


def ckpt(vars_job_id):
    return os.path.join(template(vars_job_id), 'ckpt', 'session')


def img_logits():
    return os.path.join(template(vars_job_id), 'logits')


def variables(vars_job_id):
    return os.path.join(template(vars_job_id), 'ckpt', 'session')


def progress(vars_job_id=None):
    return os.path.join(template(vars_job_id), 'progress')


def store_progress(runner, *args):
    if args[0] == 'training':
        runner.saver.save(runner.session, ckpt())
    if args:
        with open(progress(), 'wb') as f:
            pickle.dump(args, f)
    print('checkpoint saved')


# load previous session.
def load_progress(runner):
    job_id_ = runner.vars_job_id
    runner.saver.restore(runner.session, variables(job_id_))
    print('Session restored: %s' % job_id_)
    try:
        with open(progress(job_id_), 'rb') as f:
            payload = pickle.load(f)
    except IOError:
        payload = None
    print('ckeckpoint loaded from job', job_id_, bool(payload))
    return payload


# Runs PCA on encodings, and plots on a two dim.
def run_tsne(vae, number_points, perplexity, learning_step, iterations, name):
    point_counter = 0
    enc_arr = []
    labels_t = []
    # Gets encodings for 30 batches.
    for features, labels, _ in vae.data.training:
        point_counter += features.shape[0]
        feed_dict = {vae.x: features, vae.train_phase: False}
        means = vae.session.run(vae.mean_z_given_xi, feed_dict)
        enc_arr.append(means)
        labels_t.append(labels)
        if point_counter > number_points:
            break

    means = np.reshape(np.array(enc_arr), (point_counter, vae.enc_shape[1]))
    labels_t = np.reshape(np.array(labels_t), (point_counter, 1))
    labels_t = np.rint(labels_t)

    tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=iterations, learning_rate=learning_step)
    tsne_results = tsne.fit_transform(means)
    x_tsne = tsne_results[:, 0]
    y_tsne = tsne_results[:, 1]
    labels = labels_t[:, 0]
    plt.scatter(x_tsne, y_tsne, c=labels)
    plt.savefig('run/%s/img/%s.png' % (vae.vars_job_id, name))
    plt.close()


'''
Code to dump out trainable parameters in a model.
'''
def print_depth(master, depth):
    a = [el for el in master.keys() if ':' not in el]
    if len(a) != 0:
        for el in master:
            print('%s%s:' % (depth, el))
            depth_ = depth + '\t'
            print_depth(master[el], depth_)
    else:
        for el in master:
            base = '%s%s - ' % (depth, el)
            tail = ''
            for e in master[el]:
                tail += '%s: %s ' % (e, master[el][e])
            print(base + tail)


def print_param(master):
    depth = ''
    print_depth(master, depth)


def get_model_train_parms():
    # Check number of trainable parameters.
    master = dict()
    total_parameters = 0
    for variable in tf.trainable_variables():
        hier = variable.name.split('/')
        current_level = master
        for i, part in enumerate(hier):
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
            if i == len(hier) - 1:
                current_level['shape'] = variable.get_shape()
                variable_parameters = 1
                for dim in current_level['shape']:
                    variable_parameters *= dim.value
                current_level['param'] = variable_parameters
                total_parameters += variable_parameters

    print('Total Parameters: ', total_parameters)
    print_param(master)