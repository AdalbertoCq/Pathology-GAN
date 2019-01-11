import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import itertools
import sys


job_id = sys.argv[-1]
path = os.path.join(os.sep, 'home', 'user', 'project', 'run', job_id, 'logits_0')
with open(path, 'rb') as f:
    data = pickle.load(f)
print(data[1].sum(0) / data[1].sum())


def flatten_data(data):
    logits, labels = data
    repeated = np.tile(labels, [1, logits.shape[1]])
    flattened = np.reshape(repeated, [-1, logits.shape[2]])
    reshaped = np.reshape(logits, [-1, 1, logits.shape[2]])
    return reshaped, flattened


def split_into_sets_randomly(data, ratio=0.5):
    logits, labels = data
    n = len(labels)
    k = round(n * ratio)
    val_idxs = np.random.choice(n, k, replace=False)
    mask = np.ones(n, np.bool)
    mask[val_idxs] = 0
    val_logits = logits[val_idxs]
    val_labels = labels[val_idxs]
    test_logits = logits[mask]
    test_labels = labels[mask]
    return (val_logits, val_labels), (test_logits, test_labels)


def get_predictions_for_images_weighted(logits, thresholds):
    mean_logits = logits.mean(1) - np.array(thresholds)
    return np.argmax(mean_logits, axis=1)


def get_predictions_for_images_democratic(logits, thresholds):
    predictions = np.argmax(logits - np.array(thresholds), axis=2)
    freqs = []
    for i in range(logits.shape[2]):
        freqs.append((predictions == i).sum(1))
    return np.stack(freqs).argmax(0)


def get_conf_matrix(logits, labels, thresholds, strategy=None):
    predictions = None
    if strategy == 'Democratic':
        predictions = get_predictions_for_images_democratic(logits, thresholds)
    elif strategy == 'Weighted' or strategy == 'Independent':
        predictions = get_predictions_for_images_weighted(logits, thresholds)
    real_labels = np.argmax(labels, axis=1)
    conf_matrices = np.zeros(labels.shape + (labels.shape[1],))
    idxs = np.arange(len(labels))
    conf_matrices[idxs, predictions, real_labels] = 1
    conf_matrix = conf_matrices.sum(0)
    return conf_matrix


def get_accuracy(logits, labels, thresholds, strategy=None):
    conf_matrix = get_conf_matrix(logits, labels, thresholds, strategy=strategy)
    return np.trace(conf_matrix / conf_matrix.sum())


def get_auc(points, strategy=None):
    points = sorted(points)
    prev_x, prev_y = None, None
    area = 0
    for x, y in points:
        if prev_x is None:
            prev_x, prev_y = x, y
            continue
        partition = (y + prev_y) / 2 * (x - prev_x)
        area += partition
        prev_x, prev_y = x, y
    plt.plot(*list(zip(*points)), label='%s (AUC %.2f)' % (strategy, area))
    return area


def find_best_threshold(data, strategy=None, roc=False):
    best_thres, best_acc = None, None
    THRES = 11
    points = []
    points4 = [[] for _ in range(4)]
    for config in itertools.combinations_with_replacement(range(THRES), len(data[1][0]) - 1):
        prev = 0
        thresholds = []
        for pos in config:
            thresholds.append(pos - prev)
            prev = pos
        thresholds.append(THRES - 1 - prev)
        thresholds = list(map(lambda x: x / (THRES - 1), thresholds))
        if debug:
            print(thresholds)
        conf_matrix = get_conf_matrix(data[0], data[1], thresholds, strategy=strategy)
        normalized = conf_matrix / conf_matrix.sum()

        if multi:
            for i in range(4):
                # 134 x 1000 x 4 and 134 x 4
                one_logits = data[0][:, :, i, None]
                rest_logits = 1 - one_logits
                logits = np.concatenate([one_logits, rest_logits], axis=2)
                one_labels = data[1][:, i, None]
                rest_labels = 1 - one_labels
                labels = np.concatenate([one_labels, rest_labels], axis=1)
                thresholds_ = [thresholds[i], 1 - thresholds[i]]
                conf_matrix_ = get_conf_matrix(logits, labels, thresholds_, strategy=strategy)
                normalized_ = conf_matrix_ / conf_matrix_.sum()
                tp, fp = normalized_[0, :]
                fn, tn = normalized_[1, :]
                tpr = tp / (tp + fn)
                fpr = fp / (fp + tn)
                points4[i].append((fpr, tpr))
        else:
            tp, fp = normalized[0, :]
            fn, tn = normalized[1, :]
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            points.append((fpr, tpr))

        acc = np.trace(normalized)
        if best_acc is None or acc > best_acc:
            best_acc = acc
            best_thres = thresholds
    if roc:
        if multi:
            for i, points in enumerate(points4):
                get_auc(points, strategy='Class-%d vs All' % (i + 1))
        else:
            get_auc(points, strategy=strategy)
    return best_thres, best_acc


def plot_rocs_of_different_strategies():
    # comment out 2 strategies for 4-class classification results and get roc curves separately
    find_best_threshold(flatten_data(data), strategy='Independent', roc=True)
    find_best_threshold(data, strategy='Weighted', roc=True)
    find_best_threshold(data, strategy='Democratic', roc=True)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random (AUC 0.50)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=[0.5, 0.1])
    plt.show()


def display_conf_matrix(conf_matrix):
    print(conf_matrix)
    plt.imshow(conf_matrix, cmap='gray')
    ax = plt.gca()
    ax.xaxis.set_label_position('top')
    plt.xlabel('Actual class')
    plt.ylabel('Predicted class')
    for y in range(len(conf_matrix)):
        for x in range(len(conf_matrix)):
            plt.annotate('%.1f%%' % (conf_matrix[y][x] * 100), xy=(x, y), bbox={'fc': 'white'},
                         horizontalalignment='center',
                         verticalalignment='center')
    plt.tick_params(
        axis='x',
        which='both',
        bottom='off',
        labelbottom='off'
    )
    plt.tick_params(
        axis='y',
        which='both',
        left='off',
        labelleft='off'
    )
    # plt.colorbar()
    plt.show()


def evaluate_strategy(data, strategy, runs=10):  # runs=1000
    thresholds = []
    for _ in range(runs):
        if debug:
            print(_)
        val_data, test_data = split_into_sets_randomly(data)
        thres, best_acc = find_best_threshold(val_data, strategy=strategy)
        thresholds.append(thres)
    mean_thres = np.array(thresholds).mean(0)
    conf_matrices = []
    accs = []
    for _ in range(runs):
        if debug:
            print(_)
        _, test_data = split_into_sets_randomly(data)
        conf_matrix = get_conf_matrix(test_data[0], test_data[1], mean_thres, strategy=strategy)
        conf_matrices.append(conf_matrix)
        acc = np.trace(conf_matrix / conf_matrix.sum())
        accs.append(acc)
    mean_conf_matrix = np.array(conf_matrices).mean(0)
    mean_conf_matrix /= mean_conf_matrix.sum()
    mean_acc = np.array(accs).mean()
    std_acc = np.array(accs).std()
    display_conf_matrix(mean_conf_matrix)
    print(strategy, 'mean_acc:', mean_acc, 'std_acc:', std_acc, 'thres:', mean_thres)
    return accs


if __name__ == '__main__':
    debug = True
    multi = False  # True for 4-class classification results
    plot_rocs_of_different_strategies()
    # exit()
    ind_accs = evaluate_strategy(flatten_data(data), 'Independent')
    demo_accs = evaluate_strategy(data, 'Democratic')
    w_accs = evaluate_strategy(data, 'Weighted')
