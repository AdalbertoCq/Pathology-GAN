import tensorflow as tf
import numpy as np
import models.utils as utils
from models.runner import Runner
from preparation.utils import store_data
from models.cnn.densenet.densenet import DenseNet


class Classifier(Runner):  # (10,) or (5, 10, 15)
    def __init__(self, thresholds=(10,), weight_penalty=1e-4, keep_prob=0.8):
        super().__init__()
        self.data.training.set_thresholds(thresholds)
        self.data.test.set_thresholds(thresholds)
        self.keep_prob = keep_prob
        self.classes = len(thresholds) + 1

        self.x = tf.placeholder(tf.float32, [None] + self.data.training.shape[1:])
        self.Y = tf.placeholder(tf.float32)
        self.train_phase = tf.placeholder(tf.bool)
        self.y = DenseNet(classes=self.classes, keep_prob=self.keep_prob).build_graph(self.x, self.train_phase)
        self.cross_entropy = utils.cross_entropy(self.y, self.Y)
        self.l2_loss = utils.l2_loss(weight_penalty)
        self.loss = self.cross_entropy + self.l2_loss
        optimizer = tf.train.AdamOptimizer()
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = optimizer.minimize(self.loss)
        self.logits = tf.nn.softmax(self.y)
        self.prediction = tf.argmax(self.logits, 1)
        self.actual = tf.argmax(self.Y, 1)
        self.correct_prediction = tf.cast(tf.equal(self.prediction, self.actual), tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_prediction)
        self.summary_op = None
        self.summary_ops = None
        self.summarize()

    def summarize(self):
        a = tf.summary.scalar('entropy', self.cross_entropy)
        b = tf.summary.scalar('l2', self.l2_loss)
        c = tf.summary.scalar('loss', self.loss)
        d = tf.summary.scalar('training accuracy', self.accuracy)
        self.summary_op = tf.summary.merge([a, b, c, d])
        self.summary_ops = [
            tf.summary.merge([tf.summary.scalar('validation loss', self.loss),
                              tf.summary.scalar('validation accuracy', self.accuracy)]),
            tf.summary.merge([tf.summary.scalar('training loss in test mode)', self.loss),
                              tf.summary.scalar('training accuracy in test mode)', self.accuracy)]),
            tf.summary.merge([tf.summary.scalar('validation loss in train mode', self.loss),
                              tf.summary.scalar('validation accuracy in train mode', self.accuracy)])
        ]

    def train_substep(self, train_batch):
        batches = [None, train_batch, None]
        phases = [False, False, True]
        try:
            test_batch = next(self.data.test)
            batches[0] = test_batch
            batches[2] = test_batch
        except StopIteration:
            pass
        for i, batch in enumerate(batches):
            if batch is None:
                continue
            features, labels, _ = batch
            feed_dict = {self.x: features, self.Y: labels, self.train_phase: phases[i]}
            summary = self.session.run(self.summary_ops[i], feed_dict)
            self.writer.add_summary(summary, self.batch_counter)

    def train(self):
        self.reset()
        for features, labels, _ in self.data.training:
            feed_dict = {self.x: features, self.Y: labels, self.train_phase: True}
            _, summary = self.session.run([self.train_op, self.summary_op], feed_dict)
            self.writer.add_summary(summary, self.batch_counter)
            self.train_substep((features, labels, _))
            self.batch_counter += 1
            print(self.batch_counter)
            if self.batch_limit is not None and self.batch_counter >= self.batch_limit:
                break

    def test(self, training=False, suffix=''):
        self.reset()
        self.data.training.set_batch_size(self.batch_size)
        self.data.test.set_batch_size(self.batch_size)
        for i, dataset in enumerate([self.data.test]):
            img_logits, indices = [], []
            real_logits = np.ones([len(dataset.set), self.classes], dtype=np.float32) * -1
            test_counter = 0
            for features, labels, idxs in dataset:
                feed_dict = {self.x: features, self.Y: labels, self.train_phase: training}
                logits_ = self.session.run(self.logits, feed_dict)
                real_logits[idxs] = labels
                img_logits.append(logits_)
                indices.append(idxs)
                test_counter += 1
                print(test_counter)
                if self.batch_limit is not None and test_counter >= self.batch_limit:
                    break
            if i == 0:
                img_logits = np.concatenate(img_logits)
                indices = np.concatenate(indices)
                sorted_ = np.argsort(indices)
                img_logits = img_logits[sorted_]
                img_logits = np.reshape(img_logits, [len(dataset.set), -1, self.classes])
                store_data((img_logits, real_logits), utils.img_logits() + suffix)
                print('"logits%s" saved' % suffix)

    def run(self):
        for epoch in range(self.epochs):
            self.train()
            utils.store_progress(self, 'training')
            self.test(suffix='_%d' % epoch)
            self.test(training=True, suffix='_train_%d' % epoch)
