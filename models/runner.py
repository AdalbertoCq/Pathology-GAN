import tensorflow as tf
import models.utils as utils
from preparation.data import Data


# Super class Runner.
class Runner:
    def __init__(self, patch_h=28, patch_w=28, epochs=1, batch_size=50, lr=1e-3, vars_job_id=None, batch_limit=None, restore=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.vars_job_id = vars_job_id
        self.data = Data(patch_h, patch_w)
        self.batch_limit = batch_limit
        self.restore = restore

        # Parameters for training. 
        epoch = 0
        n_train = len(self.data.training.set)
        n_test = len(self.data.test.set)
        k_train = self.data.training.shape[0]
        k_test = self.data.test.shape[0] and 0  # 'and 0' for supervised learning
        self.data.training.set_batch_size(self.batch_size)
        self.data.test.set_batch_size(round(self.batch_size / n_train * n_test))  # for supervised learning
        # self.data.test.set_batch_size(self.batch_size)  # for unsupervised learning
        self.batch_counter = (k_train + k_test) // self.batch_size * epoch
        self.progress = None

    # Function for 'with' statement.
    def __enter__(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(utils.log(self.vars_job_id), graph_def=self.session.graph_def)
        self.saver = tf.train.Saver()

        # Restore previous session if wanted.
        if self.restore:
            self.progress = utils.load_progress(self)
        return self

    def __exit__(self, *args):
        self.session.close()
        self.writer.close()

    def reset(self):
        self.data.training.reset()
        self.data.test.reset()
