import tensorflow as tf
from models.score.score import Scores

# Combine
vgh_nki_quarentine_train = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/vgh_nki/he/quarentine/h224_w224_n3/hdf5_vgh_nki_he_features_train_real.h5'
vgh_nki_quarentine_test = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/vgh_nki/he/quarentine/h224_w224_n3/hdf5_vgh_nki_he_features_test_real.h5'

vgh_nki_new_train = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/vgh_nki/he/new/h224_w224_n3/hdf5_vgh_nki_he_features_train_real.h5'
vgh_nki_new_test = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/vgh_nki/he/new/h224_w224_n3/hdf5_vgh_nki_he_features_test_real.h5'


# By Method
nki_vgh_quarentine_train = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/quarentine/h224_w224_n3/hdf5_nki_vgh_he_features_train_real.h5'
nki_vgh_quarentine_test = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/quarentine/h224_w224_n3/hdf5_nki_vgh_he_features_test_real.h5'

nki_vgh_new_train = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/new/h224_w224_n3//hdf5_nki_vgh_he_features_train_real.h5'
nki_vgh_new_test = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/new/h224_w224_n3/hdf5_nki_vgh_he_features_test_real.h5'


print('NKI-VGH Quarentine.')
with tf.Graph().as_default():
    scores = Scores(nki_vgh_quarentine_train, nki_vgh_quarentine_test, 'Train NKI-VGH Quarentine', 'Test NKI-VGH Quarentine', k=1, display=False)
    scores.run_scores()
    scores.report_scores()

print('NKI-VGH New.')
with tf.Graph().as_default():
    scores = Scores(nki_vgh_new_train, nki_vgh_new_test, 'Train NKI-VGH New', 'Test NKI-VGH New', k=1, display=False)
    scores.run_scores()
    scores.report_scores()

print('NKI-VGH Train.')
with tf.Graph().as_default():
    scores = Scores(nki_vgh_quarentine_train, nki_vgh_new_train, 'Train NKI-VGH Quarentine', 'Train NKI-VGH Quarentine', k=1, display=False)
    scores.run_scores()
    scores.report_scores()

print('NKI-VGH Test.')
with tf.Graph().as_default():
    scores = Scores(nki_vgh_quarentine_test, nki_vgh_new_test, 'Test NKI-VGH New', 'Test NKI-VGH New', k=1, display=False)
    scores.run_scores()
    scores.report_scores()

print('Crossed 1.')
with tf.Graph().as_default():
    scores = Scores(nki_vgh_quarentine_train, nki_vgh_new_test, 'Train NKI-VGH Quarentine', 'Test NKI-VGH New', k=1, display=False)
    scores.run_scores()
    scores.report_scores()

print('Crossed 2.')
with tf.Graph().as_default():
    scores = Scores(nki_vgh_new_train, nki_vgh_quarentine_test, 'Train NKI-VGH New', 'Test NKI-VGH Quarentine', k=1, display=False)
    scores.run_scores()
    scores.report_scores()