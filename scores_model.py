import tensorflow as tf
from models.score.score import Scores

# real_train_hdf5 = '/home/adalberto/Documents/Cancer_TMA_Generative/data_model_output/Evaluation/Real/vgh_nki/he/h224_w224_n3/hdf5_vgh_nki_he_features_train_real.h5'
# real_test_hdf5 = '/home/adalberto/Documents/Cancer_TMA_Generative/data_model_output/Evaluation/Real/vgh_nki/he/h224_w224_n3/hdf5_vgh_nki_he_features_test_real.h5'
    
# with tf.Graph().as_default():
#     scores = Scores(real_train_hdf5, real_test_hdf5, 'Real Train', 'Real Test', k=1)
#     scores.run_scores()
#     scores.report_scores()


# percents = [90,70,50,30,10]
# for perc in percents:
#     cont_real_train_hdf5 = '/home/adalberto/Documents/Cancer_TMA_Generative/data_model_output/Evaluation/Real/vgh_nki_stanford_contaminated_%sperc/he_cd137_contaminated_%sperc/h224_w224_n3/hdf5_vgh_nki_stanford_contaminated_%sperc_he_cd137_contaminated_%sperc_%sperc_features_train_real.h5' % (perc,perc,perc,perc,perc)
#     cont_real_test_hdf5 = '/home/adalberto/Documents/Cancer_TMA_Generative/data_model_output/Evaluation/Real/vgh_nki_stanford_contaminated_%sperc/he_cd137_contaminated_%sperc/h224_w224_n3/hdf5_vgh_nki_stanford_contaminated_%sperc_he_cd137_contaminated_%sperc_%sperc_features_test_real.h5' % (perc,perc,perc,perc,perc)
#     with tf.Graph().as_default():
# 	    scores = Scores(real_train_hdf5, cont_real_train_hdf5, 'Real Train', 'Cont Real Train %s' % perc, k=1)
# 	    scores.run_scores()
# 	    scores.report_scores()



nki_vgh_new_train = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/new/h224_w224_n3//hdf5_nki_vgh_he_features_train_real.h5'
nki_vgh_new_test = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/new/h224_w224_n3/hdf5_nki_vgh_he_features_test_real.h5'

gen_hdf5 = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/BigGAN/vgh_nki/he/unconditional/h224_w224_n3_50_Epoch/hdf5_vgh_nki_he_features_BigGAN.h5'

print('Study on VGH_NKI')
with tf.Graph().as_default():
    scores = Scores(nki_vgh_new_train, nki_vgh_new_test, 'Real Train VGH_NKI', 'Real Test VGH_NKI', k=1, display=False)
    scores.run_scores()
    scores.report_scores()

with tf.Graph().as_default():
    scores = Scores(nki_vgh_new_train, gen_hdf5, 'Real Train', 'BigGAN', k=1, display=True)
    scores.run_scores()

with tf.Graph().as_default():
    scores = Scores(nki_vgh_new_test, gen_hdf5, 'Real Test', 'BigGAN', k=1, display=True)
    scores.run_scores()