from models.score.score import Scores
import tensorflow as tf
import matplotlib.pyplot as plt
import time


real_train_hdf5 = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/new/h224_w224_n3/hdf5_nki_vgh_he_features_train_real.h5'
real_test_hdf5 = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/new/h224_w224_n3/hdf5_nki_vgh_he_features_test_real.h5'
gen_hdf5 = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/BigGAN/vgh_nki/he/unconditional/h224_w224_n3_50_Epoch/hdf5_vgh_nki_he_features_BigGAN.h5'

comp_img_path = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/scores_output/complete.png'


references = dict()
references[real_train_hdf5] = 'Real Train'
references[real_test_hdf5] = 'Real Test'
references[gen_hdf5] = 'BigGAN'

complete_scores = dict()
for marker in ['cd137', 'cathepsin_l']:
	complete_scores[marker] = dict()
	for t in ['train', 'test']:
		complete_scores[marker][t] = list()
		scores_total = list()
		figure_path = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/scores_output/%s_%s_comp.png' % (marker, t)
		percents = list(range(100, -10, -10))
		scores_train = list()
		for ref in references:
		    ref_list = list()
		    print(ref)
		    for perc in percents:
		        cont_real_t_hdf5 = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/contaminated/%s/nki_vgh_stanford_contaminated_%sperc/he_%s_contaminated_%sperc/h224_w224_n3/hdf5_nki_vgh_stanford_contaminated_%sperc_he_%s_contaminated_%sperc_%sperc_features_%s_real.h5' % (marker,perc,marker,perc,perc,marker,perc,perc,t)
		        print('\t', cont_real_t_hdf5)
		        with tf.Graph().as_default():
		            scores = Scores(ref, cont_real_t_hdf5, references[ref], 'Cont Real %s' % t, k=1)
		            scores.run_scores()
		            ref_list.append(scores)
		    scores_total.append(ref_list)
		    complete_scores[marker][t].append(ref_list)

		# Plot results.
		fig, big_axes = plt.subplots(figsize=(25.0, 25.0), nrows=len(scores_total), ncols=4, sharex=False, sharey=False) 
		for row, big_ax in enumerate(big_axes):
		    scores_ref = scores_total[row]
		    
		    for col, big_col in enumerate(big_ax):
		        score = scores_ref[col]
		        if col == 0:
		            big_col.plot(percents, [i.fid for i in scores_ref], label='Frechet Inception Distance')
		            big_col.set_xlabel('Contamination Percentage')
		            big_col.legend()
		        elif col == 1:
		            big_col.plot(percents, [i.kid for i in scores_ref], label='Kernel Inception Distance')
		            big_col.set_xlabel('Contamination Percentage')
		            big_col.legend()
		        elif col == 2:
		            big_col.plot(percents, [i.mmd for i in scores_ref], label='Minimum Mean Discrepancy')
		            big_col.set_xlabel('Contamination Percentage')
		            big_col.legend()
		        else:
		            big_col.plot(percents, [i.knn_x for i in scores_ref], label='1-Nearest Neighbor x')
		            big_col.plot(percents, [i.knn_y for i in scores_ref], label='1-Nearest Neighbor y')
		            big_col.plot(percents, [i.knn for i in scores_ref], label='1-Nearest Neighbor')
		            big_col.set_xlabel('Contamination Percentage')
		            big_col.legend()

		for col, ax in enumerate(big_axes[0]):
		    col_name = scores_total[row][0].title.split('-')[1]
		    ax.set_title(col_name, size='large')

		for row, ax in enumerate(big_axes[:,0]):
		    row_name = scores_total[row][0].title.split('-')[0]
		    ax.set_ylabel(row_name, size='large')

		fig.set_facecolor('w')
		plt.tight_layout()

		plt.savefig(figure_path)
		time.sleep(2)
		plt.close(fig)


# Plot results.
fig, big_axes = plt.subplots(figsize=(25.0, 25.0), nrows=len(scores_total), ncols=4, sharex=False, sharey=False) 
for row, big_ax in enumerate(big_axes):
	for col, big_col in enumerate(big_ax):
		for marker in ['cd137', 'cathepsin_l']:
			for t in ['train', 'test']:
				score = complete_scores[marker][t][row]
				if col == 0:
					big_col.plot(percents, [i.fid for i in score], label='%s_%s' % (marker, t))
					# big_col.set_title('Frechet Inception Distance')
					big_col.set_xlabel('Contamination Percentage')
					big_col.legend()
				elif col == 1:
					big_col.plot(percents, [i.kid for i in score], label='%s_%s' % (marker, t))
					# big_col.set_title('Kernel Inception Distance')
					big_col.set_xlabel('Contamination Percentage')
					big_col.legend()
				elif col == 2:
					big_col.plot(percents, [i.mmd for i in score], label='%s_%s' % (marker, t))
					# big_col.set_title('Maximum Mean Discrepancy')
					big_col.set_xlabel('Contamination Percentage')
					big_col.legend()
				else:
					# big_col.plot(percents, [i.knn_x for i in score], label='1-Nearest Neighbor x')
					# big_col.plot(percents, [i.knn_y for i in score], label='1-Nearest Neighbor y')
					big_col.plot(percents, [i.knn for i in score], label='%s_%s' % (marker, t))
					# big_col.set_title('1-Nearest Neighbor')
					big_col.set_xlabel('Contamination Percentage')
					big_col.legend()

for col, ax in enumerate(big_axes[0]):
#     col_name = scores_total[row][0].title.split('-')[1]
#     ax.set_title(col_name, size='large')
	if col == 0:
		big_col.set_title('Frechet Inception Distance')
	elif col == 1:
		big_col.set_title('Kernel Inception Distance')
	elif col == 2:
		big_col.set_title('Maximum Mean Discrepancy')
	else:
		big_col.set_title('1-Nearest Neighbor')


for row, ax in enumerate(big_axes[:,0]):
    row_name = scores_total[row][0].title.split('-')[0]
    ax.set_ylabel(row_name, size='large')

fig.set_facecolor('w')
plt.tight_layout()

plt.savefig(comp_img_path)
time.sleep(2)
plt.close(comp_img_path)