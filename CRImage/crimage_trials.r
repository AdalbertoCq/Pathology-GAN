library(CRImage)

# Classifier definition.
t = system.file("extdata", "trainingData.txt", package="CRImage")
trainingData = read.table(t, header=TRUE)
classifierValues = createClassifier(trainingData) 
classifier = classifierValues[[1]]

# REAL
paths = vector('list', length=5)
paths[[1]] = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/new/h224_w224_n3/'
paths[[2]] = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/h224_w224_n3/er_positive/'
paths[[3]] = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/h224_w224_n3/er_negative/'
paths[[4]] = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/h224_w224_n3/survival_positive/'
paths[[5]] = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/real/nki_vgh/he/h224_w224_n3/survival_negative/'

for (base_path in paths) {
	print(base_path)
	file_name_base = paste(base_path, 'img_train/real_train_', sep='')
	c_cells = vector('list', length=5000)
	for (index in seq(1, 5000, by=1)) {
		file_name = paste(file_name_base, index-1, sep='')
		file_name = paste(file_name, '.png', sep='')
		cellularity = calculateCellularity(classifier=classifier, filename=file_name, KS=TRUE, maxShape=800, minShape=40, failureRegion=2000, classifyStructures=FALSE, cancerIdentifier="c", numDensityWindows=2, colors=c("green","red"))
		c_cells[[index]] = cellularity[[1]]
	}

	c_cells_path = paste(base_path, 'crimage_train.txt', sep='')

	file.remove(c_cells_path)

	lapply(c_cells, write, c_cells_path, append=TRUE)

}


# GENERATED
paths = vector('list', length=5)
paths[[1]] = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/BigGAN/vgh_nki/he/unconditional/h224_w224_n3_50_Epoch/'
paths[[2]] = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/BigGAN/vgh_nki/he/conditional_ER/h224_w224_n3_conditonal_ER_positive_45_Epochs/'
paths[[3]] = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/BigGAN/vgh_nki/he/conditional_ER/h224_w224_n3_conditonal_ER_negative_45_Epochs/'
paths[[4]] = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/BigGAN/vgh_nki/he/conditional_survival/h224_w224_n3_conditonal_survival_positive_60_Epochs/'
paths[[5]] = '/media/adalberto/Disk2/Cancer_TMA_Generative/evaluation/BigGAN/vgh_nki/he/conditional_survival/h224_w224_n3_conditonal_survival_negative_60_Epochs/'

for (base_path in paths) {
	print(base_path)
	file_name_base = paste(base_path, 'generated_images/gen_', sep='')
	c_cells = vector('list', length=5000)
	for (index in seq(1, 5000, by=1)) {
		file_name = paste(file_name_base, index-1, sep='')
		file_name = paste(file_name, '.png', sep='')
		cellularity = calculateCellularity(classifier=classifier, filename=file_name, KS=TRUE, maxShape=800, minShape=40, failureRegion=2000, classifyStructures=FALSE, cancerIdentifier="c", numDensityWindows=2, colors=c("green","red"))
		c_cells[[index]] = cellularity[[1]]
	}

	c_cells_path = paste(base_path, 'crimage_train.txt', sep='')

	file.remove(c_cells_path)

	lapply(c_cells, write, c_cells_path, append=TRUE)

}
