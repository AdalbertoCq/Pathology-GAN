library(CRImage)

# Classifier definition.
t = system.file("extdata", "trainingData.txt", package="CRImage")
trainingData = read.table(t, header=TRUE)
classifierValues = createClassifier(trainingData) 
classifier = classifierValues[[1]]


# GENERATED
'Rscript --vanilla sillyScript.R iris.txt out.txt'
args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} 

base_path = args[1]
samples = args[2]

file_name_base = paste(base_path, 'generated_images/gen_', sep='')
print(file_name_base)
c_cells = vector('list', length=samples)
for (index in seq(1, samples, by=1)) {
	file_name = paste(file_name_base, index-1, sep='')
	file_name = paste(file_name, '.png', sep='')
	cellularity = calculateCellularity(classifier=classifier, filename=file_name, KS=TRUE, maxShape=800, minShape=40, failureRegion=2000, classifyStructures=FALSE, cancerIdentifier="c", numDensityWindows=2, colors=c("green","red"))
	c_cells[[index]] = cellularity[[1]]
}

c_cells_path = paste(base_path, 'crimage_train.txt', sep='')

file.remove(c_cells_path)

lapply(c_cells, write, c_cells_path, append=TRUE)


