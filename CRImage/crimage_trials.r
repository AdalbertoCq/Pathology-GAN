library(CRImage)
# library(imager)


print('Opening image...')
f = system.file("extdata", "exImg2.jpg", package="CRImage")
f = "/Users/adalbertoclaudioquiros/Documents/Code/UofG/PhD/Cancer_TMA_Generative/CRImage/test/174___1_116_10_1.jpg"

# Image is converted to grayscale and thresholded. Morphological opening is used to delete clutter and to smooth the shapes of cells.
# Watershed algorithm is used to separate clustered cells.
# maxShape = Maximum shape of cell nuclei. Segmented nuclei which exceed this value will be thresholded and segmented again.
# minShape = Minimum size of cell nuclei. Cell nuclei which fall below this value will be deleted.
# failureRegion = Defines when artifacts in the image should be deleted. Dark regions which exceed this value will be deleted.
# threshold = 
# numWindows = 
# segmentationValues => [1] Orignal Image, [2] Segmented image, [3] Features, which were calculated for the segmented objects
# print('Starting segmentation...')
# segmentationValues = segmentImage(filename=f, maxShape=800, minShape=40, failureRegion=2000, threshold='otsu', numWindows=4)
# display(segmentationValues[[3]]) 

# maxShape = Maximum shape of cell nuclei. Segmented nuclei which exceed this value will be thresholded and segmented again.
# minShape = Minimum size of cell nuclei. Cell nuclei which fall below this value will be deleted.
# failureRegion = Defines when artifacts in the image should be deleted. Dark regions which exceed this value will be deleted.
# traingValues => [1] Image in which every segmented cell is numbered, [2] table with features for every cell.
#
# To create a training set, class values for the cells have to be inserted in the column "class"
# f = system.file("extdata", "exImg.jpg", package="CRImage")
# print('Creating a Training Set...')
# trainingValues = createTrainingSet(filename=f, maxShape=800, minShape=40, failureRegion=2000)
# write.table(trainingData, file="/home/adalberto/Documents/Cancer_TMA_Generative/CRImage", sept="\t", rownames=F)


t = system.file("extdata", "trainingData.txt", package="CRImage")
trainingData = read.table(t, header=TRUE)
classifierValues = createClassifier(trainingData) 
classifier = classifierValues[[1]]

f = "/Users/adalbertoclaudioquiros/Documents/Code/UofG/PhD/Cancer_TMA_Generative/CRImage/test/fake.jpg"
# f = "/Users/adalbertoclaudioquiros/Documents/Code/UofG/PhD/Cancer_TMA_Generative/CRImage/test/real.jpg"
classValues=classifyCells(classifier, filename=f, KS=TRUE, maxShape=800, minShape=40, failureRegion=2000)
print(classValues)
# display(classValues[[1]])

cellularity = calculateCellularity(classifier=classifier, filename=f, KS=TRUE, maxShape=800, minShape=40, failureRegion=2000, classifyStructures=FALSE, 
									cancerIdentifier="1", numDensityWindows=2, colors=c("green", "red"))
print(cellularity)

display(cellularity[[2]])
display(cellularity[[3]])

