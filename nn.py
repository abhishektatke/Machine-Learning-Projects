#To run this code, rename nn.txt to nn.py 
#place ubyte files at the same location as the nn.py file
#launch python nn.py

import os
import random
import numpy as np
import sklearn
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

#Global variables
inputSize = 785
outputSize = 10
trainSetSize = 60000
testSetSize = 10000
NumnerOfHiddenUnits = [20, 50, 100]
momentumValues = [0, 0.25, 0.5]
learningRate = 0.1

# function to convert the ubyte files to csv file
def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

#Sigmoid activation funnction
def activationFunctionSigmoid(z):
	return 1/(1 + np.exp(-z))

#Derivative of activation function
def derivativeActivationFunctionSigmoid(z):
	return z*(1-z)
	
#Forward propagating the input's dot product with weights of the corresponding layer and calculate output after applying the sigmoid function.
def forwardPropagation(dataSet, wtFromInputtoHiddenLayer, wtFromHiddentoOutputLayer):
	activationInputLayer = np.reshape(dataSet, (1, inputSize))
	activationHiddenLayer = activationFunctionSigmoid(np.dot(activationInputLayer, wtFromInputtoHiddenLayer))
	activationHiddenLayer[0][0] = 1
	activationOutputLayer = activationFunctionSigmoid(np.dot(activationHiddenLayer, wtFromHiddentoOutputLayer))
	return activationInputLayer, activationHiddenLayer, activationOutputLayer
	
#Calculating the errors, deltas values and using these values along with the momentum value, old weights to update weights to get the new weights.
def BackPropagation(error, activationInputLayer, activationHiddenLayer, activationOutputLayer, wtFromHiddentoOutputLayer, wtFromInputtoHiddenLayer, wtFromHiddentoOutputLayer_oldValues, wtFrominputtoHiddenLayer_oldValues, momentum):
	deltaOutputLayer = derivativeActivationFunctionSigmoid(activationOutputLayer)*error
	deltaHiddenLayer = derivativeActivationFunctionSigmoid(activationHiddenLayer)*np.dot(deltaOutputLayer, np.transpose(wtFromHiddentoOutputLayer))								
	wtHiddentoOutputCurrent = (learningRate*np.dot(np.transpose(activationHiddenLayer), deltaOutputLayer)) + (momentum*wtFromHiddentoOutputLayer_oldValues)
	wtInputtoHiddenCurrent = (learningRate*np.dot(np.transpose(activationInputLayer), deltaHiddenLayer)) + (momentum*wtFrominputtoHiddenLayer_oldValues)
	wtFromHiddentoOutputLayer += wtHiddentoOutputCurrent
	wtFromInputtoHiddenLayer += wtInputtoHiddenCurrent
	return wtFromHiddentoOutputLayer, wtFromInputtoHiddenLayer, wtHiddentoOutputCurrent, wtInputtoHiddenCurrent

#training the network for the training set
def trainNeuralNetwork(wtFromHiddentoOutputLayer, wtFromInputtoHiddenLayer, wtFromHiddentoOutputLayer_oldValues, wtFrominputtoHiddenLayer_oldValues, momentum):
	for i in range(0, trainSetSize):
		#forward propagate input times weight matrix(dot product) and get the predicted output array
		activationInputLayer, activationHiddenLayer, activationOutputLayer = forwardPropagation(trainingData[i, :], wtFromInputtoHiddenLayer, wtFromHiddentoOutputLayer)
		#target value calculation for this input.
		targetOutput = np.insert((np.zeros((1, outputSize-1)) + 0.0001), int(traininglabels[i]), 0.9999)								 
		# Calculate updated weight matrix after backpropagating errors.
		wtFromHiddentoOutputLayer, wtFromInputtoHiddenLayer, wtFromHiddentoOutputLayer_oldValues, wtFrominputtoHiddenLayer_oldValues = BackPropagation(targetOutput-activationOutputLayer, activationInputLayer, activationHiddenLayer, activationOutputLayer, wtFromHiddentoOutputLayer, wtFromInputtoHiddenLayer, wtFromHiddentoOutputLayer_oldValues, wtFrominputtoHiddenLayer_oldValues, momentum) 	
	return wtFromInputtoHiddenLayer, wtFromHiddentoOutputLayer

#testing the network over testing data and calculating the accuracy of the network predictions.
def testNeuralNetwork(inputValues, outputLabelValues, sizeOfSet, wtFromInputtoHiddenLayer, wtFromHiddentoOutputLayer):
	predictedOutput = []
	for i in range(0, sizeOfSet):
		#forward propagating the dot product of input and weight to get the activations at each layer.
		activationInputLayer, activationHiddenLayer, activationOutputLayer = forwardPropagation(inputValues[i, :], wtFromInputtoHiddenLayer, wtFromHiddentoOutputLayer)															
		#to add the predicted output to the output array
		predictedOutput.append(np.argmax(activationOutputLayer))																					
	return accuracy_score(outputLabelValues, predictedOutput), predictedOutput

def neuralNetwork(hiddenLayerSize, momentum):	
	#Initializing weights to random values
	wtFromInputtoHiddenLayer = (np.random.rand(inputSize, hiddenLayerSize) - 0.5)*0.1
	wtFromHiddentoOutputLayer = (np.random.rand(hiddenLayerSize, outputSize) - 0.5)*0.1
	#Initializing old weights matrix with 0
	wtFromHiddentoOutputLayer_oldValues = np.zeros(wtFromHiddentoOutputLayer.shape)
	wtFrominputtoHiddenLayer_oldValues = np.zeros(wtFromInputtoHiddenLayer.shape)	
	#to run for 50 epochs	
	for epoch in range(0, 50):
		#Calculate the training accuracy and output values.
		trainingAccuracy, predictedOutput = testNeuralNetwork(trainingData, traininglabels, trainSetSize, wtFromInputtoHiddenLayer, wtFromHiddentoOutputLayer)									
		#Calculate the testing accuracy and output values.
		testingAccuracy, predictedOutput = testNeuralNetwork(testingData, testingLabels, testSetSize, wtFromInputtoHiddenLayer, wtFromHiddentoOutputLayer)										
		print("Epoch " + str(epoch) + " :\tTraining Set Accuracy = " + str(trainingAccuracy) + "\n\t\tTest Set Accuracy = " + str(testingAccuracy))
		#computing new weights by training after each epoch
		wtFromInputtoHiddenLayer, wtFromHiddentoOutputLayer = trainNeuralNetwork(wtFromHiddentoOutputLayer, wtFromInputtoHiddenLayer, wtFromHiddentoOutputLayer_oldValues, wtFrominputtoHiddenLayer_oldValues, momentum)												
	epoch += 1
	#Calculate the final training accuracy and output values.
	trainingAccuracy, predictedOutput = testNeuralNetwork(trainingData, traininglabels, trainSetSize, wtFromInputtoHiddenLayer, wtFromHiddentoOutputLayer)									
	#Calculate the final testing accuracy and output values.
	testingAccuracy, predictedOutput = testNeuralNetwork(testingData, testingLabels, testSetSize, wtFromInputtoHiddenLayer, wtFromHiddentoOutputLayer)			
	print("Epoch " + str(epoch) + " :\tTraining Set Accuracy = " + str(trainingAccuracy) + "\n\t\tTest Set Accuracy = " + str(testingAccuracy) + "\n\nHidden Layer Size = " + str(hiddenLayerSize) + "\tMomentum = " + str(momentum) + "\tTraining Samples = " + str(trainSetSize) + "\n\nConfusion Matrix :\n")
	print(confusion_matrix(testingLabels, predictedOutput))
	print("\n")
	return

def loadDataToCsv(fileName):
	dataFile = np.loadtxt(fileName, delimiter=',')
	inputValues = np.insert(dataFile[:, np.arange(1, inputSize)]/255, 0, 1, axis=1)
	outputLabelValues = dataFile[:, 0]
	return inputValues, outputLabelValues

#Converting sets to csv
print("\nConverting Training Set")
convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
        "mnist_train.csv", 60000)
print("\nConverting Testing Set")
convert("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte",
        "mnist_test.csv", 10000)

#Load Training and Test Sets :
print("\nLoading Training Set")
trainingData, traininglabels = loadDataToCsv('mnist_train.csv')
print("\nLoading Test Set\n")
testingData, testingLabels = loadDataToCsv('mnist_test.csv')

#Experiment 1 :
print("\nExperiment 1 - varying the number of hidden units")
for hiddenLayerSize in NumnerOfHiddenUnits:
	neuralNetwork(hiddenLayerSize, 0.9)																		

#Experiment 2 :
print("\nExperiment 2 - varying the value of momentum")
for momentum in momentumValues:
	neuralNetwork(100, momentum)																			

#Experiment 3 :
print("\nExperiment 3 - varying the number of training examples")
for i in range(0, 2):
	trainingData, X, traininglabels, Y = train_test_split(trainingData, traininglabels, test_size=0.50)		
	trainSetSize = int(trainSetSize/2)
	neuralNetwork(100, 0.9)	