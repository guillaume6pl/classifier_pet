#!/usr/bin/env python2.7
#coding: utf-8

# ----------------- classifier_pet.py -------------------
# Goal :
#   Spark app that build a cat & dog image classifier
#
# Dependencies can be installed by running:
# 	pip install keras tensorflow h5py pillow
#
# Execution :
#   Written and tested using Python 2.7.13 / Spark 2.1.0
#   Run script as:
#       spark-submit classifieur_pet.py ./input/test/

''' next improvements
choose dir
extract features automatically
timing
GridSearch
validation data
return image that got wrong evaluation
'''

from __future__ import print_function
from __future__ import division
# pyspark package
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, StringIndexer
from pyspark.sql import Row
from pyspark.mllib.classification import SVMWithSGD
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.classification import LabeledPoint
# ML package
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
# regular package
import numpy as np
import sys, os, json, re, subprocess, itertools

sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

def load_features_value(path):
	# [LOADING FEATURES] Feature files -> features-row list
	features_row_list = []
	print(">>>>> extracting values from features file..")
	for feature_file in os.listdir(path):
		# handling label
		label = re.sub(r'[0-9]', '', feature_file)
		label = label[:-9].strip('_')
		# getting values
		for line in open(path + feature_file, "r"):
			values = line.strip('[]').split(',')
			values = [float(x) for x in values]
			# handling features
		features_value_list.append([label,values])
	return features_value_list

def load_features_df_1vs1(features_value_list):
	# [LOADING FEATURES] Feature files -> features-row list
	features_row_list = []
	print(">>>>> creating dataframes from features values for 1vs1 classifier..")
	for feature in features_value_list:
		features_row = Row(label=feature[0], features=feature[1])
		features_row_list.append(features_row)
	print(">>>>> creating rdd from features-row list..")
	features_rdd = sc.parallelize(features_row_list)
	print(">>>>> creating dataframe from rdd..")
	features_df = spark.createDataFrame(features_rdd)
	return features_df

def load_features_df_1vsAll(features_value_list,class1):
	# [LOADING FEATURES] Feature files -> features-row list
	features_row_list = []
	print(">>>>> creating dataframes from features values for 1vs1 classifier..")
	for feature in features_value_list:
		if feature[0] == class1:
			features_row = Row(label=feature[0], features=feature[1])
		else:
			features_row = Row(label="All", features=feature[1])
		features_row_list.append(features_row)
	print(">>>>> creating rdd from features-row list..")
	features_rdd = sc.parallelize(features_row_list)
	print(">>>>> creating dataframe from rdd..")
	features_df = spark.createDataFrame(features_rdd)
	return features_df

def convert_labels(train_features_df,test_features_df):
	# [CONVERT LABELS] Convert string labels to floats with Estimator
	label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
	label_indexer_transformer = label_indexer.fit(train_features_df)
	train_features_df = label_indexer_transformer.transform(train_features_df)
	test_features_df = label_indexer_transformer.transform(test_features_df)
	return train_features_df, test_features_df

def training(class1,class2):
	print("\n" + "+"*40)
	print("+++++ Training for %s vs. %s classifier" % (class1,class2))
	print("+"*40 + "\n")
	# grid search on SVMWithSGD model parameters - train model for each combination
	# return best model parameters
	from pyspark.mllib.classification import SVMWithSGD
	'''
	classmethod train(  data,                   # training data, an RDD of LabeledPoint
	                    iterations=100,         # number of iterations. (default: 100)
	                    step=1.0,               # step parameter used in SGD (default: 1.0)
	                    regParam=0.01,          # regularizer parameter. (default: 0.01)
	                    miniBatchFraction=1.0,  # fraction of data to be used for each SGD iteration. (default: 1.0)
	                    initialWeights=None,    # initial weights. (default: None)
	                    regType='l2',           # regularizer type
	                    intercept=False,        # Boolean - indicates use or not of augmented representation for training data (i.e. whether bias features are activated or not). (default: False)
	                    validateData=True,      # Boolean - indicates if algorithm should validate data before training (default: True)
	                    convergenceTol=0.001    # condition which decides iteration termination. (default: 0.001)
	                    )[source]
	'''
	# model parameters
	model_number = 0
	numIters = [10,50,100,300]
	stepSizes = [1]
	regParams = [0.01]
	best_model_number = None
	best_model = None
	best_prediction_df = None
	best_accuracy = 0
	best_trainErr = 0
	best_errors = 0
	best_iter = None
	best_stepSize = None
	best_regParam = None
	# grid training
	for numIter,stepSize,regParam in itertools.product(numIters,stepSizes,regParams):
		model_number += 1
		print(">>>>> Building model #%i.." % (model_number))
		model = SVMWithSGD.train(train_features_lp, numIter, stepSize, regParam)
		# [TEST] Guess labels on test data
		print(">>>>> Testing model #%i.." % (model_number))
		prediction_lp = test_features_lp.map(lambda p : Row(label_index_predicted=model.predict(p.features), label_index=p.label))
		prediction_df = spark.createDataFrame(prediction_lp)
		# [EVALUATION] the model on training data
		print(">>>>> Evaluating model #%i.." % (model_number))
		#accuracy = prediction_lp.filter(lambda lp: lp[0] == lp[1]).count() / float(prediction_lp.count())
		#trainErr = prediction_lp.filter(lambda lp: lp[0] != lp[1]).count() / float(prediction_lp.count())
		accuracy = prediction_df\
			.filter(prediction_df.label_index_predicted == prediction_df.label_index)\
			.count() / float(prediction_df.count())
		trainErr = prediction_df\
			.filter(prediction_df.label_index_predicted != prediction_df.label_index)\
			.count() / float(prediction_df.count())
		errors = prediction_df\
			.filter(prediction_df.label_index_predicted != prediction_df.label_index)\
			.count()
		# Is it the best model yet?
		if accuracy > best_accuracy:
			best_model_number = model_number
			best_model = model
			best_prediction_df = prediction_df
			best_accuracy = accuracy
			best_trainErr = trainErr
			best_errors = errors
			best_stepSize = stepSize
			best_regParam = regParam
			best_numIter = numIter
		# [RESULTS]
		print("""
		|Model #%i
		|Model trained with (numIter: %.2f, stepSize = %.2f, regParam = %.2f)
		|Model has accuracy of %.3f (errors: %i / training error: %.3f) on test
		""" % (model_number,numIter,stepSize,regParam,accuracy,errors,trainErr))

	# [RESULTS]
	# Display results for best model
	print("""
	|Model #%i is the best model
	|The best model was trained with (numIter: %.2f, stepSize = %.2f, regParam = %.2f)
	|The best model has accuracy of %.3f (errors: %i / training error: %.3f) on test
	""" % (best_model_number,best_numIter,best_stepSize,best_regParam,best_accuracy,best_errors,best_trainErr))
	# Add entry into best_models dictionnary
	best_models[("%s_vs._%s" % (class1,class2))] = { "Accuracy":best_accuracy\
													,"numIter":best_numIter\
													,"stepSize":best_stepSize\
													,"regParam":best_regParam}

def main():
	# parameters
	features_dir = sys.argv[1]
	global train_features_lp
	global test_features_lp
	global best_models
	global features_value_list
	features_value_list = []
	best_models = {}
	classes = []
	for feature_file in os.listdir(features_dir):
		new_class = re.sub(r'[0-9]', '', feature_file)
		new_class = new_class[:-9].strip('_')
		classes.append(new_class)
	classes = sorted(list(set(classes)))
	classes_dup = classes
	# [FEATURES EXTRACTION]
	# subprocess.call(["python", "features_extract.py"])

	# [LOADING FEATURE VALUES] loading featuresvalues into dictionnary
	print(">>>>> Loading features values into list of rows..")
	features_value_list = load_features_value(features_dir)

	# [CLASSIFIER SELECTION] Selecting classifiers (1vs1, 1vsAll)
	# 1vs1 classifiers
	for class1 in classes:
		class2_set = [x for x in classes_dup]
		del class2_set[0:(classes.index(class1)+1)]
		print("classes")
		print(classes)
		print("class2_set")
		print(class2_set)
		for class2 in class2_set:
			print(">>>>> Building dataframes for classifier %s vs. %s.." % (class1,class2))
			# [LOADING FEATURES] loading features values into dataframe
			print("_____ Loading features values into main dataframe")
			features_df = load_features_df_1vs1(features_value_list)
			print("_____ Filtering data within dataframe")
			features_classifier_df = features_df\
									.filter((features_df.label == class1)\
									| (features_df.label ==  class2))
			# [SPLIT DATA] Split data into train & test
			print("_____ Spliting data into training & test data..")
			train_features_df, test_features_df = features_classifier_df.randomSplit([0.8, 0.20])
			train_count = train_features_df.count()
			test_count = test_features_df.count()
			print("%i training data" % (train_count))
			print("%i testing data" % (test_count))
			# [CONVERET LABELS] Convert string labels into floats with an estimator
			print("_____ Converting string labels into floats with an estimator..")
			train_features_df, test_features_df = convert_labels(train_features_df,test_features_df)
			# [CONVERT INTO LABELDPOINTS]
			print(">>>>> Converting dataframe into labelpoint rdd..")
			train_features_lp = train_features_df.rdd.map(lambda row: LabeledPoint(row.label_index, row.features))
			test_features_lp = test_features_df.rdd.map(lambda row: LabeledPoint(row.label_index, row.features))
			# [BUILD MODEL] Learn classifier on training data
			print(">>>>> Training classifier..")
			training(class1,class2)
	# 1vsAll classifiers
	print("1vsALL---------------------------------------------------------------------------------------")
	print("classes")
	print(classes)
	print("class2_set")
	for class1 in classes:
		print(">>>>> Building dataframes for classifier %s vs. All.." % (class1))
		# [LOADING FEATURES] loading features values into dataframe
		print("_____ Loading features values into main dataframe")
		features_df = load_features_df_1vsAll(features_value_list,class1)
		# [SPLIT DATA] Split data into train & test
		print("_____ Spliting data into training & test data..")
		train_features_df, test_features_df = features_df.randomSplit([0.8, 0.20])
		train_count = train_features_df.count()
		test_count = test_features_df.count()
		print("%i training data" % (train_count))
		print("%i testing data" % (test_count))
		# [CONVERET LABELS] Convert string labels into floats with an estimator
		print("_____ Converting string labels into floats with an estimator..")
		train_features_df, test_features_df = convert_labels(train_features_df,test_features_df)
		# [CONVERT INTO LABELDPOINTS]
		print(">>>>> Converting dataframe into labelpoint rdd..")
		train_features_lp = train_features_df.rdd.map(lambda row: LabeledPoint(row.label_index, row.features))
		test_features_lp = test_features_df.rdd.map(lambda row: LabeledPoint(row.label_index, row.features))
		# [BUILD MODEL] Learn classifier on training data
		print(">>>>> Training classifier..")
		training(class1,"All")

	# [OUTPUT]
	# For each classifier, send model parameters in best_classifiers.json
	print(">>>>> Sending best model information to \"best_classifiers.json\"..")
	with open("./output/best_classifiers.json", "w") as out:
		json.dump(best_models, out)

	# hang script to tune it with Spark Web UI (available @ http://localhost:4040)
	raw_input("press ctrl+c to exit")

if __name__ == "__main__":
	main()
