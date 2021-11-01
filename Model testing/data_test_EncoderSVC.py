# -----------------------------------------------------------------------
# Import learning model and get the testing result
# Input original dataset then drop unused feature & drop Noise
# Input feature 241
# Test the data with full pure selected feature
# python this.py ctu-XX_new_features.csv
# -----------------------------------------------------------------------
import sys
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf

from parso.python.tree import Flow
from sklearn.metrics import confusion_matrix

import keras
from keras import layers
from matplotlib import pyplot as plt

# def get_predictions(model, x_test_scaled, threshold):
# 	predictions = model.predict(x_test_scaled)
# 	# provides losses of individual instances
# 	errors = tf.keras.losses.msle(predictions, x_test_scaled)
# 	# 0 = anomaly, 1 = normal
# 	anomaly_mask = pd.Series(errors) > threshold
# 	preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
# 	return preds

def main():
	if(len(sys.argv) < 1):
		print("Usage: python3 test.py testing.csv\n")
		exit(1)
	
	print("Loading data...")
	TestingSet = pd.read_csv(sys.argv[1], low_memory = False)
	Data = TestingSet.loc[TestingSet["Label"] == 1]
	model = joblib.load('Encoder_9')
	# 訓練集處理
	Data.dropna(axis=0, inplace=True)	# 去除Label空值
	testing_label = Data['Label']
	# val_label = val_data['Label']
	X = Data.drop(["Label"], axis=1)
	# Y = val_data.drop(["Label"], axis=1)

	X = X.drop(["sip"], axis=1)
	X = X.drop(["dip"], axis=1)
	X = X.drop(["sport"], axis=1)
	X = X.drop(["dport"], axis=1)
	X = X.drop(["proto"], axis=1)

	print("Calculating...")
	# predictions = get_predictions(model, X, 0.1)
	# print(accuracy_score(predictions, y_test))
	
	test_predicted = model.predict(X)
	for i in range(len(test_predicted)):
		# mse_test=np.sum((test_predicted[i]-X[i])**2)/39
		temp = np.sum(sum(int(test_predicted[i]))-sum(int(X[i]))**2)/39
		print(temp)

	# print(len(test_predicted))
	# mse_test=np.sum((test_predicted-X)**2)/len(X)
	# actual = testing_label
	# print("mse: ", mse_test)
	# plt.hist(mse_test, bins=50)
	# plt.xlabel("test MSE loss")
	# plt.ylabel("No of samples")
	# plt.show()


	anomalies = mse_test > 0.1
	# print(anomalies)
	print("Number of attack samples: ", len(X))
	print("Number of anomaly samples: ", np.sum(anomalies))
	print("Indices of anomaly samples: ", np.where(anomalies))
	# print("Complete!!!")
	# print(confusion_matrix(actual, test_predicted, labels=[1,0]))
	# print("=================================")
	# tp, fn, fp, tn = confusion_matrix(actual, test_predicted, labels=[1,0]).ravel()
	# Accuracy = (tp+tn)/(tp+fp+fn+tn)
	# Precision = tp/(tp+fp)
	# Recall = tp/(tp+fn)
	# F1 = 2/((1/Precision)+(1/Recall))
	# FPR = fp/(fp+tn)
	# ER = (fp+fn)/(tp+fp+fn+tn)

	# print("tp: {}, fn: {}, fp: {}, tn: {}".format(tp, fn, fp, tn))
	# print("Accuracy: " +str(Accuracy)+ " ,Precision: " +str(Precision)+ " ,Recall: " +str(Recall)+ " ,F1: " +str(F1)+" ,FPR: " + str(FPR)+" ,ER: "+str(ER))
# end of main()

if __name__ == '__main__':
	main()
