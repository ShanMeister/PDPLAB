import sys
import csv
import pandas as pd
import numpy as np
import numba as nb

import joblib

from parso.python.tree import Flow
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from keras import backend as K


def main():
	if(len(sys.argv) < 1):
		print("Usage: python3 test.py original_HDB.csv\n")
		exit(1)
	
	print("Loading data...")
	TestingSet = pd.read_csv(sys.argv[1], low_memory = False)
	model = joblib.load('Encoder_9')

	# tes = TestingSet.groupby('DB_labels')
	# tes_L = TestingSet.groupby('Label')
	# 訓練集處理
	TestingSet.dropna(axis=0, inplace=True)	# 去除Label空值
	testing_label = TestingSet['Label']

	X = TestingSet.drop(["Label"], axis=1)
	X = X.drop(["DB_labels"], axis=1)
	X = X.drop(["sip"], axis=1)
	X = X.drop(["dip"], axis=1)
	X = X.drop(["sport"], axis=1)
	X = X.drop(["dport"], axis=1)
	X = X.drop(["proto"], axis=1)

	print("Calculating...")
	predicted = model.predict(X)
	print("Complete!!!")
	actual = testing_label
	print(confusion_matrix(actual, predicted, labels=[1,0]))
	print("=================================")
	tp, fn, fp, tn = confusion_matrix(actual, predicted, labels=[1,0]).ravel()
	Accuracy = (tp+tn)/(tp+fp+fn+tn)
	Precision = tp/(tp+fp)
	Recall = tp/(tp+fn)
	F1 = 2/((1/Precision)+(1/Recall))
	FPR = fp/(fp+tn)
	ER = (fp+fn)/(tp+fp+fn+tn)
	print("tp: {}, fn: {}, fp: {}, tn: {}".format(tp, fn, fp, tn))
	print("Accuracy: " +str(Accuracy)+ " ,Precision: " +str(Precision)+ " ,Recall: " +str(Recall)+ " ,F1: " +str(F1)+" ,FPR: " + str(FPR)+" ,ER: " + str(ER))

	K.clear_session()
# end of main()


if __name__ == '__main__':
	main()
