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

@nb.jit
def MappingLabel(data, origin):
	for i in range(0, len(data)):
		# print(i)
		origin.loc[origin["DB_labels"] == i, "Label_fin"] = data[i]
	# End of for loop
	return origin["Label_fin"].values
# End of def MappingLabel()

# @nb.jit
# def DropNAN(actual, predicted):
# 	actual_new = np.empty(len(actual))
# 	predicted_new = np.empty(len(actual))
# 	for i in range(0, len(actual)):
# 		if predicted[i] != 1 and predicted[i] != 0 and predicted[i] != -1:
# 			continue
# 		else:
# 			if predicted[i] != -1:
# 				predicted[i] = 1
# 			np.append(actual_new, actual[i])
# 			np.append(predicted_new, predicted[i])
# 	# End of for loop
# 	return actual_new, predicted_new
# # End of def DropNAN()

def DropNAN_2(actual, predicted):
	data_dict = {"actual": actual,
				"predicted": predicted
				}
	data = pd.DataFrame(data_dict)
	data.dropna(axis=0, inplace=True)	# 去除Label空值
	return data["actual"].values, data["predicted"].values
# End of def DropNAN_2()

def main():
	if(len(sys.argv) < 1):
		print("Usage: python3 test.py testing_with_noise.csv original_HDB.csv\n")
		exit(1)
	
	print("Loading data...")
	TestingSet = pd.read_csv(sys.argv[1], low_memory = False)
	OriginSet = pd.read_csv(sys.argv[2], low_memory = False)
	model = joblib.load('Encoder_9')

	tes = TestingSet.groupby('DB_labels')
	ori = OriginSet.groupby('DB_labels')
	tes_L = TestingSet.groupby('Label')
	ori_L = OriginSet.groupby('Label')
	# print(len(tes))
	# print(len(ori))
	# print(len(tes_L))
	# print(len(ori_L))
	# 訓練集處理
	# TestingSet.dropna(axis=0, inplace=True)	# 去除Label空值
	TestingSet.loc[TestingSet["Label"] == -1, "Label"] = 0
	testing_label = TestingSet['Label']
	origin_label = OriginSet['Label']

	X = TestingSet.drop(["Label"], axis=1)
	X = X.drop(["DB_labels"], axis=1)

	print("Calculating...")
	test_predicted = model.predict(X)
	print("Complete!!!")
	print("Cluster-based result:")
	actual = testing_label
	print(confusion_matrix(actual, test_predicted, labels=[1,0]))
	print("=================================")
	tp, fn, fp, tn = confusion_matrix(actual, test_predicted, labels=[1,0]).ravel()
	Accuracy = (tp+tn)/(tp+fp+fn+tn)
	Precision = tp/(tp+fp)
	Recall = tp/(tp+fn)
	F1 = 2/((1/Precision)+(1/Recall))
	FPR = fp/(fp+tn)
	ER = (fp+fn)/(tp+fp+fn+tn)
	print("tp: {}, fn: {}, fp: {}, tn: {}".format(tp, fn, fp, tn))
	print("Accuracy: " +str(Accuracy)+ " ,Precision: " +str(Precision)+ " ,Recall: " +str(Recall)+ " ,F1: " +str(F1)+" ,FPR: " + str(FPR)+" ,ER: " + str(ER))

	print("Calculating...")
	predicted = MappingLabel(test_predicted, OriginSet)
	actual = OriginSet["Label"]
	
	print(predicted.shape)
	print(actual.shape)
	actual, predicted = DropNAN_2(actual, predicted)
	print(predicted.shape)
	print(actual.shape)
	
	print("Complete!!!")
	print("Flow-based result:")
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
# end of main()


if __name__ == '__main__':
	main()
