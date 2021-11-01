# -----------------------------------------------------------------------
# Learning the training model and save model by Joblib
# Input original data then drop unused features & drop Noise & drop first column index
# Input feature 241
# python this.py ctu-XX_new_feature.csv
# -----------------------------------------------------------------------
import sys
import csv
import pandas as pd
import numpy as np

import joblib

from parso.python.tree import Flow
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def main():
	if(len(sys.argv) < 1):
		print("Usage: python3 train.py training.csv\n")
		exit(1)

	print("Loading data...")
	TrainingSet = pd.read_csv(sys.argv[1], low_memory = False)
	# 訓練集處理
	TrainingSet.dropna(axis=0, inplace=True)	# 去除Label空值
	# traing_label = TrainingSet['DB_labels']		# 紀錄Label
	# X = TrainingSet.drop(["DB_labels"], axis=1)
	# X = TrainingSet.loc[TrainingSet["DB_labels"] != -1]		#去除Noise
	traing_label = TrainingSet['DB_labels']		# 紀錄Label
	# TrainingSet.drop(Droprow, axis=0, inplace=True)
	# X = X.drop('Unnamed: 0', axis=1)
	X = TrainingSet.drop(["Label"], axis=1)
	X = X.drop(["DB_labels"], axis=1)
	# X = X.drop(["sip"], axis=1)
	# X = X.drop(["dip"], axis=1)
	# X = X.drop(["sport"], axis=1)
	# X = X.drop(["dport"], axis=1)
	# X = X.drop(["proto"], axis=1)

	# Flow = FlowGetFeature(dataSet)
	print("Calculating...")
	clf = tree.DecisionTreeClassifier(random_state=42).fit(X, traing_label)
	# clf = MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=(30,), random_state=42, activation='relu', learning_rate='adaptive').fit(X, traing_label)
	# clf = make_pipeline(StandardScaler(), SVC(gamma='auto')).fit(X, traing_label)
	# clf = AdaBoostClassifier(n_estimators=100, random_state=42).fit(X, traing_label)
	# clf = LogisticRegression(random_state=42).fit(X, traing_label)
	# clf = GaussianNB().fit(X, traing_label)
	# clf = RandomForestClassifier(criterion='entropy', random_state=42)
	# DT_clf = clf.fit(X, traing_label)

	# save model
	print("Saving model...")
	joblib.dump(clf, "DT_DB_10")
# end of main()

if __name__ == '__main__':
	main()
