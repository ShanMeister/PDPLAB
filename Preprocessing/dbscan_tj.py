# -----------------------------------------------------------------------
# Clustering input data by using HDBSCAN
# Input dataset and add new label "DB_labels" into the dataset
# Feature: 45 + 5(tuple) + 1 (DB_labels) 
# python this.py ctu-XX_labeled_new.csv
# -----------------------------------------------------------------------
import sys
import csv
import pandas as pd
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import DBSCAN

# HSBscan for ctu13-5 2021-03-10
sample = 50
size = 15

def GetFeature(data):
	return pd.DataFrame([
		# Flow_ID---,
		# data["sport"], data["dport"], data["proto"], 
		data["Duration"],
		# data["sip"], data["dip"], 

		# Total---,
		data["IAT_Mean"], data["IAT_Std"], data["IAT_Max"], data["IAT_Min"], 
		data["Length_Min"], data["Length_Max"], data["Length_Mean"], data["Length_Std"],
		# data["Total_Packets"], data["IAT_Sum"], data["Length_First"], data["Length_Sum"],
		 
		# Other---,
		# data["Num_Dir_Change"], data["Freq_Dir_Change"],

		# Fwd_flow---,
		data["Fwd_Total_Packets"], data["Fwd_Length_Max"], data["Fwd_Length_Min"], 
		data["Fwd_Length_Mean"], data["Fwd_Length_Std"], data["Fwd_IAT_Sum"], 
		data["Fwd_IAT_Mean"], data["Fwd_IAT_Std"], data["Fwd_IAT_Max"], data["Fwd_IAT_Min"],
		# data["Fwd_Length_First"], data["Fwd_Length_Sum"],

		# Bwd_flow---,
		data["Bwd_Total_Packets"], data["Bwd_Length_Max"], data["Bwd_Length_Min"],
		data["Bwd_Length_Mean"], data["Bwd_Length_Std"], data["Bwd_IAT_Sum"], 
		data["Bwd_IAT_Mean"], data["Bwd_IAT_Std"], data["Bwd_IAT_Max"], data["Bwd_IAT_Min"],
		# data["Bwd_Length_First"], data["Bwd_Length_Sum"],

		# Label---,
		# data["Label"]
		]).T
# End of def GetFeature()

def FlowGetFeature(data):
	return pd.DataFrame([
		# Flow_ID---,
		data["sip"], data["dip"], 
		data["sport"], data["dport"], data["proto"], data["Duration"],
		

		# Total---,
		data["IAT_Mean"], data["IAT_Std"], data["IAT_Max"], data["IAT_Min"], 
		data["Length_Min"], data["Length_Max"], data["Length_Mean"], data["Length_Std"],
		data["Total_Packets"], data["IAT_Sum"], data["Length_First"], data["Length_Sum"],
		 
		# Other---,
		data["Num_Dir_Change"], data["Freq_Dir_Change"],

		# Fwd_flow---,
		data["Fwd_Total_Packets"], data["Fwd_Length_Max"], data["Fwd_Length_Min"], 
		data["Fwd_Length_Mean"], data["Fwd_Length_Std"], data["Fwd_IAT_Sum"], 
		data["Fwd_IAT_Mean"], data["Fwd_IAT_Std"], data["Fwd_IAT_Max"], data["Fwd_IAT_Min"],
		data["Fwd_Length_First"], data["Fwd_Length_Sum"],

		# Bwd_flow---,
		data["Bwd_Total_Packets"], data["Bwd_Length_Max"], data["Bwd_Length_Min"],
		data["Bwd_Length_Mean"], data["Bwd_Length_Std"], data["Bwd_IAT_Sum"], 
		data["Bwd_IAT_Mean"], data["Bwd_IAT_Std"], data["Bwd_IAT_Max"], data["Bwd_IAT_Min"],
		data["Bwd_Length_First"], data["Bwd_Length_Sum"],

		# Label---,
		data["Label"]
		]).T
# End of def FlowGetFeature()


def main():
	if(len(sys.argv) < 1):
		print("Usage: python3 HDBSCAN.py dataset.csv\n")
		exit(1)

	print("Loading data...")
	dataSet = pd.read_csv(sys.argv[1], low_memory = False)
	# dataSet["sport"].astype('int')
	# dataSet["dport"].astype('int')
	# print(dataSet.dtypes)
	print("Standardizing data...")
	X = GetFeature(dataSet)
	X = StandardScaler().fit_transform(X)			# Normalized: z = (x - mean) / std
	# db = DBSCAN(eps = 0.5, min_samples = 5).fit(X)
	# hdb = hdbscan.HDBSCAN(min_cluster_size=15).fit(X)
	print("Clustering...")
	hdb = hdbscan.HDBSCAN(min_samples = sample, min_cluster_size= size).fit(X)
	
	Flow = FlowGetFeature(dataSet)
	Flow["DB_labels"] = hdb.labels_
	filename = sys.argv[1].replace("_labeled_new.csv", "")
	# Flow.to_csv(filename + "_HDB_size_" + str(size) + ".csv", index=None)
	Flow.to_csv(filename + "_HDB" + ".csv", index=None)
	print("Complete!!!")
	
	# dataSet["DB_labels"] = db.labels_
	# dataSet.to_csv("result.csv")
	
	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0)
	n_noise_ = list(hdb.labels_).count(-1)

	print('Estimated number of clusters: %d' % n_clusters_)
	print('Estimated number of noise points: %d' % n_noise_)

	
# end of main()
if __name__ == '__main__':
	main()