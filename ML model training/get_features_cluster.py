# -----------------------------------------------------------------------
# Get 2nd features by clustering group 
# feature: 39 + (39*5) = 234	(count sum/max/min/std/mean)
# 			234 + 5(5-tuple) + 2 (Label + DB_labels) = 241 (Total)
# python this.py HDB.csv
# -----------------------------------------------------------------------
import sys
import datetime
import math
import csv
import pandas as pd
import numpy as np
import numba as nb

columns = ["Duration_sum", "Duration_max", "Duration_min", "Duration_std", "Duration_mean", 	# Duration
			"IAT_Mean_sum", "IAT_Mean_max", "IAT_Mean_min", "IAT_Mean_std", "IAT_Mean_mean",	# IAT_Mean
			"IAT_Std_sum", "IAT_Std_max", "IAT_Std_min", "IAT_Std_std", "IAT_Std_mean",	# IAT_Std
			"IAT_Max_sum", "IAT_Max_max", "IAT_Max_min", "IAT_Max_std", "IAT_Max_mean",	# IAT_Max
			"IAT_Min_sum", "IAT_Min_max", "IAT_Min_min", "IAT_Min_std", "IAT_Min_mean",	# IAT_Min
			"Length_Min_sum", "Length_Min_max", "Length_Min_min", "Length_Min_std", "Length_Min_mean",	# Length_Min
			"Length_Max_sum", "Length_Max_max", "Length_Max_min", "Length_Max_std", "Length_Max_mean",	# Length_Max
			"Length_Mean_sum", "Length_Mean_max", "Length_Mean_min", "Length_Mean_std", "Length_Mean_mean",	# Length_Mean
			"Length_Std_sum", "Length_Std_max", "Length_Std_min", "Length_Std_std", "Length_Std_mean",	# Length_Std
			"Total_Packets_sum", "Total_Packets_max", "Total_Packets_min", "Total_Packets_std", "Total_Packets_mean",	# Total_Packets
			"IAT_Sum_sum", "IAT_Sum_max", "IAT_Sum_min", "IAT_Sum_std", "IAT_Sum_mean",	# IAT_Sum
			"Length_First_sum", "Length_First_max", "Length_First_min", "Length_First_std", "Length_First_mean",	# Length_First
			"Length_Sum_sum", "Length_Sum_max", "Length_Sum_min", "Length_Sum_std", "Length_Sum_mean",	# Length_Sum
			"Num_Dir_Change_sum", "Num_Dir_Change_max", "Num_Dir_Change_min", "Num_Dir_Change_std", "Num_Dir_Change_mean",	# Num_Dir_Change
			"Freq_Dir_Change_sum", "Freq_Dir_Change_max", "Freq_Dir_Change_min", "Freq_Dir_Change_std", "Freq_Dir_Change_mean",	# Freq_Dir_Change
			"Fwd_Total_Packets_sum", "Fwd_Total_Packets_max", "Fwd_Total_Packets_min", "Fwd_Total_Packets_std", "Fwd_Total_Packets_mean",	# Fwd_Total_Packets
			"Fwd_Length_Max_sum", "Fwd_Length_Max_max", "Fwd_Length_Max_min", "Fwd_Length_Max_std", "Fwd_Length_Max_mean",	# Fwd_Length_Max
			"Fwd_Length_Min_sum", "Fwd_Length_Min_max", "Fwd_Length_Min_min", "Fwd_Length_Min_std", "Fwd_Length_Min_mean",	# Fwd_Length_Min
			"Fwd_Length_Mean_sum", "Fwd_Length_Mean_max", "Fwd_Length_Mean_min", "Fwd_Length_Mean_std", "Fwd_Length_Mean_mean",	# Fwd_Length_Mean
			"Fwd_Length_Std_sum", "Fwd_Length_Std_max", "Fwd_Length_Std_min", "Fwd_Length_Std_std", "Fwd_Length_Std_mean",	# Fwd_Length_Std
			"Fwd_IAT_Sum_sum", "Fwd_IAT_Sum_max", "Fwd_IAT_Sum_min", "Fwd_IAT_Sum_std", "Fwd_IAT_Sum_mean",	# Fwd_IAT_Sum
			"Fwd_IAT_Mean_sum", "Fwd_IAT_Mean_max", "Fwd_IAT_Mean_min", "Fwd_IAT_Mean_std", "Fwd_IAT_Mean_mean",	# Fwd_IAT_Mean
			"Fwd_IAT_Std_sum", "Fwd_IAT_Std_max", "Fwd_IAT_Std_min", "Fwd_IAT_Std_std", "Fwd_IAT_Std_mean",	# Fwd_IAT_Std
			"Fwd_IAT_Max_sum", "Fwd_IAT_Max_max", "Fwd_IAT_Max_min", "Fwd_IAT_Max_std", "Fwd_IAT_Max_mean",	# Fwd_IAT_Max
			"Fwd_IAT_Min_sum", "Fwd_IAT_Min_max", "Fwd_IAT_Min_min", "Fwd_IAT_Min_std", "Fwd_IAT_Min_mean",	# Fwd_IAT_Min
			"Fwd_Length_First_sum", "Fwd_Length_First_max", "Fwd_Length_First_min", "Fwd_Length_First_std", "Fwd_Length_First_mean",	# Fwd_Length_First
			"Fwd_Length_Sum_sum", "Fwd_Length_Sum_max", "Fwd_Length_Sum_min", "Fwd_Length_Sum_std", "Fwd_Length_Sum_mean",	# Fwd_Length_Sum
			"Bwd_Total_Packets_sum", "Bwd_Total_Packets_max", "Bwd_Total_Packets_min", "Bwd_Total_Packets_std", "Bwd_Total_Packets_mean",	# Bwd_Total_Packets
			"Bwd_Length_Max_sum", "Bwd_Length_Max_max", "Bwd_Length_Max_min", "Bwd_Length_Max_std", "Bwd_Length_Max_mean",	# Bwd_Length_Max
			"Bwd_Length_Min_sum", "Bwd_Length_Min_max", "Bwd_Length_Min_min", "Bwd_Length_Min_std", "Bwd_Length_Min_mean",	# Bwd_Length_Min
			"Bwd_Length_Mean_sum", "Bwd_Length_Mean_max", "Bwd_Length_Mean_min", "Bwd_Length_Mean_std", "Bwd_Length_Mean_mean",	# Bwd_Length_Mean
			"Bwd_Length_Std_sum", "Bwd_Length_Std_max", "Bwd_Length_Std_min", "Bwd_Length_Std_std", "Bwd_Length_Std_mean",	# Bwd_Length_Std
			"Bwd_IAT_Sum_sum", "Bwd_IAT_Sum_max", "Bwd_IAT_Sum_min", "Bwd_IAT_Sum_std", "Bwd_IAT_Sum_mean",	# Bwd_IAT_Sum
			"Bwd_IAT_Mean_sum", "Bwd_IAT_Mean_max", "Bwd_IAT_Mean_min", "Bwd_IAT_Mean_std", "Bwd_IAT_Mean_mean",	# Bwd_IAT_Mean
			"Bwd_IAT_Std_sum", "Bwd_IAT_Std_max", "Bwd_IAT_Std_min", "Bwd_IAT_Std_std", "Bwd_IAT_Std_mean",	# Bwd_IAT_Std
			"Bwd_IAT_Max_sum", "Bwd_IAT_Max_max", "Bwd_IAT_Max_min", "Bwd_IAT_Max_std", "Bwd_IAT_Max_mean",	# Bwd_IAT_Max
			"Bwd_IAT_Min_sum", "Bwd_IAT_Min_max", "Bwd_IAT_Min_min", "Bwd_IAT_Min_std", "Bwd_IAT_Min_mean",	# Bwd_IAT_Min
			"Bwd_Length_First_sum", "Bwd_Length_First_max", "Bwd_Length_First_min", "Bwd_Length_First_std", "Bwd_Length_First_mean",	# Bwd_Length_First
			"Bwd_Length_Sum_sum", "Bwd_Length_Sum_max", "Bwd_Length_Sum_min", "Bwd_Length_Sum_std", "Bwd_Length_Sum_mean",	# Bwd_Length_Sum
			"DB_labels"
			]

def GetFeature(data):
	return pd.DataFrame([
		# Flow_ID---(6),
		# data["sport"], data["dport"], data["proto"], 
		data["Duration"],
		# data["sip"], data["dip"], 

		# Total---(12),
		data["IAT_Mean"], data["IAT_Std"], data["IAT_Max"], data["IAT_Min"], 
		data["Length_Min"], data["Length_Max"], data["Length_Mean"], data["Length_Std"],
		data["Total_Packets"], data["IAT_Sum"], data["Length_First"], data["Length_Sum"],
		 
		# Other---(2),
		data["Num_Dir_Change"], data["Freq_Dir_Change"],

		# Fwd_flow---(12),
		data["Fwd_Total_Packets"], data["Fwd_Length_Max"], data["Fwd_Length_Min"], 
		data["Fwd_Length_Mean"], data["Fwd_Length_Std"], data["Fwd_IAT_Sum"], 
		data["Fwd_IAT_Mean"], data["Fwd_IAT_Std"], data["Fwd_IAT_Max"], data["Fwd_IAT_Min"],
		data["Fwd_Length_First"], data["Fwd_Length_Sum"],

		# Bwd_flow---(12),
		data["Bwd_Total_Packets"], data["Bwd_Length_Max"], data["Bwd_Length_Min"],
		data["Bwd_Length_Mean"], data["Bwd_Length_Std"], data["Bwd_IAT_Sum"], 
		data["Bwd_IAT_Mean"], data["Bwd_IAT_Std"], data["Bwd_IAT_Max"], data["Bwd_IAT_Min"],
		data["Bwd_Length_First"], data["Bwd_Length_Sum"],

		# data["Label"], data["DB_labels"]
		]).T
# End of def GetFeature()

def FlowGetFeature(data):
	return pd.DataFrame([
		# Flow_ID---,
		data["sip"], data["dip"], 
		data["sport"], data["dport"], data["proto"], data["Duration"],

		# Total---,
		data["IAT_Sum"], data["IAT_Mean"], data["IAT_Std"], data["IAT_Max"], data["IAT_Min"], 
		data["Length_Min"], data["Length_Max"], data["Length_Mean"], data["Length_Std"], data["Length_Sum"], 
		data["Length_First"], data["Total_Packets"],
		 
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
		data["Label"], data["DB_labels"]
		]).T
# End of def FlowGetFeature()

@nb.jit
def GetClusterFeature(data):
	df = pd.DataFrame(data=None, columns=columns)
	clu = data.groupby('DB_labels')
	for i in range(0, len(clu)-1):
		# print(i)
		group = data.loc[data["DB_labels"] == i]
		# Duration
		df.loc[i, "Duration_sum"] = group["Duration"].sum()
		df.loc[i, "Duration_max"] = group["Duration"].max()
		df.loc[i, "Duration_min"] = group["Duration"].min()
		df.loc[i, "Duration_std"] = group["Duration"].std()
		df.loc[i, "Duration_mean"] = group["Duration"].mean()
		# IAT_Mean
		df.loc[i, "IAT_Mean_sum"] = group["IAT_Mean"].sum()
		df.loc[i, "IAT_Mean_max"] = group["IAT_Mean"].max()
		df.loc[i, "IAT_Mean_min"] = group["IAT_Mean"].min()
		df.loc[i, "IAT_Mean_std"] = group["IAT_Mean"].std()
		df.loc[i, "IAT_Mean_mean"] = group["IAT_Mean"].mean()
		# IAT_Std
		df.loc[i, "IAT_Std_sum"] = group["IAT_Std"].sum()
		df.loc[i, "IAT_Std_max"] = group["IAT_Std"].max()
		df.loc[i, "IAT_Std_min"] = group["IAT_Std"].min()
		df.loc[i, "IAT_Std_std"] = group["IAT_Std"].std()
		df.loc[i, "IAT_Std_mean"] = group["IAT_Std"].mean()
		# IAT_Max
		df.loc[i, "IAT_Max_sum"] = group["IAT_Max"].sum()
		df.loc[i, "IAT_Max_max"] = group["IAT_Max"].max()
		df.loc[i, "IAT_Max_min"] = group["IAT_Max"].min()
		df.loc[i, "IAT_Max_std"] = group["IAT_Max"].std()
		df.loc[i, "IAT_Max_mean"] = group["IAT_Max"].mean()
		# IAT_Min
		df.loc[i, "IAT_Min_sum"] = group["IAT_Min"].sum()
		df.loc[i, "IAT_Min_max"] = group["IAT_Min"].max()
		df.loc[i, "IAT_Min_min"] = group["IAT_Min"].min()
		df.loc[i, "IAT_Min_std"] = group["IAT_Min"].std()
		df.loc[i, "IAT_Min_mean"] = group["IAT_Min"].mean()
		# Length_Min
		df.loc[i, "Length_Min_sum"] = group["Length_Min"].sum()
		df.loc[i, "Length_Min_max"] = group["Length_Min"].max()
		df.loc[i, "Length_Min_min"] = group["Length_Min"].min()
		df.loc[i, "Length_Min_std"] = group["Length_Min"].std()
		df.loc[i, "Length_Min_mean"] = group["Length_Min"].mean()
		# Length_Max
		df.loc[i, "Length_Max_sum"] = group["Length_Max"].sum()
		df.loc[i, "Length_Max_max"] = group["Length_Max"].max()
		df.loc[i, "Length_Max_min"] = group["Length_Max"].min()
		df.loc[i, "Length_Max_std"] = group["Length_Max"].std()
		df.loc[i, "Length_Max_mean"] = group["Length_Max"].mean()
		# Length_Mean
		df.loc[i, "Length_Mean_sum"] = group["Length_Mean"].sum()
		df.loc[i, "Length_Mean_max"] = group["Length_Mean"].max()
		df.loc[i, "Length_Mean_min"] = group["Length_Mean"].min()
		df.loc[i, "Length_Mean_std"] = group["Length_Mean"].std()
		df.loc[i, "Length_Mean_mean"] = group["Length_Mean"].mean()
		# Length_Std
		df.loc[i, "Length_Std_sum"] = group["Length_Std"].sum()
		df.loc[i, "Length_Std_max"] = group["Length_Std"].max()
		df.loc[i, "Length_Std_min"] = group["Length_Std"].min()
		df.loc[i, "Length_Std_std"] = group["Length_Std"].std()
		df.loc[i, "Length_Std_mean"] = group["Length_Std"].mean()
		# Total_Packets
		df.loc[i, "Total_Packets_sum"] = group["Total_Packets"].sum()
		df.loc[i, "Total_Packets_max"] = group["Total_Packets"].max()
		df.loc[i, "Total_Packets_min"] = group["Total_Packets"].min()
		df.loc[i, "Total_Packets_std"] = group["Total_Packets"].std()
		df.loc[i, "Total_Packets_mean"] = group["Total_Packets"].mean()
		# IAT_Sum
		df.loc[i, "IAT_Sum_sum"] = group["IAT_Sum"].sum()
		df.loc[i, "IAT_Sum_max"] = group["IAT_Sum"].max()
		df.loc[i, "IAT_Sum_min"] = group["IAT_Sum"].min()
		df.loc[i, "IAT_Sum_std"] = group["IAT_Sum"].std()
		df.loc[i, "IAT_Sum_mean"] = group["IAT_Sum"].mean()
		# Length_First
		df.loc[i, "Length_First_sum"] = group["Length_First"].sum()
		df.loc[i, "Length_First_max"] = group["Length_First"].max()
		df.loc[i, "Length_First_min"] = group["Length_First"].min()
		df.loc[i, "Length_First_std"] = group["Length_First"].std()
		df.loc[i, "Length_First_mean"] = group["Length_First"].mean()
		# Length_Sum
		df.loc[i, "Length_Sum_sum"] = group["Length_Sum"].sum()
		df.loc[i, "Length_Sum_max"] = group["Length_Sum"].max()
		df.loc[i, "Length_Sum_min"] = group["Length_Sum"].min()
		df.loc[i, "Length_Sum_std"] = group["Length_Sum"].std()
		df.loc[i, "Length_Sum_mean"] = group["Length_Sum"].mean()
		# Num_Dir_Change
		df.loc[i, "Num_Dir_Change_sum"] = group["Num_Dir_Change"].sum()
		df.loc[i, "Num_Dir_Change_max"] = group["Num_Dir_Change"].max()
		df.loc[i, "Num_Dir_Change_min"] = group["Num_Dir_Change"].min()
		df.loc[i, "Num_Dir_Change_std"] = group["Num_Dir_Change"].std()
		df.loc[i, "Num_Dir_Change_mean"] = group["Num_Dir_Change"].mean()
		# Freq_Dir_Change
		df.loc[i, "Freq_Dir_Change_sum"] = group["Freq_Dir_Change"].sum()
		df.loc[i, "Freq_Dir_Change_max"] = group["Freq_Dir_Change"].max()
		df.loc[i, "Freq_Dir_Change_min"] = group["Freq_Dir_Change"].min()
		df.loc[i, "Freq_Dir_Change_std"] = group["Freq_Dir_Change"].std()
		df.loc[i, "Freq_Dir_Change_mean"] = group["Freq_Dir_Change"].mean()
		# Fwd_Total_Packets
		df.loc[i, "Fwd_Total_Packets_sum"] = group["Fwd_Total_Packets"].sum()
		df.loc[i, "Fwd_Total_Packets_max"] = group["Fwd_Total_Packets"].max()
		df.loc[i, "Fwd_Total_Packets_min"] = group["Fwd_Total_Packets"].min()
		df.loc[i, "Fwd_Total_Packets_std"] = group["Fwd_Total_Packets"].std()
		df.loc[i, "Fwd_Total_Packets_mean"] = group["Fwd_Total_Packets"].mean()
		# Fwd_Length_Max
		df.loc[i, "Fwd_Length_Max_sum"] = group["Fwd_Length_Max"].sum()
		df.loc[i, "Fwd_Length_Max_max"] = group["Fwd_Length_Max"].max()
		df.loc[i, "Fwd_Length_Max_min"] = group["Fwd_Length_Max"].min()
		df.loc[i, "Fwd_Length_Max_std"] = group["Fwd_Length_Max"].std()
		df.loc[i, "Fwd_Length_Max_mean"] = group["Fwd_Length_Max"].mean()
		# Fwd_Length_Min
		df.loc[i, "Fwd_Length_Min_sum"] = group["Fwd_Length_Min"].sum()
		df.loc[i, "Fwd_Length_Min_max"] = group["Fwd_Length_Min"].max()
		df.loc[i, "Fwd_Length_Min_min"] = group["Fwd_Length_Min"].min()
		df.loc[i, "Fwd_Length_Min_std"] = group["Fwd_Length_Min"].std()
		df.loc[i, "Fwd_Length_Min_mean"] = group["Fwd_Length_Min"].mean()
		# Fwd_Length_Mean
		df.loc[i, "Fwd_Length_Mean_sum"] = group["Fwd_Length_Mean"].sum()
		df.loc[i, "Fwd_Length_Mean_max"] = group["Fwd_Length_Mean"].max()
		df.loc[i, "Fwd_Length_Mean_min"] = group["Fwd_Length_Mean"].min()
		df.loc[i, "Fwd_Length_Mean_std"] = group["Fwd_Length_Mean"].std()
		df.loc[i, "Fwd_Length_Mean_mean"] = group["Fwd_Length_Mean"].mean()
		# Fwd_Length_Std
		df.loc[i, "Fwd_Length_Std_sum"] = group["Fwd_Length_Std"].sum()
		df.loc[i, "Fwd_Length_Std_max"] = group["Fwd_Length_Std"].max()
		df.loc[i, "Fwd_Length_Std_min"] = group["Fwd_Length_Std"].min()
		df.loc[i, "Fwd_Length_Std_std"] = group["Fwd_Length_Std"].std()
		df.loc[i, "Fwd_Length_Std_mean"] = group["Fwd_Length_Std"].mean()
		# Fwd_IAT_Sum
		df.loc[i, "Fwd_IAT_Sum_sum"] = group["Fwd_IAT_Sum"].sum()
		df.loc[i, "Fwd_IAT_Sum_max"] = group["Fwd_IAT_Sum"].max()
		df.loc[i, "Fwd_IAT_Sum_min"] = group["Fwd_IAT_Sum"].min()
		df.loc[i, "Fwd_IAT_Sum_std"] = group["Fwd_IAT_Sum"].std()
		df.loc[i, "Fwd_IAT_Sum_mean"] = group["Fwd_IAT_Sum"].mean()
		# Fwd_IAT_Mean
		df.loc[i, "Fwd_IAT_Mean_sum"] = group["Fwd_IAT_Mean"].sum()
		df.loc[i, "Fwd_IAT_Mean_max"] = group["Fwd_IAT_Mean"].max()
		df.loc[i, "Fwd_IAT_Mean_min"] = group["Fwd_IAT_Mean"].min()
		df.loc[i, "Fwd_IAT_Mean_std"] = group["Fwd_IAT_Mean"].std()
		df.loc[i, "Fwd_IAT_Mean_mean"] = group["Fwd_IAT_Mean"].mean()
		# Fwd_IAT_Std
		df.loc[i, "Fwd_IAT_Std_sum"] = group["Fwd_IAT_Std"].sum()
		df.loc[i, "Fwd_IAT_Std_max"] = group["Fwd_IAT_Std"].max()
		df.loc[i, "Fwd_IAT_Std_min"] = group["Fwd_IAT_Std"].min()
		df.loc[i, "Fwd_IAT_Std_std"] = group["Fwd_IAT_Std"].std()
		df.loc[i, "Fwd_IAT_Std_mean"] = group["Fwd_IAT_Std"].mean()
		# Fwd_IAT_Max
		df.loc[i, "Fwd_IAT_Max_sum"] = group["Fwd_IAT_Max"].sum()
		df.loc[i, "Fwd_IAT_Max_max"] = group["Fwd_IAT_Max"].max()
		df.loc[i, "Fwd_IAT_Max_min"] = group["Fwd_IAT_Max"].min()
		df.loc[i, "Fwd_IAT_Max_std"] = group["Fwd_IAT_Max"].std()
		df.loc[i, "Fwd_IAT_Max_mean"] = group["Fwd_IAT_Max"].mean()
		# Fwd_IAT_Min
		df.loc[i, "Fwd_IAT_Min_sum"] = group["Fwd_IAT_Min"].sum()
		df.loc[i, "Fwd_IAT_Min_max"] = group["Fwd_IAT_Min"].max()
		df.loc[i, "Fwd_IAT_Min_min"] = group["Fwd_IAT_Min"].min()
		df.loc[i, "Fwd_IAT_Min_std"] = group["Fwd_IAT_Min"].std()
		df.loc[i, "Fwd_IAT_Min_mean"] = group["Fwd_IAT_Min"].mean()
		# Fwd_Length_First
		df.loc[i, "Fwd_Length_First_sum"] = group["Fwd_Length_First"].sum()
		df.loc[i, "Fwd_Length_First_max"] = group["Fwd_Length_First"].max()
		df.loc[i, "Fwd_Length_First_min"] = group["Fwd_Length_First"].min()
		df.loc[i, "Fwd_Length_First_std"] = group["Fwd_Length_First"].std()
		df.loc[i, "Fwd_Length_First_mean"] = group["Fwd_Length_First"].mean()
		# Fwd_Length_Sum
		df.loc[i, "Fwd_Length_Sum_sum"] = group["Fwd_Length_Sum"].sum()
		df.loc[i, "Fwd_Length_Sum_max"] = group["Fwd_Length_Sum"].max()
		df.loc[i, "Fwd_Length_Sum_min"] = group["Fwd_Length_Sum"].min()
		df.loc[i, "Fwd_Length_Sum_std"] = group["Fwd_Length_Sum"].std()
		df.loc[i, "Fwd_Length_Sum_mean"] = group["Fwd_Length_Sum"].mean()
		# Bwd_Total_Packets
		df.loc[i, "Bwd_Total_Packets_sum"] = group["Bwd_Total_Packets"].sum()
		df.loc[i, "Bwd_Total_Packets_max"] = group["Bwd_Total_Packets"].max()
		df.loc[i, "Bwd_Total_Packets_min"] = group["Bwd_Total_Packets"].min()
		df.loc[i, "Bwd_Total_Packets_std"] = group["Bwd_Total_Packets"].std()
		df.loc[i, "Bwd_Total_Packets_mean"] = group["Bwd_Total_Packets"].mean()
		# Bwd_Length_Max
		df.loc[i, "Bwd_Length_Max_sum"] = group["Bwd_Length_Max"].sum()
		df.loc[i, "Bwd_Length_Max_max"] = group["Bwd_Length_Max"].max()
		df.loc[i, "Bwd_Length_Max_min"] = group["Bwd_Length_Max"].min()
		df.loc[i, "Bwd_Length_Max_std"] = group["Bwd_Length_Max"].std()
		df.loc[i, "Bwd_Length_Max_mean"] = group["Bwd_Length_Max"].mean()
		# Bwd_Length_Min
		df.loc[i, "Bwd_Length_Min_sum"] = group["Bwd_Length_Min"].sum()
		df.loc[i, "Bwd_Length_Min_max"] = group["Bwd_Length_Min"].max()
		df.loc[i, "Bwd_Length_Min_min"] = group["Bwd_Length_Min"].min()
		df.loc[i, "Bwd_Length_Min_std"] = group["Bwd_Length_Min"].std()
		df.loc[i, "Bwd_Length_Min_mean"] = group["Bwd_Length_Min"].mean()
		# Bwd_Length_Mean
		df.loc[i, "Bwd_Length_Mean_sum"] = group["Bwd_Length_Mean"].sum()
		df.loc[i, "Bwd_Length_Mean_max"] = group["Bwd_Length_Mean"].max()
		df.loc[i, "Bwd_Length_Mean_min"] = group["Bwd_Length_Mean"].min()
		df.loc[i, "Bwd_Length_Mean_std"] = group["Bwd_Length_Mean"].std()
		df.loc[i, "Bwd_Length_Mean_mean"] = group["Bwd_Length_Mean"].mean()
		# Bwd_Length_Std
		df.loc[i, "Bwd_Length_Std_sum"] = group["Bwd_Length_Std"].sum()
		df.loc[i, "Bwd_Length_Std_max"] = group["Bwd_Length_Std"].max()
		df.loc[i, "Bwd_Length_Std_min"] = group["Bwd_Length_Std"].min()
		df.loc[i, "Bwd_Length_Std_std"] = group["Bwd_Length_Std"].std()
		df.loc[i, "Bwd_Length_Std_mean"] = group["Bwd_Length_Std"].mean()
		# Bwd_IAT_Sum
		df.loc[i, "Bwd_IAT_Sum_sum"] = group["Bwd_IAT_Sum"].sum()
		df.loc[i, "Bwd_IAT_Sum_max"] = group["Bwd_IAT_Sum"].max()
		df.loc[i, "Bwd_IAT_Sum_min"] = group["Bwd_IAT_Sum"].min()
		df.loc[i, "Bwd_IAT_Sum_std"] = group["Bwd_IAT_Sum"].std()
		df.loc[i, "Bwd_IAT_Sum_mean"] = group["Bwd_IAT_Sum"].mean()
		# Bwd_IAT_Mean
		df.loc[i, "Bwd_IAT_Mean_sum"] = group["Bwd_IAT_Mean"].sum()
		df.loc[i, "Bwd_IAT_Mean_max"] = group["Bwd_IAT_Mean"].max()
		df.loc[i, "Bwd_IAT_Mean_min"] = group["Bwd_IAT_Mean"].min()
		df.loc[i, "Bwd_IAT_Mean_std"] = group["Bwd_IAT_Mean"].std()
		df.loc[i, "Bwd_IAT_Mean_mean"] = group["Bwd_IAT_Mean"].mean()
		# Bwd_IAT_Std
		df.loc[i, "Bwd_IAT_Std_sum"] = group["Bwd_IAT_Std"].sum()
		df.loc[i, "Bwd_IAT_Std_max"] = group["Bwd_IAT_Std"].max()
		df.loc[i, "Bwd_IAT_Std_min"] = group["Bwd_IAT_Std"].min()
		df.loc[i, "Bwd_IAT_Std_std"] = group["Bwd_IAT_Std"].std()
		df.loc[i, "Bwd_IAT_Std_mean"] = group["Bwd_IAT_Std"].mean()
		# Bwd_IAT_Max
		df.loc[i, "Bwd_IAT_Max_sum"] = group["Bwd_IAT_Max"].sum()
		df.loc[i, "Bwd_IAT_Max_max"] = group["Bwd_IAT_Max"].max()
		df.loc[i, "Bwd_IAT_Max_min"] = group["Bwd_IAT_Max"].min()
		df.loc[i, "Bwd_IAT_Max_std"] = group["Bwd_IAT_Max"].std()
		df.loc[i, "Bwd_IAT_Max_mean"] = group["Bwd_IAT_Max"].mean()
		# Bwd_IAT_Min
		df.loc[i, "Bwd_IAT_Min_sum"] = group["Bwd_IAT_Min"].sum()
		df.loc[i, "Bwd_IAT_Min_max"] = group["Bwd_IAT_Min"].max()
		df.loc[i, "Bwd_IAT_Min_min"] = group["Bwd_IAT_Min"].min()
		df.loc[i, "Bwd_IAT_Min_std"] = group["Bwd_IAT_Min"].std()
		df.loc[i, "Bwd_IAT_Min_mean"] = group["Bwd_IAT_Min"].mean()
		# Bwd_Length_First
		df.loc[i, "Bwd_Length_First_sum"] = group["Bwd_Length_First"].sum()
		df.loc[i, "Bwd_Length_First_max"] = group["Bwd_Length_First"].max()
		df.loc[i, "Bwd_Length_First_min"] = group["Bwd_Length_First"].min()
		df.loc[i, "Bwd_Length_First_std"] = group["Bwd_Length_First"].std()
		df.loc[i, "Bwd_Length_First_mean"] = group["Bwd_Length_First"].mean()
		# Bwd_Length_Sum
		df.loc[i, "Bwd_Length_Sum_sum"] = group["Bwd_Length_Sum"].sum()
		df.loc[i, "Bwd_Length_Sum_max"] = group["Bwd_Length_Sum"].max()
		df.loc[i, "Bwd_Length_Sum_min"] = group["Bwd_Length_Sum"].min()
		df.loc[i, "Bwd_Length_Sum_std"] = group["Bwd_Length_Sum"].std()
		df.loc[i, "Bwd_Length_Sum_mean"] = group["Bwd_Length_Sum"].mean()

		df.loc[i, "DB_labels"] = i

		num = group.groupby("Label")
		if len(num) == 1:
			x = group["Label"].values
			if x[0] == 0:
				df.loc[i, "Label"] = 0
			else:
				df.loc[i, "Label"] = 1
		else:
			df.loc[i, "Label"] = -1
	return df
# End of GetClusterFeature()

def Filter(data):
	clu = data.groupby('DB_labels')
	result = pd.DataFrame(data=None, columns=columns)
	for i in range(0, len(clu)-1):
		# print("i = " + str(i))
		if data.loc[i, "Label"] == -1:	
			continue
		else:
			group = data.loc[data["DB_labels"] == i]
			result = result.append(group, ignore_index=True)	# 將data加入dataframe中
	return result
# End of def Filter()

def main():
	if len(sys.argv) < 2:
		print("Usage: python3 get_features_clustering.py filename.csv")
		sys.exit(1)
	
	print("Loading data...")
	DataSet = pd.read_csv(sys.argv[1], low_memory = False) 
	print("Calculating...")
	Cluster = GetClusterFeature(DataSet)
	Cluster_final = Filter(Cluster)
	filename = sys.argv[1].replace("_HDB_5.csv", "")
	Cluster.to_csv(filename + str('_clu_features_unclean_5.csv'), index=None)
	Cluster_final.to_csv(filename + str('_clu_features_fin_5.csv'), index=None)
	print("Complete!!!")
# end of main()

if __name__ == '__main__':
	main()


# @nb.jit
# def CountFeature(data):
# 	feature = GetFeature(data)
# 	result = feature.groupby("DB_labels").aggregate(['sum', 'max', 'min', 'mean', 'std'])
# 	Flow = FlowGetFeature(data)
# 	for i in range(-1, len(result.index)):
# 		j=i+1
# 		print("i = " + str(i) + ", j = " + str(j))
# 		# Duration
# 		Flow.loc[Flow['DB_labels'] == i, "Duration_sum"] = result.iloc[j]['Duration']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Duration_max"] = result.iloc[j]['Duration']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Duration_min"] = result.iloc[j]['Duration']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Duration_std"] = result.iloc[j]['Duration']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Duration_mean"] = result.iloc[j]['Duration']['mean']
# 		# IAT_Sum
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Sum_sum"] = result.iloc[j]['IAT_Sum']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Sum_max"] = result.iloc[j]['IAT_Sum']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Sum_min"] = result.iloc[j]['IAT_Sum']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Sum_std"] = result.iloc[j]['IAT_Sum']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Sum_mean"] = result.iloc[j]['IAT_Sum']['mean']
# 		# IAT_Mean
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Mean_sum"] = result.iloc[j]['IAT_Mean']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Mean_max"] = result.iloc[j]['IAT_Mean']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Mean_min"] = result.iloc[j]['IAT_Mean']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Mean_std"] = result.iloc[j]['IAT_Mean']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Mean_mean"] = result.iloc[j]['IAT_Mean']['mean']
# 		# IAT_Std
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Std_sum"] = result.iloc[j]['IAT_Std']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Std_max"] = result.iloc[j]['IAT_Std']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Std_min"] = result.iloc[j]['IAT_Std']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Std_std"] = result.iloc[j]['IAT_Std']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Std_mean"] = result.iloc[j]['IAT_Std']['mean']
# 		# IAT_Max
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Max_sum"] = result.iloc[j]['IAT_Max']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Max_max"] = result.iloc[j]['IAT_Max']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Max_min"] = result.iloc[j]['IAT_Max']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Max_std"] = result.iloc[j]['IAT_Max']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Max_mean"] = result.iloc[j]['IAT_Max']['mean']
# 		# IAT_Min
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Min_sum"] = result.iloc[j]['IAT_Min']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Min_max"] = result.iloc[j]['IAT_Min']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Min_min"] = result.iloc[j]['IAT_Min']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Min_std"] = result.iloc[j]['IAT_Min']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "IAT_Min_mean"] = result.iloc[j]['IAT_Min']['mean']
# 		# Length_Min
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Min_sum"] = result.iloc[j]['Length_Min']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Min_max"] = result.iloc[j]['Length_Min']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Min_min"] = result.iloc[j]['Length_Min']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Min_std"] = result.iloc[j]['Length_Min']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Min_mean"] = result.iloc[j]['Length_Min']['mean']
# 		# Length_Max
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Max_sum"] = result.iloc[j]['Length_Max']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Max_max"] = result.iloc[j]['Length_Max']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Max_min"] = result.iloc[j]['Length_Max']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Max_std"] = result.iloc[j]['Length_Max']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Max_mean"] = result.iloc[j]['Length_Max']['mean']
# 		# Length_Mean
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Mean_sum"] = result.iloc[j]['Length_Mean']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Mean_max"] = result.iloc[j]['Length_Mean']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Mean_min"] = result.iloc[j]['Length_Mean']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Mean_std"] = result.iloc[j]['Length_Mean']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Mean_mean"] = result.iloc[j]['Length_Mean']['mean']
# 		# Length_Std
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Std_sum"] = result.iloc[j]['Length_Std']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Std_max"] = result.iloc[j]['Length_Std']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Std_min"] = result.iloc[j]['Length_Std']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Std_std"] = result.iloc[j]['Length_Std']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Std_mean"] = result.iloc[j]['Length_Std']['mean']
# 		# Length_Sum
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Sum_sum"] = result.iloc[j]['Length_Sum']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Sum_max"] = result.iloc[j]['Length_Sum']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Sum_min"] = result.iloc[j]['Length_Sum']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Sum_std"] = result.iloc[j]['Length_Sum']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_Sum_mean"] = result.iloc[j]['Length_Sum']['mean']
# 		# Length_First
# 		Flow.loc[Flow['DB_labels'] == i, "Length_First_sum"] = result.iloc[j]['Length_First']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_First_max"] = result.iloc[j]['Length_First']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_First_min"] = result.iloc[j]['Length_First']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_First_std"] = result.iloc[j]['Length_First']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Length_First_mean"] = result.iloc[j]['Length_First']['mean']
# 		# Total_Packets
# 		Flow.loc[Flow['DB_labels'] == i, "Total_Packets_sum"] = result.iloc[j]['Total_Packets']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Total_Packets_max"] = result.iloc[j]['Total_Packets']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Total_Packets_min"] = result.iloc[j]['Total_Packets']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Total_Packets_std"] = result.iloc[j]['Total_Packets']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Total_Packets_mean"] = result.iloc[j]['Total_Packets']['mean']
# 		# Num_Dir_Change
# 		Flow.loc[Flow['DB_labels'] == i, "Num_Dir_Change_sum"] = result.iloc[j]['Num_Dir_Change']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Num_Dir_Change_max"] = result.iloc[j]['Num_Dir_Change']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Num_Dir_Change_min"] = result.iloc[j]['Num_Dir_Change']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Num_Dir_Change_std"] = result.iloc[j]['Num_Dir_Change']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Num_Dir_Change_mean"] = result.iloc[j]['Num_Dir_Change']['mean']
# 		# Freq_Dir_Change
# 		Flow.loc[Flow['DB_labels'] == i, "Freq_Dir_Change_sum"] = result.iloc[j]['Freq_Dir_Change']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Freq_Dir_Change_max"] = result.iloc[j]['Freq_Dir_Change']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Freq_Dir_Change_min"] = result.iloc[j]['Freq_Dir_Change']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Freq_Dir_Change_std"] = result.iloc[j]['Freq_Dir_Change']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Freq_Dir_Change_mean"] = result.iloc[j]['Freq_Dir_Change']['mean']
# 		# Fwd_Total_Packets
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Total_Packets_sum"] = result.iloc[j]['Fwd_Total_Packets']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Total_Packets_max"] = result.iloc[j]['Fwd_Total_Packets']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Total_Packets_min"] = result.iloc[j]['Fwd_Total_Packets']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Total_Packets_std"] = result.iloc[j]['Fwd_Total_Packets']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Total_Packets_mean"] = result.iloc[j]['Fwd_Total_Packets']['mean']
# 		# Fwd_Length_Max
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Max_sum"] = result.iloc[j]['Fwd_Length_Max']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Max_max"] = result.iloc[j]['Fwd_Length_Max']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Max_min"] = result.iloc[j]['Fwd_Length_Max']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Max_std"] = result.iloc[j]['Fwd_Length_Max']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Max_mean"] = result.iloc[j]['Fwd_Length_Max']['mean']
# 		# Fwd_Length_Min
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Min_sum"] = result.iloc[j]['Fwd_Length_Min']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Min_max"] = result.iloc[j]['Fwd_Length_Min']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Min_min"] = result.iloc[j]['Fwd_Length_Min']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Min_std"] = result.iloc[j]['Fwd_Length_Min']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Min_mean"] = result.iloc[j]['Fwd_Length_Min']['mean']
# 		# Fwd_Length_Mean
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Mean_sum"] = result.iloc[j]['Fwd_Length_Mean']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Mean_max"] = result.iloc[j]['Fwd_Length_Mean']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Mean_min"] = result.iloc[j]['Fwd_Length_Mean']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Mean_std"] = result.iloc[j]['Fwd_Length_Mean']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Mean_mean"] = result.iloc[j]['Fwd_Length_Mean']['mean']
# 		# Fwd_Length_Std
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Std_sum"] = result.iloc[j]['Fwd_Length_Std']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Std_max"] = result.iloc[j]['Fwd_Length_Std']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Std_min"] = result.iloc[j]['Fwd_Length_Std']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Std_std"] = result.iloc[j]['Fwd_Length_Std']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Std_mean"] = result.iloc[j]['Fwd_Length_Std']['mean']
# 		# Fwd_IAT_Sum
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Sum_sum"] = result.iloc[j]['Fwd_IAT_Sum']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Sum_max"] = result.iloc[j]['Fwd_IAT_Sum']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Sum_min"] = result.iloc[j]['Fwd_IAT_Sum']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Sum_std"] = result.iloc[j]['Fwd_IAT_Sum']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Sum_mean"] = result.iloc[j]['Fwd_IAT_Sum']['mean']
# 		# Fwd_IAT_Mean
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Mean_sum"] = result.iloc[j]['Fwd_IAT_Mean']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Mean_max"] = result.iloc[j]['Fwd_IAT_Mean']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Mean_min"] = result.iloc[j]['Fwd_IAT_Mean']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Mean_std"] = result.iloc[j]['Fwd_IAT_Mean']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Mean_mean"] = result.iloc[j]['Fwd_IAT_Mean']['mean']
# 		# Fwd_IAT_Std
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Std_sum"] = result.iloc[j]['Fwd_IAT_Std']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Std_max"] = result.iloc[j]['Fwd_IAT_Std']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Std_min"] = result.iloc[j]['Fwd_IAT_Std']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Std_std"] = result.iloc[j]['Fwd_IAT_Std']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Std_mean"] = result.iloc[j]['Fwd_IAT_Std']['mean']
# 		# Fwd_IAT_Max
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Max_sum"] = result.iloc[j]['Fwd_IAT_Max']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Max_max"] = result.iloc[j]['Fwd_IAT_Max']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Max_min"] = result.iloc[j]['Fwd_IAT_Max']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Max_std"] = result.iloc[j]['Fwd_IAT_Max']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Max_mean"] = result.iloc[j]['Fwd_IAT_Max']['mean']
# 		# Fwd_IAT_Min
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Min_sum"] = result.iloc[j]['Fwd_IAT_Min']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Min_max"] = result.iloc[j]['Fwd_IAT_Min']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Min_min"] = result.iloc[j]['Fwd_IAT_Min']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Min_std"] = result.iloc[j]['Fwd_IAT_Min']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_IAT_Min_mean"] = result.iloc[j]['Fwd_IAT_Min']['mean']
# 		# Fwd_Length_First
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_First_sum"] = result.iloc[j]['Fwd_Length_First']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_First_max"] = result.iloc[j]['Fwd_Length_First']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_First_min"] = result.iloc[j]['Fwd_Length_First']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_First_std"] = result.iloc[j]['Fwd_Length_First']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_First_mean"] = result.iloc[j]['Fwd_Length_First']['mean']
# 		# Fwd_Length_Sum
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Sum_sum"] = result.iloc[j]['Fwd_Length_Sum']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Sum_max"] = result.iloc[j]['Fwd_Length_Sum']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Sum_min"] = result.iloc[j]['Fwd_Length_Sum']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Sum_std"] = result.iloc[j]['Fwd_Length_Sum']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Fwd_Length_Sum_mean"] = result.iloc[j]['Fwd_Length_Sum']['mean']
# 		# Bwd_Total_Packets
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Total_Packets_sum"] = result.iloc[j]['Bwd_Total_Packets']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Total_Packets_max"] = result.iloc[j]['Bwd_Total_Packets']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Total_Packets_min"] = result.iloc[j]['Bwd_Total_Packets']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Total_Packets_std"] = result.iloc[j]['Bwd_Total_Packets']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Total_Packets_mean"] = result.iloc[j]['Bwd_Total_Packets']['mean']
# 		# Bwd_Length_Max
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Max_sum"] = result.iloc[j]['Bwd_Length_Max']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Max_max"] = result.iloc[j]['Bwd_Length_Max']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Max_min"] = result.iloc[j]['Bwd_Length_Max']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Max_std"] = result.iloc[j]['Bwd_Length_Max']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Max_mean"] = result.iloc[j]['Bwd_Length_Max']['mean']
# 		# Bwd_Length_Min
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Min_sum"] = result.iloc[j]['Bwd_Length_Min']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Min_max"] = result.iloc[j]['Bwd_Length_Min']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Min_min"] = result.iloc[j]['Bwd_Length_Min']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Min_std"] = result.iloc[j]['Bwd_Length_Min']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Min_mean"] = result.iloc[j]['Bwd_Length_Min']['mean']
# 		# Bwd_Length_Mean
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Mean_sum"] = result.iloc[j]['Bwd_Length_Mean']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Mean_max"] = result.iloc[j]['Bwd_Length_Mean']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Mean_min"] = result.iloc[j]['Bwd_Length_Mean']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Mean_std"] = result.iloc[j]['Bwd_Length_Mean']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Mean_mean"] = result.iloc[j]['Bwd_Length_Mean']['mean']
# 		# Bwd_Length_Std
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Std_sum"] = result.iloc[j]['Bwd_Length_Std']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Std_max"] = result.iloc[j]['Bwd_Length_Std']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Std_min"] = result.iloc[j]['Bwd_Length_Std']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Std_std"] = result.iloc[j]['Bwd_Length_Std']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Std_mean"] = result.iloc[j]['Bwd_Length_Std']['mean']
# 		# Bwd_IAT_Sum
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Sum_sum"] = result.iloc[j]['Bwd_IAT_Sum']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Sum_max"] = result.iloc[j]['Bwd_IAT_Sum']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Sum_min"] = result.iloc[j]['Bwd_IAT_Sum']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Sum_std"] = result.iloc[j]['Bwd_IAT_Sum']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Sum_mean"] = result.iloc[j]['Bwd_IAT_Sum']['mean']
# 		# Bwd_IAT_Mean
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Mean_sum"] = result.iloc[j]['Bwd_IAT_Mean']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Mean_max"] = result.iloc[j]['Bwd_IAT_Mean']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Mean_min"] = result.iloc[j]['Bwd_IAT_Mean']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Mean_std"] = result.iloc[j]['Bwd_IAT_Mean']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Mean_mean"] = result.iloc[j]['Bwd_IAT_Mean']['mean']
# 		# Bwd_IAT_Std
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Std_sum"] = result.iloc[j]['Bwd_IAT_Std']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Std_max"] = result.iloc[j]['Bwd_IAT_Std']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Std_min"] = result.iloc[j]['Bwd_IAT_Std']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Std_std"] = result.iloc[j]['Bwd_IAT_Std']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Std_mean"] = result.iloc[j]['Bwd_IAT_Std']['mean']
# 		# Bwd_IAT_Max
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Max_sum"] = result.iloc[j]['Bwd_IAT_Max']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Max_max"] = result.iloc[j]['Bwd_IAT_Max']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Max_min"] = result.iloc[j]['Bwd_IAT_Max']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Max_std"] = result.iloc[j]['Bwd_IAT_Max']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Max_mean"] = result.iloc[j]['Bwd_IAT_Max']['mean']
# 		# Bwd_IAT_Min
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Min_sum"] = result.iloc[j]['Bwd_IAT_Min']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Min_max"] = result.iloc[j]['Bwd_IAT_Min']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Min_min"] = result.iloc[j]['Bwd_IAT_Min']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Min_std"] = result.iloc[j]['Bwd_IAT_Min']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_IAT_Min_mean"] = result.iloc[j]['Bwd_IAT_Min']['mean']
# 		# Bwd_Length_First
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_First_sum"] = result.iloc[j]['Bwd_Length_First']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_First_max"] = result.iloc[j]['Bwd_Length_First']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_First_min"] = result.iloc[j]['Bwd_Length_First']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_First_std"] = result.iloc[j]['Bwd_Length_First']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_First_mean"] = result.iloc[j]['Bwd_Length_First']['mean']
# 		# Bwd_Length_Sum
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Sum_sum"] = result.iloc[j]['Bwd_Length_Sum']['sum']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Sum_max"] = result.iloc[j]['Bwd_Length_Sum']['max']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Sum_min"] = result.iloc[j]['Bwd_Length_Sum']['min']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Sum_std"] = result.iloc[j]['Bwd_Length_Sum']['std']
# 		Flow.loc[Flow['DB_labels'] == i, "Bwd_Length_Sum_mean"] = result.iloc[j]['Bwd_Length_Sum']['mean']
# 	return Flow
# End of CountFeature()

# @nb.jit(nopython=True)
# def CountFeatureLong(data):
# 	clu = data.groupby('DB_labels')
# 	for i in range(-1, len(clu)):
# 		j=i+1
# 		print("i = " + str(i) + ", j = " + str(j))
# 		group = data.loc[data["DB_Label"] == i]
# 		# Duration
# 		Duration_sum = group["Duration"].sum()
# 		Duration_max = group["Duration"].max()
# 		Duration_min = group["Duration"].min()
# 		Duration_std = group["Duration"].std()
# 		Duration_mean = group["Duration"].mean()
# 		# IAT_Mean
# 		IAT_Mean_sum = group["IAT_Mean"].sum()
# 		IAT_Mean_max = group["IAT_Mean"].max()
# 		IAT_Mean_min = group["IAT_Mean"].min()
# 		IAT_Mean_std = group["IAT_Mean"].std()
# 		IAT_Mean_mean = group["IAT_Mean"].mean()
# 		# IAT_Std
# 		IAT_Std_sum = group["IAT_Std"].sum()
# 		IAT_Std_max = group["IAT_Std"].max()
# 		IAT_Std_min = group["IAT_Std"].min()
# 		IAT_Std_std = group["IAT_Std"].std()
# 		IAT_Std_mean = group["IAT_Std"].mean()
# 		# IAT_Max
# 		IAT_Max_sum = group["IAT_Max"].sum()
# 		IAT_Max_max = group["IAT_Max"].max()
# 		IAT_Max_min = group["IAT_Max"].min()
# 		IAT_Max_std = group["IAT_Max"].std()
# 		IAT_Max_mean = group["IAT_Max"].mean()
# 		# IAT_Min
# 		IAT_Min_sum = group["IAT_Min"].sum()
# 		IAT_Min_max = group["IAT_Min"].max()
# 		IAT_Min_min = group["IAT_Min"].min()
# 		IAT_Min_std = group["IAT_Min"].std()
# 		IAT_Min_mean = group["IAT_Min"].mean()
# 		# Length_Min
# 		Length_Min_sum = group["Length_Min"].sum()
# 		Length_Min_max = group["Length_Min"].max()
# 		Length_Min_min = group["Length_Min"].min()
# 		Length_Min_std = group["Length_Min"].std()
# 		Length_Min_mean = group["Length_Min"].mean()
# 		# Length_Max
# 		Length_Max_sum = group["Length_Max"].sum()
# 		Length_Max_max = group["Length_Max"].max()
# 		Length_Max_min = group["Length_Max"].min()
# 		Length_Max_std = group["Length_Max"].std()
# 		Length_Max_mean = group["Length_Max"].mean()
# 		# Length_Mean
# 		Length_Mean_sum = group["Length_Mean"].sum()
# 		Length_Mean_max = group["Length_Mean"].max()
# 		Length_Mean_min = group["Length_Mean"].min()
# 		Length_Mean_std = group["Length_Mean"].std()
# 		Length_Mean_mean = group["Length_Mean"].mean()
# 		# Length_Std
# 		Length_Std_sum = group["Length_Std"].sum()
# 		Length_Std_max = group["Length_Std"].max()
# 		Length_Std_min = group["Length_Std"].min()
# 		Length_Std_std = group["Length_Std"].std()
# 		Length_Std_mean = group["Length_Std"].mean()
# 		# Total_Packets
# 		Total_Packets_sum = group["Total_Packets"].sum()
# 		Total_Packets_max = group["Total_Packets"].max()
# 		Total_Packets_min = group["Total_Packets"].min()
# 		Total_Packets_std = group["Total_Packets"].std()
# 		Total_Packets_mean = group["Total_Packets"].mean()
# 		# IAT_Sum
# 		IAT_Sum_sum = group["IAT_Sum"].sum()
# 		IAT_Sum_max = group["IAT_Sum"].max()
# 		IAT_Sum_min = group["IAT_Sum"].min()
# 		IAT_Sum_std = group["IAT_Sum"].std()
# 		IAT_Sum_mean = group["IAT_Sum"].mean()
# 		# Length_First
# 		Length_First_sum = group["Length_First"].sum()
# 		Length_First_max = group["Length_First"].max()
# 		Length_First_min = group["Length_First"].min()
# 		Length_First_std = group["Length_First"].std()
# 		Length_First_mean = group["Length_First"].mean()
# 		# Length_Sum
# 		Length_Sum_sum = group["Length_Sum"].sum()
# 		Length_Sum_max = group["Length_Sum"].max()
# 		Length_Sum_min = group["Length_Sum"].min()
# 		Length_Sum_std = group["Length_Sum"].std()
# 		Length_Sum_mean = group["Length_Sum"].mean()
# 		# Num_Dir_Change
# 		Num_Dir_Change_sum = group["Num_Dir_Change"].sum()
# 		Num_Dir_Change_max = group["Num_Dir_Change"].max()
# 		Num_Dir_Change_min = group["Num_Dir_Change"].min()
# 		Num_Dir_Change_std = group["Num_Dir_Change"].std()
# 		Num_Dir_Change_mean = group["Num_Dir_Change"].mean()
# 		# Freq_Dir_Change
# 		Freq_Dir_Change_sum = group["Freq_Dir_Change"].sum()
# 		Freq_Dir_Change_max = group["Freq_Dir_Change"].max()
# 		Freq_Dir_Change_min = group["Freq_Dir_Change"].min()
# 		Freq_Dir_Change_std = group["Freq_Dir_Change"].std()
# 		Freq_Dir_Change_mean = group["Freq_Dir_Change"].mean()
# 		# Fwd_Total_Packets
# 		Fwd_Total_Packets_sum = group["Fwd_Total_Packets"].sum()
# 		Fwd_Total_Packets_max = group["Fwd_Total_Packets"].max()
# 		Fwd_Total_Packets_min = group["Fwd_Total_Packets"].min()
# 		Fwd_Total_Packets_std = group["Fwd_Total_Packets"].std()
# 		Fwd_Total_Packets_mean = group["Fwd_Total_Packets"].mean()
# 		# Fwd_Length_Max
# 		Fwd_Length_Max_sum = group["Fwd_Length_Max"].sum()
# 		Fwd_Length_Max_max = group["Fwd_Length_Max"].max()
# 		Fwd_Length_Max_min = group["Fwd_Length_Max"].min()
# 		Fwd_Length_Max_std = group["Fwd_Length_Max"].std()
# 		Fwd_Length_Max_mean = group["Fwd_Length_Max"].mean()
# 		# Fwd_Length_Min
# 		Fwd_Length_Min_sum = group["Fwd_Length_Min"].sum()
# 		Fwd_Length_Min_max = group["Fwd_Length_Min"].max()
# 		Fwd_Length_Min_min = group["Fwd_Length_Min"].min()
# 		Fwd_Length_Min_std = group["Fwd_Length_Min"].std()
# 		Fwd_Length_Min_mean = group["Fwd_Length_Min"].mean()
# 		# Fwd_Length_Mean
# 		Fwd_Length_Mean_sum = group["Fwd_Length_Mean"].sum()
# 		Fwd_Length_Mean_max = group["Fwd_Length_Mean"].max()
# 		Fwd_Length_Mean_min = group["Fwd_Length_Mean"].min()
# 		Fwd_Length_Mean_std = group["Fwd_Length_Mean"].std()
# 		Fwd_Length_Mean_mean = group["Fwd_Length_Mean"].mean()
# 		# Fwd_Length_Std
# 		Fwd_Length_Std_sum = group["Fwd_Length_Std"].sum()
# 		Fwd_Length_Std_max = group["Fwd_Length_Std"].max()
# 		Fwd_Length_Std_min = group["Fwd_Length_Std"].min()
# 		Fwd_Length_Std_std = group["Fwd_Length_Std"].std()
# 		Fwd_Length_Std_mean = group["Fwd_Length_Std"].mean()
# 		# Fwd_IAT_Sum
# 		Fwd_IAT_Sum_sum = group["Fwd_IAT_Sum"].sum()
# 		Fwd_IAT_Sum_max = group["Fwd_IAT_Sum"].max()
# 		Fwd_IAT_Sum_min = group["Fwd_IAT_Sum"].min()
# 		Fwd_IAT_Sum_std = group["Fwd_IAT_Sum"].std()
# 		Fwd_IAT_Sum_mean = group["Fwd_IAT_Sum"].mean()
# 		# Fwd_IAT_Mean
# 		Fwd_IAT_Mean_sum = group["Fwd_IAT_Mean"].sum()
# 		Fwd_IAT_Mean_max = group["Fwd_IAT_Mean"].max()
# 		Fwd_IAT_Mean_min = group["Fwd_IAT_Mean"].min()
# 		Fwd_IAT_Mean_std = group["Fwd_IAT_Mean"].std()
# 		Fwd_IAT_Mean_mean = group["Fwd_IAT_Mean"].mean()
# 		# Fwd_IAT_Std
# 		Fwd_IAT_Std_sum = group["Fwd_IAT_Std"].sum()
# 		Fwd_IAT_Std_max = group["Fwd_IAT_Std"].max()
# 		Fwd_IAT_Std_min = group["Fwd_IAT_Std"].min()
# 		Fwd_IAT_Std_std = group["Fwd_IAT_Std"].std()
# 		Fwd_IAT_Std_mean = group["Fwd_IAT_Std"].mean()
# 		# Fwd_IAT_Max
# 		Fwd_IAT_Max_sum = group["Fwd_IAT_Max"].sum()
# 		Fwd_IAT_Max_max = group["Fwd_IAT_Max"].max()
# 		Fwd_IAT_Max_min = group["Fwd_IAT_Max"].min()
# 		Fwd_IAT_Max_std = group["Fwd_IAT_Max"].std()
# 		Fwd_IAT_Max_mean = group["Fwd_IAT_Max"].mean()
# 		# Fwd_IAT_Min
# 		Fwd_IAT_Min_sum = group["Fwd_IAT_Min"].sum()
# 		Fwd_IAT_Min_max = group["Fwd_IAT_Min"].max()
# 		Fwd_IAT_Min_min = group["Fwd_IAT_Min"].min()
# 		Fwd_IAT_Min_std = group["Fwd_IAT_Min"].std()
# 		Fwd_IAT_Min_mean = group["Fwd_IAT_Min"].mean()
# 		# Fwd_Length_First
# 		Fwd_Length_First_sum = group["Fwd_Length_First"].sum()
# 		Fwd_Length_First_max = group["Fwd_Length_First"].max()
# 		Fwd_Length_First_min = group["Fwd_Length_First"].min()
# 		Fwd_Length_First_std = group["Fwd_Length_First"].std()
# 		Fwd_Length_First_mean = group["Fwd_Length_First"].mean()
# 		# Fwd_Length_Sum
# 		Fwd_Length_Sum_sum = group["Fwd_Length_Sum"].sum()
# 		Fwd_Length_Sum_max = group["Fwd_Length_Sum"].max()
# 		Fwd_Length_Sum_min = group["Fwd_Length_Sum"].min()
# 		Fwd_Length_Sum_std = group["Fwd_Length_Sum"].std()
# 		Fwd_Length_Sum_mean = group["Fwd_Length_Sum"].mean()
# 		# Bwd_Total_Packets
# 		Bwd_Total_Packets_sum = group["Bwd_Total_Packets"].sum()
# 		Bwd_Total_Packets_max = group["Bwd_Total_Packets"].max()
# 		Bwd_Total_Packets_min = group["Bwd_Total_Packets"].min()
# 		Bwd_Total_Packets_std = group["Bwd_Total_Packets"].std()
# 		Bwd_Total_Packets_mean = group["Bwd_Total_Packets"].mean()
# 		# Bwd_Length_Max
# 		Bwd_Length_Max_sum = group["Bwd_Length_Max"].sum()
# 		Bwd_Length_Max_max = group["Bwd_Length_Max"].max()
# 		Bwd_Length_Max_min = group["Bwd_Length_Max"].min()
# 		Bwd_Length_Max_std = group["Bwd_Length_Max"].std()
# 		Bwd_Length_Max_mean = group["Bwd_Length_Max"].mean()
# 		# Bwd_Length_Min
# 		Bwd_Length_Min_sum = group["Bwd_Length_Min"].sum()
# 		Bwd_Length_Min_max = group["Bwd_Length_Min"].max()
# 		Bwd_Length_Min_min = group["Bwd_Length_Min"].min()
# 		Bwd_Length_Min_std = group["Bwd_Length_Min"].std()
# 		Bwd_Length_Min_mean = group["Bwd_Length_Min"].mean()
# 		# Bwd_Length_Mean
# 		Bwd_Length_Mean_sum = group["Bwd_Length_Mean"].sum()
# 		Bwd_Length_Mean_max = group["Bwd_Length_Mean"].max()
# 		Bwd_Length_Mean_min = group["Bwd_Length_Mean"].min()
# 		Bwd_Length_Mean_std = group["Bwd_Length_Mean"].std()
# 		Bwd_Length_Mean_mean = group["Bwd_Length_Mean"].mean()
# 		# Bwd_Length_Std
# 		Bwd_Length_Std_sum = group["Bwd_Length_Std"].sum()
# 		Bwd_Length_Std_max = group["Bwd_Length_Std"].max()
# 		Bwd_Length_Std_min = group["Bwd_Length_Std"].min()
# 		Bwd_Length_Std_std = group["Bwd_Length_Std"].std()
# 		Bwd_Length_Std_mean = group["Bwd_Length_Std"].mean()
# 		# Bwd_IAT_Sum
# 		Bwd_IAT_Sum_sum = group["Bwd_IAT_Sum"].sum()
# 		Bwd_IAT_Sum_max = group["Bwd_IAT_Sum"].max()
# 		Bwd_IAT_Sum_min = group["Bwd_IAT_Sum"].min()
# 		Bwd_IAT_Sum_std = group["Bwd_IAT_Sum"].std()
# 		Bwd_IAT_Sum_mean = group["Bwd_IAT_Sum"].mean()
# 		# Bwd_IAT_Mean
# 		Bwd_IAT_Mean_sum = group["Bwd_IAT_Mean"].sum()
# 		Bwd_IAT_Mean_max = group["Bwd_IAT_Mean"].max()
# 		Bwd_IAT_Mean_min = group["Bwd_IAT_Mean"].min()
# 		Bwd_IAT_Mean_std = group["Bwd_IAT_Mean"].std()
# 		Bwd_IAT_Mean_mean = group["Bwd_IAT_Mean"].mean()
# 		# Bwd_IAT_Std
# 		Bwd_IAT_Std_sum = group["Bwd_IAT_Std"].sum()
# 		Bwd_IAT_Std_max = group["Bwd_IAT_Std"].max()
# 		Bwd_IAT_Std_min = group["Bwd_IAT_Std"].min()
# 		Bwd_IAT_Std_std = group["Bwd_IAT_Std"].std()
# 		Bwd_IAT_Std_mean = group["Bwd_IAT_Std"].mean()
# 		# Bwd_IAT_Max
# 		Bwd_IAT_Max_sum = group["Bwd_IAT_Max"].sum()
# 		Bwd_IAT_Max_max = group["Bwd_IAT_Max"].max()
# 		Bwd_IAT_Max_min = group["Bwd_IAT_Max"].min()
# 		Bwd_IAT_Max_std = group["Bwd_IAT_Max"].std()
# 		Bwd_IAT_Max_mean = group["Bwd_IAT_Max"].mean()
# 		# Bwd_IAT_Min
# 		Bwd_IAT_Min_sum = group["Bwd_IAT_Min"].sum()
# 		Bwd_IAT_Min_max = group["Bwd_IAT_Min"].max()
# 		Bwd_IAT_Min_min = group["Bwd_IAT_Min"].min()
# 		Bwd_IAT_Min_std = group["Bwd_IAT_Min"].std()
# 		Bwd_IAT_Min_mean = group["Bwd_IAT_Min"].mean()
# 		# Bwd_Length_First
# 		Bwd_Length_First_sum = group["Bwd_Length_First"].sum()
# 		Bwd_Length_First_max = group["Bwd_Length_First"].max()
# 		Bwd_Length_First_min = group["Bwd_Length_First"].min()
# 		Bwd_Length_First_std = group["Bwd_Length_First"].std()
# 		Bwd_Length_First_mean = group["Bwd_Length_First"].mean()
# 		# Bwd_Length_Sum
# 		Bwd_Length_Sum_sum = group["Bwd_Length_Sum"].sum()
# 		Bwd_Length_Sum_max = group["Bwd_Length_Sum"].max()
# 		Bwd_Length_Sum_min = group["Bwd_Length_Sum"].min()
# 		Bwd_Length_Sum_std = group["Bwd_Length_Sum"].std()
# 		Bwd_Length_Sum_mean = group["Bwd_Length_Sum"].mean()
		

# 		
# 		# Get Flow & input new feature
# 		Flow = FlowGetFeature(data)
		
# 		# Duration
# 		Flow["Duration_sum"] = Duration_sum
# 		Flow["Duration_max"] = Duration_max
# 		Flow["Duration_min"] = Duration_min
# 		Flow["Duration_std"] = Duration_std
# 		Flow["Duration_mean"] = Duration_mean
# 		# IAT_Mean
# 		Flow["IAT_Mean_sum"] = IAT_Mean_sum
# 		Flow["IAT_Mean_max"] = IAT_Mean_max
# 		Flow["IAT_Mean_min"] = IAT_Mean_min
# 		Flow["IAT_Mean_std"] = IAT_Mean_std
# 		Flow["IAT_Mean_mean"] = IAT_Mean_mean
# 		# IAT_Std
# 		Flow["IAT_Std_sum"] = IAT_Std_sum
# 		Flow["IAT_Std_max"] = IAT_Std_max
# 		Flow["IAT_Std_min"] = IAT_Std_min
# 		Flow["IAT_Std_std"] = IAT_Std_std
# 		Flow["IAT_Std_mean"] = IAT_Std_mean
# 		# IAT_Max
# 		Flow["IAT_Max_sum"] = IAT_Max_sum
# 		Flow["IAT_Max_max"] = IAT_Max_max
# 		Flow["IAT_Max_min"] = IAT_Max_min
# 		Flow["IAT_Max_std"] = IAT_Max_std
# 		Flow["IAT_Max_mean"] = IAT_Max_mean
# 		# IAT_Min
# 		Flow["IAT_Min_sum"] = IAT_Min_sum
# 		Flow["IAT_Min_max"] = IAT_Min_max
# 		Flow["IAT_Min_min"] = IAT_Min_min
# 		Flow["IAT_Min_std"] = IAT_Min_std
# 		Flow["IAT_Min_mean"] = IAT_Min_mean
# 		# Length_Min
# 		Flow["Length_Min_sum"] = Length_Min_sum
# 		Flow["Length_Min_max"] = Length_Min_max
# 		Flow["Length_Min_min"] = Length_Min_min
# 		Flow["Length_Min_std"] = Length_Min_std
# 		Flow["Length_Min_mean"] = Length_Min_mean
# 		# Length_Max
# 		Flow["Length_Max_sum"] = Length_Max_sum
# 		Flow["Length_Max_max"] = Length_Max_max
# 		Flow["Length_Max_min"] = Length_Max_min
# 		Flow["Length_Max_std"] = Length_Max_std
# 		Flow["Length_Max_mean"] = Length_Max_mean
# 		# Length_Mean
# 		Flow["Length_Mean_sum"] = Length_Mean_sum
# 		Flow["Length_Mean_max"] = Length_Mean_max
# 		Flow["Length_Mean_min"] = Length_Mean_min
# 		Flow["Length_Mean_std"] = Length_Mean_std
# 		Flow["Length_Mean_mean"] = Length_Mean_mean
# 		# Length_Std
# 		Flow["Length_Std_sum"] = Length_Std_sum
# 		Flow["Length_Std_max"] = Length_Std_max
# 		Flow["Length_Std_min"] = Length_Std_min
# 		Flow["Length_Std_std"] = Length_Std_std
# 		Flow["Length_Std_mean"] = Length_Std_mean
# 		# Total_Packets
# 		Flow["Total_Packets_sum"] = Total_Packets_sum
# 		Flow["Total_Packets_max"] = Total_Packets_max
# 		Flow["Total_Packets_min"] = Total_Packets_min
# 		Flow["Total_Packets_std"] = Total_Packets_std
# 		Flow["Total_Packets_mean"] = Total_Packets_mean
# 		# IAT_Sum
# 		Flow["IAT_Sum_sum"] = IAT_Sum_sum
# 		Flow["IAT_Sum_max"] = IAT_Sum_max
# 		Flow["IAT_Sum_min"] = IAT_Sum_min
# 		Flow["IAT_Sum_std"] = IAT_Sum_std
# 		Flow["IAT_Sum_mean"] = IAT_Sum_mean
# 		# IAT_Sum
# 		Flow["IAT_Sum_sum"] = IAT_Sum_sum
# 		Flow["IAT_Sum_max"] = IAT_Sum_max
# 		Flow["IAT_Sum_min"] = IAT_Sum_min
# 		Flow["IAT_Sum_std"] = IAT_Sum_std
# 		Flow["IAT_Sum_mean"] = IAT_Sum_mean
# 		# Length_First
# 		Flow["Length_First_sum"] = Length_First_sum
# 		Flow["Length_First_max"] = Length_First_max
# 		Flow["Length_First_min"] = Length_First_min
# 		Flow["Length_First_std"] = Length_First_std
# 		Flow["Length_First_mean"] = Length_First_mean
# 		# Length_Sum
# 		Flow["Length_Sum_sum"] = Length_Sum_sum
# 		Flow["Length_Su_max"] = Length_Sum_max
# 		Flow["Length_Sum_min"] = Length_Sum_min
# 		Flow["Length_Sum_std"] = Length_Sum_std
# 		Flow["Length_Sum_mean"] = Length_Sum_mean
# 		# Num_Dir_Change
# 		Flow["Num_Dir_Change_sum"] = Num_Dir_Change_sum
# 		Flow["Num_Dir_Change_max"] = Num_Dir_Change_max
# 		Flow["Num_Dir_Change_min"] = Num_Dir_Change_min
# 		Flow["Num_Dir_Change_std"] = Num_Dir_Change_std
# 		Flow["Num_Dir_Change_mean"] = Num_Dir_Change_mean
# 		# Freq_Dir_Change
# 		Flow["Freq_Dir_Change_sum"] = Freq_Dir_Change_sum
# 		Flow["Freq_Dir_Change_max"] = Freq_Dir_Change_max
# 		Flow["Freq_Dir_Change_min"] = Freq_Dir_Change_min
# 		Flow["Freq_Dir_Change_std"] = Freq_Dir_Change_std
# 		Flow["Freq_Dir_Change_mean"] = Freq_Dir_Change_mean
# 		# Fwd_Total_Packets
# 		Flow["Fwd_Total_Packets_sum"] = Fwd_Total_Packets_sum
# 		Flow["Fwd_Total_Packets_max"] = Fwd_Total_Packets_max
# 		Flow["Fwd_Total_Packets_min"] = Fwd_Total_Packets_min
# 		Flow["Fwd_Total_Packets_std"] = Fwd_Total_Packets_std
# 		Flow["Fwd_Total_Packets_mean"] = Fwd_Total_Packets_mean
# 		# Fwd_Length_Max
# 		Flow["Fwd_Length_Max_sum"] = Fwd_Length_Max_sum
# 		Flow["Fwd_Length_Max_max"] = Fwd_Length_Max_max
# 		Flow["Fwd_Length_Max_min"] = Fwd_Length_Max_min
# 		Flow["Fwd_Length_Max_std"] = Fwd_Length_Max_std
# 		Flow["Fwd_Length_Max_mean"] = Fwd_Length_Max_mean
# 		# Fwd_Length_Min
# 		Flow["Fwd_Length_Min_sum"] = Fwd_Length_Min_sum
# 		Flow["Fwd_Length_Min_max"] = Fwd_Length_Min_max
# 		Flow["Fwd_Length_Min_min"] = Fwd_Length_Min_min
# 		Flow["Fwd_Length_Min_std"] = Fwd_Length_Min_std
# 		Flow["Fwd_Length_Min_mean"] = Fwd_Length_Min_mean
# 		# Fwd_Length_Mean
# 		Flow["Fwd_Length_Mean_sum"] = Fwd_Length_Mean_sum
# 		Flow["Fwd_Length_Mean_max"] = Fwd_Length_Mean_max
# 		Flow["Fwd_Length_Mean_min"] = Fwd_Length_Mean_min
# 		Flow["Fwd_Length_Mean_std"] = Fwd_Length_Mean_std
# 		Flow["Fwd_Length_Mean_mean"] = Fwd_Length_Mean_mean
# 		# Fwd_IAT_Sum
# 		Flow["Fwd_IAT_Sum_sum"] = Fwd_IAT_Sum_sum
# 		Flow["Fwd_IAT_Sum_max"] = Fwd_IAT_Sum_max
# 		Flow["Fwd_IAT_Sum_min"] = Fwd_IAT_Sum_min
# 		Flow["Fwd_IAT_Sum_std"] = Fwd_IAT_Sum_std
# 		Flow["Fwd_IAT_Sum_mean"] = Fwd_IAT_Sum_mean
# 		# Fwd_Length_Std
# 		Flow["Fwd_Length_Std_sum"] = Fwd_Length_Std_sum
# 		Flow["Fwd_Length_Std_max"] = Fwd_Length_Std_max
# 		Flow["Fwd_Length_Std_min"] = Fwd_Length_Std_min
# 		Flow["Fwd_Length_Std_std"] = Fwd_Length_Std_std
# 		Flow["Fwd_Length_Std_mean"] = Fwd_Length_Std_mean
# 		# Fwd_IAT_Mean
# 		Flow["Fwd_IAT_Mean_sum"] = Fwd_IAT_Mean_sum
# 		Flow["Fwd_IAT_Mean_max"] = Fwd_IAT_Mean_max
# 		Flow["Fwd_IAT_Mean_min"] = Fwd_IAT_Mean_min
# 		Flow["Fwd_IAT_Mean_std"] = Fwd_IAT_Mean_std
# 		Flow["Fwd_IAT_Mean_mean"] = Fwd_IAT_Mean_mean
# 		# Fwd_IAT_Mean
# 		Flow["Fwd_IAT_Mean_sum"] = Fwd_IAT_Mean_sum
# 		Flow["Fwd_IAT_Mean_max"] = Fwd_IAT_Mean_max
# 		Flow["Fwd_IAT_Mean_min"] = Fwd_IAT_Mean_min
# 		Flow["Fwd_IAT_Mean_std"] = Fwd_IAT_Mean_std
# 		Flow["Fwd_IAT_Mean_mean"] = Fwd_IAT_Mean_mean
# 		# Fwd_IAT_Std
# 		Flow["Fwd_IAT_Std_sum"] = Fwd_IAT_Std_sum
# 		Flow["Fwd_IAT_Std_max"] = Fwd_IAT_Std_max
# 		Flow["Fwd_IAT_Std_min"] = Fwd_IAT_Std_min
# 		Flow["Fwd_IAT_Std_std"] = Fwd_IAT_Std_std
# 		Flow["Fwd_IAT_Std_mean"] = Fwd_IAT_Std_mean
# 		# Fwd_IAT_Max
# 		Flow["Fwd_IAT_Max_sum"] = Fwd_IAT_Max_sum
# 		Flow["Fwd_IAT_Max_max"] = Fwd_IAT_Max_max
# 		Flow["Fwd_IAT_Max_min"] = Fwd_IAT_Max_min
# 		Flow["Fwd_IAT_Max_std"] = Fwd_IAT_Max_std
# 		Flow["Fwd_IAT_Max_mean"] = Fwd_IAT_Max_mean
# 		# Fwd_IAT_Min
# 		Flow["Fwd_IAT_Min_sum"] = Fwd_IAT_Min_sum
# 		Flow["Fwd_IAT_Min_max"] = Fwd_IAT_Min_max
# 		Flow["Fwd_IAT_Min_min"] = Fwd_IAT_Min_min
# 		Flow["Fwd_IAT_Min_std"] = Fwd_IAT_Min_std
# 		Flow["Fwd_IAT_Min_mean"] = Fwd_IAT_Min_mean
# 		# Fwd_Length_First
# 		Flow["Fwd_Length_First_sum"] = Fwd_Length_First_sum
# 		Flow["Fwd_Length_First_max"] = Fwd_Length_First_max
# 		Flow["Fwd_Length_First_min"] = Fwd_Length_First_min
# 		Flow["Fwd_Length_First_std"] = Fwd_Length_First_std
# 		Flow["Fwd_Length_First_mean"] = Fwd_Length_First_mean
# 		# Fwd_Length_Sum
# 		Flow["Fwd_Length_Sum_sum"] = Fwd_Length_Sum_sum
# 		Flow["Fwd_Length_Sum_max"] = Fwd_Length_Sum_max
# 		Flow["Fwd_Length_Sum_min"] = Fwd_Length_Sum_min
# 		Flow["Fwd_Length_Sum_std"] = Fwd_Length_Sum_std
# 		Flow["Fwd_Length_Sum_mean"] = Fwd_Length_Sum_mean
# 		# Bwd_Total_Packets
# 		Flow["Bwd_Total_Packets_sum"] = Bwd_Total_Packets_sum
# 		Flow["Bwd_Total_Packets_max"] = Bwd_Total_Packets_max
# 		Flow["Bwd_Total_Packets_min"] = Bwd_Total_Packets_min
# 		Flow["Bwd_Total_Packets_std"] = Bwd_Total_Packets_std
# 		Flow["Bwd_Total_Packets_mean"] = Bwd_Total_Packets_mean
# 		# Bwd_Length_Max
# 		Flow["Bwd_Length_Max_sum"] = Bwd_Length_Max_sum
# 		Flow["Bwd_Length_Max_max"] = Bwd_Length_Max_max
# 		Flow["Bwd_Length_Max_min"] = Bwd_Length_Max_min
# 		Flow["Bwd_Length_Max_std"] = Bwd_Length_Max_std
# 		Flow["Bwd_Length_Max_mean"] = Bwd_Length_Max_mean
# 		# Bwd_Length_Min
# 		Flow["Bwd_Length_Min_sum"] = Bwd_Length_Min_sum
# 		Flow["Bwd_Length_Min_max"] = Bwd_Length_Min_max
# 		Flow["Bwd_Length_Min_min"] = Bwd_Length_Min_min
# 		Flow["Bwd_Length_Min_std"] = Bwd_Length_Min_std
# 		Flow["Bwd_Length_Min_mean"] = Bwd_Length_Min_mean
# 		# Bwd_Length_Mean
# 		Flow["Bwd_Length_Mean_sum"] = Bwd_Length_Mean_sum
# 		Flow["Bwd_Length_Mean_max"] = Bwd_Length_Mean_max
# 		Flow["Bwd_Length_Mean_min"] = Bwd_Length_Mean_min
# 		Flow["Bwd_Length_Mean_std"] = Bwd_Length_Mean_std
# 		Flow["Bwd_Length_Mean_mean"] = Bwd_Length_Mean_mean
# 		# Bwd_Length_Std
# 		Flow["Bwd_Length_Std_sum"] = Bwd_Length_Std_sum
# 		Flow["Bwd_Length_Std_max"] = Bwd_Length_Std_max
# 		Flow["Bwd_Length_Std_min"] = Bwd_Length_Std_min
# 		Flow["Bwd_Length_Std_std"] = Bwd_Length_Std_std
# 		Flow["Bwd_Length_Std_mean"] = Bwd_Length_Std_mean
# 		# Bwd_IAT_Sum
# 		Flow["Bwd_IAT_Sum_sum"] = Bwd_IAT_Sum_sum
# 		Flow["Bwd_IAT_Sum_max"] = Bwd_IAT_Sum_max
# 		Flow["Bwd_IAT_Sum_min"] = Bwd_IAT_Sum_min
# 		Flow["Bwd_IAT_Sum_std"] = Bwd_IAT_Sum_std
# 		Flow["Bwd_IAT_Sum_mean"] = Bwd_IAT_Sum_mean
# 		# Bwd_IAT_Mean
# 		Flow["Bwd_IAT_Mean_sum"] = Bwd_IAT_Mean_sum
# 		Flow["Bwd_IAT_Mean_max"] = Bwd_IAT_Mean_max
# 		Flow["Bwd_IAT_Mean_min"] = Bwd_IAT_Mean_min
# 		Flow["Bwd_IAT_Mean_std"] = Bwd_IAT_Mean_std
# 		Flow["Bwd_IAT_Mean_mean"] = Bwd_IAT_Mean_mean
# 		# Bwd_IAT_Std_sum
# 		Flow["Bwd_IAT_Std_sum"] = Bwd_IAT_Std_sum
# 		Flow["Bwd_IAT_Std_max"] = Bwd_IAT_Std_max
# 		Flow["Bwd_IAT_Std_min"] = Bwd_IAT_Std_min
# 		Flow["Bwd_IAT_Std_std"] = Bwd_IAT_Std_std
# 		Flow["Bwd_IAT_Std_mean"] = Bwd_IAT_Std_mean
# 		# Bwd_IAT_Max
# 		Flow["Bwd_IAT_Max_sum"] = Bwd_IAT_Max_sum
# 		Flow["Bwd_IAT_Max_max"] = Bwd_IAT_Max_max
# 		Flow["Bwd_IAT_Max_min"] = Bwd_IAT_Max_min
# 		Flow["Bwd_IAT_Max_std"] = Bwd_IAT_Max_std
# 		Flow["Bwd_IAT_Max_mean"] = Bwd_IAT_Max_mean
# 		# Bwd_IAT_Min
# 		Flow["Bwd_IAT_Min_sum"] = Bwd_IAT_Min_sum
# 		Flow["Bwd_IAT_Min_max"] = Bwd_IAT_Min_max
# 		Flow["Bwd_IAT_Min_min"] = Bwd_IAT_Min_min
# 		Flow["Bwd_IAT_Min_std"] = Bwd_IAT_Min_std
# 		Flow["Bwd_IAT_Min_mean"] = Bwd_IAT_Min_mean
# 		# Bwd_Length_First
# 		Flow["Bwd_Length_First_sum"] = Bwd_Length_First_sum
# 		Flow["Bwd_Length_First_max"] = Bwd_Length_First_max
# 		Flow["Bwd_Length_First_min"] = Bwd_Length_First_min
# 		Flow["Bwd_Length_First_std"] = Bwd_Length_First_std
# 		Flow["Bwd_Length_First_mean"] = Bwd_Length_First_mean
# 		# Bwd_Length_Sum
# 		Flow["Bwd_Length_Sum_sum"] = Bwd_Length_Sum_sum
# 		Flow["Bwd_Length_Sum_max"] = Bwd_Length_Sum_max
# 		Flow["Bwd_Length_Sum_min"] = Bwd_Length_Sum_min
# 		Flow["Bwd_Length_Sum_std"] = Bwd_Length_Sum_std
# 		Flow["Bwd_Length_Sum_mean"] = Bwd_Length_Sum_mean
# 	return Flow
# # End of CountFeatureLong()