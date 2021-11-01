#--------------------------------------------------------------------------------------------
# This program reads packets in csv files and calculate flow features
# Usage: python3 get_features.py filename.csv segment_len
#      where filename.csv is the packets in csv file
#                 segment_len = 0, full flow; segment_len = t, only first t sec
#  Date: 2021.03.03
#  Features: 36
#    Duration, Total_Packets,  IAT_Sum, IAT_Mean, IAT_Std, IAT_Max, IAT_Min, Length_First, Length_Min, Length_Max, Length_Sum, Length_Mean, Length_Std, Num_Dir_Change, Freq_Dir_Change
#    Fwd_Total_Packets, Fwd_Length_First, Fwd_Length_Max, Fwd_Length_Min, Fwd_Length_Sum, Fwd_Length_Mean, Fwd_Length_Std, Fwd_IAT_Sum, Fwd_IAT_Mean, Fwd_IAT_Std, Fwd_IAT_Max, Fwd_IAT_Min, 
#    Bwd_Total_Packets, Bwd_Length_First, Bwd_Length_Max, Bwd_Length_Min, Bwd_Length_Sum, Bwd_Length_Mean, Bwd_Length_Std, Bwd_IAT_Sum, Bwd_IAT_Mean, Bwd_IAT_Std, Bwd_IAT_Max, Bwd_IAT_Min,
#--------------------------------------------------------------------------------------------
# Thic program consider ICMP packets
import sys
import datetime
import math

HashTableSize = 3119
class Packet:
	def __init__(self, time, sip, dip, proto, length, sport, dport):
		try:
			self.time = float(time)
			self.sip = sip.replace(" ", "")		# skip " "
			self.dip = dip.replace(" ", "")
			self.sport = int(sport)
			self.dport = int(dport)
			self.proto = int(proto)
			self.length = int(length)
			return
		except ValueError:
			self.length = -1
			return
# end of Packet

class Flow:
	def __init__(self, time, sip, dip, sport, dport, proto, length):
		self.time = float(time)
		self.sip = sip
		self.dip = dip
		self.sport = sport
		self.dport = dport
		self.proto = proto
		self.starttime = float(time)
		self.endtime = float(time)
		self.Duration = self.endtime - self.starttime		
		self.Total_Packets = 1
		
		# temp variable for standard devision
		self.Length_sqr = float(length * length)
		self.IAT_sqr = float(0)
		self.Fwd_Length_sqr = float(length * length)
		self.Fwd_IAT_sqr = float(0)
		self.Bwd_Length_sqr = float(0)
		self.Bwd_IAT_sqr = float(0)
		
		# length features
		self.Length_First = length
		self.Length_Sum = length
		self.Length_Min = length
		self.Length_Max = length
		self.Length_Mean = float(length)
		self.Length_Std = float(0)
		
		# interval features
		self.last_packet_time = float(time)
		self.IAT_Sum = float(0)
		self.IAT_Mean = float(0)
		self.IAT_Std = float(0)
		self.IAT_Max = float(0)
		self.IAT_Min = float(0)
		
		# direction feature
		self.last_dir = 1
		self.Num_Dir_Change = 0
		self.Freq_Dir_Change = float(0)
		
		# Fwd features
		self.fwd_last_packet_time = float(time)
		self.Fwd_Total_Packets = 1
		self.Fwd_Length_First = length
		self.Fwd_Length_Max = length
		self.Fwd_Length_Min = length
		self.Fwd_Length_Sum = length
		self.Fwd_Length_Mean = float(length)
		self.Fwd_Length_Std = float(0)
		self.Fwd_IAT_Sum = float(0)
		self.Fwd_IAT_Mean = float(0)
		self.Fwd_IAT_Std = float(0)
		self.Fwd_IAT_Max = float(0)
		self.Fwd_IAT_Min = float(0) 
		
		# Bwd features
		self.bwd_last_packet_time = float(-1)
		self.Bwd_Total_Packets = 0
		self.Bwd_Length_First = 0
		self.Bwd_Length_Max = 0
		self.Bwd_Length_Min = 0
		self.Bwd_Length_Sum = 0
		self.Bwd_Length_Mean = float(0)
		self.Bwd_Length_Std = float(0)
		self.Bwd_IAT_Sum = float(0)
		self.Bwd_IAT_Mean = float(0)
		self.Bwd_IAT_Std = float(0)
		self.Bwd_IAT_Max = float(0)
		self.Bwd_IAT_Min = float(0)
		return

	def is_same_flow(self, sip, dip, sport, dport, proto):
		if((self.sip == sip) and (self.dip == dip) and (self.sport == sport) and \
			(self.dport == dport) and (self.proto == proto)):
			result = True
			direction = 1		# 1: sip -> dip, -1: dip -> sip
		elif((self.sip == dip) and (self.dip == sip) and (self.sport == dport) and \
			(self.dport == sport) and (self.proto == proto)):
			result = True
			direction = -1
		else:
			result = False
			direction = 0
		return result, direction

	def add_packet(self, time, sip, dip, sport, dport, proto, length, direction):
		# length features
		global segment_len
		if segment_len == 0:
			pass
		elif(time-self.starttime) > segment_len:
			return

		self.endtime = float(time)
		self.Duration = self.endtime - self.starttime		
		self.Total_Packets += 1

		# length features
		if(length > self.Length_Max):
			self.Length_Max = length
		if(length < self.Length_Min):
			self.Length_Min = length
		self.Length_Sum += length
		self.Length_Mean = float(self.Length_Sum / self.Total_Packets)
		self.Length_sqr += float(length * length)
		self.Length_Std = math.sqrt(self.Length_sqr/self.Total_Packets -  self.Length_Mean*self.Length_Mean)
				
		# interval features
		interval = float(time - self.last_packet_time)
		self.last_packet_time = float(time)
		if(self.IAT_Max == 0):	# the second packet
			self.IAT_Max = interval
			self.IAT_Min = interval
		else:
			if(interval > self.IAT_Max):			self.IAT_Max = interval
			if(interval < self.IAT_Min):			self.IAT_Min = interval
		self.IAT_Sum += interval
		self.IAT_Mean = self.IAT_Sum / (self.Total_Packets - 1)
		self.IAT_sqr += float(interval * interval)
		try:
			self.IAT_Std = math.sqrt(self.IAT_sqr/(self.Total_Packets-1) -  self.IAT_Mean*self.IAT_Mean)
		except:
			self.IAT_Std = math.sqrt(-1*(self.IAT_sqr/(self.Total_Packets-1) -  self.IAT_Mean*self.IAT_Mean))

		# direction feature
		if(self.last_dir != direction):
			self.Num_Dir_Change += 1
			self.last_dir = direction
		self.Freq_Dir_Change = float(self.Num_Dir_Change / (self.Total_Packets - 1))

		if(direction == 1):
			# Fwd features
			self.Fwd_Total_Packets += 1
			if(length > self.Fwd_Length_Max):		self.Fwd_Length_Max = length
			if(length < self.Fwd_Length_Min):		self.Fwd_Length_Min = length
			self.Fwd_Length_Sum += length
			self.Fwd_Length_Mean = float(self.Fwd_Length_Sum / self.Fwd_Total_Packets)
			self.Fwd_Length_sqr += float(length * length)
			self.Fwd_Length_Std = math.sqrt(self.Fwd_Length_sqr/self.Fwd_Total_Packets -  self.Fwd_Length_Mean*self.Fwd_Length_Mean)
			fwd_interval = float(time - self.fwd_last_packet_time)
			self.fwd_last_packet_time = float(time)
			if(self.Fwd_IAT_Max == 0): 		# the second fwd packet
				self.Fwd_IAT_Max = fwd_interval
				self.Fwd_IAT_Min = fwd_interval
			else:
				if(fwd_interval > self.Fwd_IAT_Max):		self.Fwd_IAT_Max = fwd_interval
				if(fwd_interval < self.Fwd_IAT_Min):		self.Fwd_IAT_Min = fwd_interval
			self.Fwd_IAT_Sum += fwd_interval
			self.Fwd_IAT_Mean = self.Fwd_IAT_Sum / (self.Fwd_Total_Packets - 1)
			self.Fwd_IAT_sqr += float(fwd_interval * fwd_interval)
			try:
				self.Fwd_IAT_Std = math.sqrt(self.Fwd_IAT_sqr/(self.Fwd_Total_Packets-1) -  self.Fwd_IAT_Mean*self.Fwd_IAT_Mean)
			except:
				self.Fwd_IAT_Std = math.sqrt(-1*(self.Fwd_IAT_sqr/(self.Fwd_Total_Packets-1) -  self.Fwd_IAT_Mean*self.Fwd_IAT_Mean))
		else:			# direction = 0
			# Bwd features
			if(self.Bwd_Total_Packets == 0):	# first bwd packet
				self.Bwd_Total_Packets = 1
				self.Bwd_Length_First = length
				self.Bwd_Length_Max = length
				self.Bwd_Length_Min = length
				self.Bwd_Length_Sum = length
				self.Bwd_Length_Mean = float(length)
				self.Bwd_Length_sqr = float(length * length)
				self.Bwd_Length_Std = float(0)
				# Bwd_IAT_Sum, Bwd_IAT_Mean, Bwd_IAT_Std, Bwd_IAT_Max, Bwd_IAT_Min are all 0 when Flow is created
				self.bwd_last_packet_time = float(time)
			else:
				self.Bwd_Total_Packets += 1
				if(length > self.Bwd_Length_Max):		self.Bwd_Length_Max = length
				if(length < self.Bwd_Length_Min):		self.Bwd_Length_Min = length
				self.Bwd_Length_Sum += length
				self.Bwd_Length_Mean = float(self.Bwd_Length_Sum / self.Bwd_Total_Packets)
				self.Bwd_Length_sqr += float(length * length)
				self.Bwd_Length_Std = math.sqrt(self.Bwd_Length_sqr/self.Bwd_Total_Packets -  self.Bwd_Length_Mean*self.Bwd_Length_Mean)
				bwd_interval = float(time - self.bwd_last_packet_time)
				self.bwd_last_packet_time = float(time)
				if(self.Bwd_IAT_Max == 0): 		# the second fwd packet
					self.Bwd_IAT_Max = bwd_interval
					self.Bwd_IAT_Min = bwd_interval
				else:
					if(bwd_interval > self.Bwd_IAT_Max):		self.Bwd_IAT_Max = bwd_interval
					if(bwd_interval < self.Bwd_IAT_Min):		self.Bwd_IAT_Min = bwd_interval
				self.Bwd_IAT_Sum += bwd_interval
				self.Bwd_IAT_Mean = self.Bwd_IAT_Sum / (self.Bwd_Total_Packets - 1)
				self.Bwd_IAT_sqr += float(bwd_interval * bwd_interval)
				try:
					self.Bwd_IAT_Std = math.sqrt(self.Bwd_IAT_sqr/(self.Bwd_Total_Packets-1) -  self.Bwd_IAT_Mean*self.Bwd_IAT_Mean)
				except:
					self.Bwd_IAT_Std = math.sqrt(-1*(self.Bwd_IAT_sqr/(self.Bwd_Total_Packets-1) -  self.Bwd_IAT_Mean*self.Bwd_IAT_Mean))
		return

	def flow_feature(self): 
		features = self.sip + "," + self.dip + "," + str(self.sport) + "," + str(self.dport) + "," + str(self.proto) + "," \
				+ str(self.Duration) + "," +  str(self.Total_Packets) + "," + str(self.IAT_Sum) + "," + str(self.IAT_Mean) + "," + str(self.IAT_Std) + "," + str(self.IAT_Max) + "," + str(self.IAT_Min) + "," + str(self.Length_First) + ","\
				+ str(self.Length_Min) + "," + str(self.Length_Max) + "," + str(self.Length_Sum) + "," + str(self.Length_Mean) + "," + str(self.Length_Std) + "," + str(self.Num_Dir_Change) + "," + str(self.Freq_Dir_Change) + ","\
				+ str(self.Fwd_Total_Packets) + "," + str(self.Fwd_Length_First) + "," + str(self.Fwd_Length_Max) + "," + str(self.Fwd_Length_Min) + "," + str(self.Fwd_Length_Sum) + "," + str(self.Fwd_Length_Mean) + ","\
				+ str(self.Fwd_Length_Std) + "," + str(self.Fwd_IAT_Sum) + "," + str(self.Fwd_IAT_Mean) + "," + str(self.Fwd_IAT_Std) + "," + str(self.Fwd_IAT_Max) + "," + str(self.Fwd_IAT_Min) + ","\
				+ str(self.Bwd_Total_Packets) + "," + str(self.Bwd_Length_First) + "," + str(self.Bwd_Length_Max) + "," + str(self.Bwd_Length_Min) + "," + str(self.Bwd_Length_Sum) + "," + str(self.Bwd_Length_Mean) + ","\
				+ str(self.Bwd_Length_Std) + "," + str(self.Bwd_IAT_Sum) + "," + str(self.Bwd_IAT_Mean) + "," +  str(self.Bwd_IAT_Std) + "," + str(self.Bwd_IAT_Max) + ","  + str(self.Bwd_IAT_Min) + ","\
				+ str('\n')
		return features

#end of Flow

FlowList = [None] * HashTableSize			# Each element is also a set
segment_len = 0
def Hash_key(sip, dip, sport, dport, proto):
	if(sip < dip):
		val = dip + sip + str(dport) + str(sport) + str(proto)
		key = hash(val) % len(FlowList)
	else:
		val = sip + dip + str(sport) + str(dport) + str(proto)
		key = hash(val) % len(FlowList)
	return key
# end of Hashing_key(sip, dip, sport, dport, proto)

def main():
	global FlowSet
	global segment_len
	
	if len(sys.argv) < 2:
		print("Usage: python3 get_features.py filename.csv segment_len")
		sys.exit(1)
	
	filename = sys.argv[1].replace(".csv", "")
	segment_len = int(sys.argv[2])
	InputFileName = filename + ".csv"
	OutputFileName = filename + "_" +str(segment_len) + "s_features.csv" 

	# Initialize Hash Table
	for i in range(len(FlowList)):
		FlowList[i] = set()
	
	# open pcap file
	f_in = open(InputFileName, 'r')
	print('Process packets. .......')

	i = 0
	n_flows = 0
	for line in f_in:
		token = line.split(",")
		
		# Skip ICMP packets
		#if int(token[3]) == 1:	
		#	continue
		#	
		if int(token[3]) == 1:		#ICMP
			p = Packet(token[0], token[1], token[2], token[3], token[4], str(0), str(0))
		else:
			p = Packet(token[0], token[1], token[2], token[3], token[4], token[5], token[6])
		
		# Consider ICMP packets
		# for ICMP packet: TYPE-> sport, CODE->dport
		p = Packet(token[0], token[1], token[2], token[3], token[4], token[5], token[6])
		
		if(p.length < 0):	# format error
			continue
		
		found = False
		key = Hash_key(p.sip, p.dip, p.sport, p.dport, p.proto)
		for f in FlowList[key]:
			result, direction = f.is_same_flow(p.sip, p.dip, p.sport, p.dport, p.proto)
			if(result):
				found = True
				break		
		if(found):		#same flow
			f.add_packet(p.time, p.sip, p.dip, p.sport, p.dport, p.proto, p.length, direction)
		else:				# new flow
			new_f = Flow(p.time, p.sip, p.dip, p.sport, p.dport, p.proto, p.length)
			FlowList[key].add(new_f)
			n_flows += 1
				
		i += 1
		# if(i % 10000) == 0:
			# print(i, n_flows)
		
	f_in.close()	
	
	# Write flow information
	print('Write flow information')
	
	fp = open(OutputFileName, 'w')
	label = "sip,dip,sport,dport,proto," \
			+ "Duration, Total_Packets,  IAT_Sum, IAT_Mean, IAT_Std, IAT_Max, IAT_Min, Length_First, Length_Min, Length_Max, Length_Sum, Length_Mean, Length_Std, Num_Dir_Change, Freq_Dir_Change,"\
			+ "Fwd_Total_Packets, Fwd_Length_First, Fwd_Length_Max, Fwd_Length_Min, Fwd_Length_Sum, Fwd_Length_Mean, Fwd_Length_Std, Fwd_IAT_Sum, Fwd_IAT_Mean, Fwd_IAT_Std, Fwd_IAT_Max, Fwd_IAT_Min,"\
			+ "Bwd_Total_Packets, Bwd_Length_First, Bwd_Length_Max, Bwd_Length_Min, Bwd_Length_Sum, Bwd_Length_Mean, Bwd_Length_Std, Bwd_IAT_Sum, Bwd_IAT_Mean, Bwd_IAT_Std, Bwd_IAT_Max, Bwd_IAT_Min,"\
			+ "Label\n"
	fp.write(label)
	
	for key in range(len(FlowList)):
		# print(key, len(FlowList[key]))
		for f in FlowList[key]:
			fp.write(f.flow_feature())
	fp.close()
# end of main()

if __name__ == '__main__':
	main()
