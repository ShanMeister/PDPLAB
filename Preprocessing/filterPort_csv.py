##################################################################
# This program filter the packets of some specific Port number.
# The file reads packets from a csv file and print packets match Port
# Usage: python3 filterPort_csv.py input.csv output.csv Port_nums Port1 Port2 Port3 ...
# ICMP: datetime, srcIP, destIP, protocol, length
# TCP/UDP: datetime, srcIP, destIP, srcPort, destPort, protocol, length
###################################################################

import sys



def main():
	if len(sys.argv) < 3:
		print("Usage: python3 filterPort_csv.py input.csv output.csv Port_nums Port1 Port2 Port3 ...")
		sys.exit(1)
		
	Target_Port = set()

	port_nums = int(sys.argv[3].replace(" ", ""))
	for i in range(port_nums):
		t_port = sys.argv[4+i].replace(" ", "")
		Target_Port.add(t_port)
		print("Target Port: %s" % (t_port))	
	
	# open csv file
	f_in = open(sys.argv[1], 'r')
	f_out = open(sys.argv[2], 'w')
	line = f_in.readline()
	# f_out.write(line)
	while(line):	#run until EOF
		token = line.split(",")
		sport = token[5].replace(" ", "")
		dport = token[6].replace(" ", "").replace("\n", "")
		for t_port in Target_Port:
			if((t_port == sport) or (t_port == dport)):
				f_out.write(line)	
				break			
		line = f_in.readline()
			
	f_in.close()
	f_out.close()
	
# end of main()

if __name__ == '__main__':
	main()
