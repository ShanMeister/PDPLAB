##################################################################
# This program filter the packets of some specific IP host.
# The file reads packets from a csv file and print packets match IP
# Usage: python3 filterIP_csv.py input.csv output.csv IP_nums IP1 IP2 IP3 ...
# ICMP: datetime, srcIP, destIP, protocol, length
# TCP/UDP: datetime, srcIP, destIP, srcPort, destPort, protocol, length
###################################################################

import sys



def main():
	if len(sys.argv) < 3:
		print("Usage: python3 filterIP_csv.py input.csv output.csv IP_nums IP1 IP2 IP3 ...")
		sys.exit(1)
		
	Target_ip = set()

	ip_nums = int(sys.argv[3].replace(" ", ""))
	for i in range(ip_nums):
		t_ip = sys.argv[4+i].replace(" ", "")
		Target_ip.add(t_ip)
		print("Target ip: %s" % (t_ip))	
	
	# open csv file
	f_in = open(sys.argv[1], 'r')
	f_out = open(sys.argv[2], 'w')
	line = f_in.readline()
	# f_out.write(line)
	while(line):	#run until EOF
		token = line.split(",")
		sip = token[1].replace(" ", "")
		dip = token[2].replace(" ", "")
		for t_ip in Target_ip:
			if((t_ip == sip) or (t_ip == dip)):
				f_out.write(line)	
				break			
		line = f_in.readline()
			
	f_in.close()
	f_out.close()
	
# end of main()

if __name__ == '__main__':
	main()
