##################################################################
# Usage: python3 read_packet.py filename.pcap output.csv
# The file reads packets from a pcap file and print TCP, UDP and ICMP
# packet information
# ICMP: datetime, srcIP, destIP, protocol, length, type, code
# TCP/UDP: datetime, srcIP, destIP, protocol, length, srcPort, destPort
# 2021.03.03 add ICMP packet
###################################################################

import sys
import dpkt
import datetime
import socket
from dpkt.compat import compat_ord
import csv

def inet_to_str(inet):
    """Convert inet object to a string

        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)
# end of ine_to_str(inet)

def process_packets(pcap):
	begin_timestamp = 0
	savefile = open(sys.argv[2], 'w', newline='')
	csvwriter = csv.writer(savefile)
	# csvwriter.writerow(["datetime", "srcIP", "destIP", "srcPort", "destPort", "protocol", "length"])
	for timestamp, buf in pcap:
		if begin_timestamp == 0:
			begin_timestamp = timestamp
    	
        # Unpack the Ethernet frame (mac src/dst, ethertype)
		try:
			eth = dpkt.ethernet.Ethernet(buf)

    	    # Make sure the Ethernet data contains an IP packet
			if not isinstance(eth.data, dpkt.ip.IP):
				#	print('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
				continue
			
			# the IP packet
			try:
				ip = eth.data
				if (ip.p == 1):		# ICMP
					icmp = ip.data
					# print('%s, %s, %s, %s, %s' % (str(timestamp-begin_timestamp), inet_to_str(ip.src), inet_to_str(ip.dst), ip.p, ip.len))
					csvwriter.writerow([str(timestamp-begin_timestamp), inet_to_str(ip.src), inet_to_str(ip.dst), ip.p, ip.len, icmp.type, icmp.code])
				elif (ip.p == 6):		# TCP
					tcp = ip.data
					# print('%s, %s, %s, %s, %s, %s, %s' % (str(timestamp-begin_timestamp), inet_to_str(ip.src), inet_to_str(ip.dst), ip.p, ip.len, tcp.sport, tcp.dport))
					csvwriter.writerow([str(timestamp-begin_timestamp), inet_to_str(ip.src), inet_to_str(ip.dst), ip.p, ip.len, tcp.sport, tcp.dport, tcp.flags])
				elif (ip.p == 17):	# UDP
					udp = ip.data
					# print('%s, %s, %s, %s, %s, %s, %s' % (str(timestamp-begin_timestamp), inet_to_str(ip.src), inet_to_str(ip.dst), ip.p, ip.len, udp.sport, udp.dport))
					csvwriter.writerow([str(timestamp-begin_timestamp), inet_to_str(ip.src), inet_to_str(ip.dst), ip.p, ip.len, udp.sport, udp.dport])
			except AttributeError:
				continue
		except:
			
			continue
	savefile.close()
# end of process_packets(pcap)

def main():
	if len(sys.argv) < 3:
		print("Usage: python3 read_pcap.py filename.pcap outfile.csv")
		sys.exit(1)
		
	# open pcap file
	f = open(sys.argv[1], 'rb')
	pcap = dpkt.pcap.Reader(f)
	process_packets(pcap)
# end of main()

if __name__ == '__main__':
	main()
