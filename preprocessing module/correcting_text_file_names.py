import os

def main():

	path = "ground_truth\\gt_"
	extension = ".txt"

	for i in range(100,140):

		src_file_name = path + str(i) + extension
		dst_file_name = path + "img_" + str(i+134) + extension 
		os.rename(src_file_name,dst_file_name)

	# Filenumber 140 is missing.

	for i in range(141,329):
		
		src_file_name = path + str(i) + extension
		dst_file_name = path + "img_" + str(i+133) + extension 
		os.rename(src_file_name,dst_file_name)

if __name__ == '__main__':

	main()