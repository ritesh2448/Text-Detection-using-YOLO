import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os

grid_size = 16
img_size = 512
per_grid_pix = img_size//grid_size

def preprocess(init_path):
	X = []
	Y = []

	n = len(os.listdir(init_path+'images'))+1
	for i in range(1,n):

		img = cv2.imread(init_path+'images/img_'+str(i)+'.jpg')
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		#Removing noise
		#img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
		
		x_s = img_size/img.shape[1]
		y_s = img_size/img.shape[0]

		resized_img = cv2.resize(img,(img_size,img_size))

		file = init_path+"ground_truth/gt_img_"+str(i)+".txt"

		labels = np.zeros((grid_size,grid_size,5))
		
		
		with open(file,'r') as f:
			for row in f:
				if row.count(',') > 1:
					line = row.split(',')
				else:
					line = row.split()
				x1 = int(int(line[0].strip())*x_s)
				y1 = int(int(line[1].strip())*y_s)
				x2 = int(int(line[2].strip())*x_s)
				y2 = int(int(line[3].strip())*y_s)
				center_x = (x1+x2)/2 
				center_y = (y1+y2)/2
				w = abs(x2-x1)
				h = abs(y2-y1)
				labels[int(center_y)//per_grid_pix][int(center_x)//per_grid_pix][0] = 1
				labels[int(center_y)//per_grid_pix][int(center_x)//per_grid_pix][1] = (center_x%per_grid_pix)/per_grid_pix
				labels[int(center_y)//per_grid_pix][int(center_x)//per_grid_pix][2] = (center_y%per_grid_pix)/per_grid_pix
				labels[int(center_y)//per_grid_pix][int(center_x)//per_grid_pix][3] = w/img_size
				labels[int(center_y)//per_grid_pix][int(center_x)//per_grid_pix][4] = h/img_size
				#cv2.rectangle(resized_img,(int(x*x1),int(y*y1)),(int(x*x2),int(y*y2)),(255,0,0))
				#cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0))


		X.append(resized_img)
		Y.append(labels)
		print(i)

	X = np.asarray(X)
	#print(X.shape)
	Y = np.asarray(Y)
	#print(Y.shape)
	return X,Y

def show_random(X,Y):

	idx = random.randint(0,X.shape[0]-1)
	img = X[idx]
	for i in range(Y.shape[1]):
		for j in range(Y.shape[2]):
			if Y[idx][i][j][0]>=0.5:
				x1 = Y[idx][i][j][1]*per_grid_pix + j*per_grid_pix - Y[idx][i][j][3]*img_size/2
				x2 = Y[idx][i][j][1]*per_grid_pix + j*per_grid_pix + Y[idx][i][j][3]*img_size/2
				y1 = Y[idx][i][j][2]*per_grid_pix + i*per_grid_pix - Y[idx][i][j][4]*img_size/2
				y2 = Y[idx][i][j][2]*per_grid_pix + i*per_grid_pix + Y[idx][i][j][4]*img_size/2
				cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255))
	plt.imshow(img)
	plt.show()

def show(init_path):
	X,Y = preprocess(init_path)
	for i in range(5):
		show_random(X,Y)

show("")