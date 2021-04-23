from flask import url_for
from flaskpackage import app


import cv2
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow.keras.backend as K


import pytesseract
#from PIL import Image

pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
#pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract'



def preprocess():
    X = []
    #img = cv2.imread(file_path)
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Removing noise
    # img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

    # x_s = img_size/img.shape[1]
    # y_s = img_size/img.shape[0]

    resized_img = cv2.resize(img, (512, 512))
    X.append(resized_img)
    X = np.asarray(X)
    return X


def text_extract(image,x1,y1,x2,y2):
    #image =cv2.imread(img)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    thresh=255-cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    ROI =thresh[y1:y2,x1:x2]


    custom_config = ("-l eng --oem 1 --psm 6")
    data = pytesseract.image_to_string(ROI, lang='eng',config=custom_config)
    return data


def gen_bounding_boxes(model):

    X=preprocess()
    Y=model.predict(X)
    grid_size = 16
    img_size = 512
    per_grid_pix = img_size // grid_size
    idx = 0
    img = X[idx].copy()
    bounding_boxes = []
    scores = []
    data = []

    for i in range(Y.shape[1]):
        for j in range(Y.shape[2]):
            if Y[idx][i][j][0] >= 0.4:
                x1 = Y[idx][i][j][1] * per_grid_pix + j * per_grid_pix - (Y[idx][i][j][3] * img_size) / 2
                x2 = Y[idx][i][j][1] * per_grid_pix + j * per_grid_pix + (Y[idx][i][j][3] * img_size) / 2
                y1 = Y[idx][i][j][2] * per_grid_pix + i * per_grid_pix - (Y[idx][i][j][4] * img_size) / 2
                y2 = Y[idx][i][j][2] * per_grid_pix + i * per_grid_pix + (Y[idx][i][j][4] * img_size) / 2
                bounding_boxes.append([int(y1), int(x1), int(y2), int(x2)])
                scores.append(Y[idx][i][j][0])

    t_bounding_boxes = []
    if len(bounding_boxes)!=0:
    	t_bounding_boxes = nn(bounding_boxes,scores)
    	t_bounding_boxes = sorted(t_bounding_boxes , key=lambda k: [k[0]+k[2], k[1]+k[3]])

    for idx,i in enumerate(t_bounding_boxes):

        if i[1] == i[3] or i[0] == i[2]:
            continue

        data.append("{} - {}".format(idx,text_extract(img, i[1], i[0], i[3], i[2])))

    for idx,i in enumerate(t_bounding_boxes):
    	cv2.rectangle(img,(int(i[1]),int(i[0])),(int(i[3]),int(i[2])),(0,0,255),2)
    	cv2.putText(img, str(idx), (i[1], i[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'answer.jpg'), img)
    return data



'''
def iou(b1,b2):
  inter_x1 = max(b1[1],b2[1])
  inter_y1 = max(b1[0],b2[0])
  inter_x2 = min(b1[3],b2[3])
  inter_y2 = min(b1[2],b2[2])

  inter_width = (inter_x2 - inter_x1) if (inter_x2 - inter_x1)>0 else 0
  inter_height = (inter_y2 - inter_y1) if (inter_y2 - inter_y1)>0 else 0
  inter_area = inter_width*inter_height

  b1_area = abs((b1[0] - b1[2])*(b1[1] - b1[3]))
  b2_area = abs((b2[0] - b2[2])*(b2[1] - b2[3]))
  return inter_area/(min(b1_area,b2_area))
'''


def nn(bounding_boxes, scores):
    # scores = []
    # for b in bounding_boxes:
    #  scores.append(abs((b[0] - b[2])*(b[1] - b[3])))
    indices = tf.image.non_max_suppression(bounding_boxes, scores, 300, iou_threshold=0.2)
    true_bounding_boxes = []
    for i in indices:
        true_bounding_boxes.append(bounding_boxes[i])
    return true_bounding_boxes

