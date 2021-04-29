import numpy as np
import argparse
import imutils
import time
import cv2
import os
import matplotlib.pyplot as plt

SET_WIDTH = int(600)

# normalize_image = 1 / 255.0

# resize_image_shape = (1024, 512)

# sample_img = cv2.imread('data\images\image1.jpg')

# sample_img = imutils.resize(sample_img, width=SET_WIDTH)

# blob_img = cv2.dnn.blobFromImage(sample_img, normalize_image, resize_image_shape, 0, swapRB = True, crop=False)

# Enet 모델 가져오기.
cv_enet_model = cv2.dnn.readNet('enet-cityscapes/enet-model.net')
cv_enet_model.setInput(blob_img)
cv_enet_model_output = cv_enet_model.forward()

# 레이블 이름을 로딩
open('enet-cityscapes/enet-classes.txt').read()
label_values = open('enet-cityscapes/enet-classes.txt').read().split('\n')
label_values = label_values[ : -2+1]

IMG_OUTPUT_SHAPE_START = 1 
IMG_OUTPUT_SHAPE_END = 4
classes_num, h, w = cv_enet_model_output.shape[IMG_OUTPUT_SHAPE_START : IMG_OUTPUT_SHAPE_END]

class_map = np.argmax(cv_enet_model_output[0], axis = 0)

CV_ENET_SHAPE_IMG_COLORS = open('enet-cityscapes/enet-colors.txt').read().split('\n')
CV_ENET_SHAPE_IMG_COLORS = CV_ENET_SHAPE_IMG_COLORS[ : -2+1]

temp = []
np.array([np.array(color.split(',')).astype('int')  for color in CV_ENET_SHAPE_IMG_COLORS  ])

for color in CV_ENET_SHAPE_IMG_COLORS :
  color_list = color.split(',')      
  color_num_list = np.array(color_list).astype('int')  
  print(color_num_list)
  temp.append(color_num_list)

CV_ENET_SHAPE_IMG_COLORS = np.array(temp)

# 각 픽셀별로, 클래스에 해당하는 숫자가 적힌 class_map을
# 각 숫자에 매핑되는 색깔로 셋팅해 준것이다.
# 따라서 각 픽셀별 색깔 정보가 들어가게 되었다.
# 2차원 행렬을, 3차원 채널이 있는 RGB 행렬로 만든다.
mask_class_map = CV_ENET_SHAPE_IMG_COLORS[class_map]

# 리사이즈 한다.

# 인터폴레이션을 INTER_NEAREST 로 한 이유는?? 
# 레이블 정보(0~19) 와 컬러정보 (23,100,243) 는 둘다 int 이므로, 
# 가장 가까운 픽셀 정보와 동일하게 셋팅해주기 위해서.

mask_class_map = cv2.resize(mask_class_map, (sample_img.shape[1], sample_img.shape[0]) , 
           interpolation = cv2.INTER_NEAREST )

class_map = cv2.resize(class_map, (sample_img.shape[1], sample_img.shape[0]) , 
                       interpolation=cv2.INTER_NEAREST)

cv_enet_model_output = ( ( 0.4 * sample_img ) + (0.6 * mask_class_map) ).astype('uint8')

### 비디오 처리

DEFAULT_FRAME = 1

sv = cv2.VideoCapture('data\\videos\dashcam2.mp4')

sample_video_writer = None

prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
total = sv.get(prop)

try :
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = sv.get(prop)
    print("[INFO] {} total frames in video.".format(total))
except :
    print("[INFO] could not determine number of frames in video")
    total = -1

while True :
    grabbed, frame = sv.read()

    if grabbed == False :
        break

    normalize_image = 1 / 255.0
    resize_image_shape = (1024, 512)
    video_frame = imutils.resize(frame, width=SET_WIDTH)
    blob_img = cv2.dnn.blobFromImage(frame, normalize_image, resize_image_shape, 0, 
                                    swapRB = True, crop = False)
    cv_enet_model.setInput(blob_img)
    # 모델이, 세그멘테이션 추론(예측)하는데 얼마나 걸렸는지 측정.
    start_time = time.time()
    cv_enet_model_output = cv_enet_model.forward()
    end_time = time.time()

    (classes_num, height, width) = cv_enet_model_output.shape[1:4]

    class_map = np.argmax(cv_enet_model_output[0], axis=0)

    mask_class_map = CV_ENET_SHAPE_IMG_COLORS[class_map]

    mask_class_map = cv2.resize(mask_class_map, (video_frame.shape[1], video_frame.shape[0]) ,
                interpolation = cv2.INTER_NEAREST)
    
    cv_enet_model_output = ( (0.3 * video_frame) + (0.7 * mask_class_map) ).astype('uint8')

    cv2.show('result',cv_enet_model_output)
    if cv2.waitKey(25) & 0xFF == 27 :
                    break
    else : 
        break

cap.release()
cv2.destroyAllWindows()