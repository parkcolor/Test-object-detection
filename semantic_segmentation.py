# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import matplotlib.pyplot as plt

DEFAULT_FRAME = 1
SET_WIDTH = 600

sv = cv2.VideoCapture('data/videos/dashcam2.mp4')

# 프레임의 정보 가져오기 : 화면 크기 ( width, height )
frame_width = int(sv.get(3))
frame_height = int(sv.get(4))

# 사이즈를 반으로 줄이는 방법
# if int(frame_width / 2) % 2 == 0:
#     frame_width = int(frame_width / 2)
# else :
#     frame_width = int(frame_width / 2) + 1
# if int(frame_height / 2) % 2 == 0:
#     frame_height = int(frame_height / 2)
# else :
#     frame_height = int(frame_height / 2) + 1


out = cv2.VideoWriter('data/videos/video_seg.mp4', 
                cv2.VideoWriter_fourcc(*'H264'), 
                10,
                ( frame_width , frame_height ) )  


# Enet 모델 가져오기.
cv_enet_model = cv2.dnn.readNet('data/enet-cityscapes/enet-model.net')

# 색 정보도 가져온다.
CV_ENET_SHAPE_IMG_COLORS = open('data/enet-cityscapes/enet-colors.txt').read().split('\n')
# 맨 마지막 따옴표를 없애기
CV_ENET_SHAPE_IMG_COLORS = CV_ENET_SHAPE_IMG_COLORS[ : -2+1]

CV_ENET_SHAPE_IMG_COLORS = np.array([np.array(color.split(',')).astype('int')  for color in CV_ENET_SHAPE_IMG_COLORS  ])



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

  print(end_time - start_time)

  (classes_num, height, width) = cv_enet_model_output.shape[1:4]

  class_map = np.argmax(cv_enet_model_output[0], axis=0)

  mask_class_map = CV_ENET_SHAPE_IMG_COLORS[class_map]

  mask_class_map = cv2.resize(mask_class_map, (video_frame.shape[1], video_frame.shape[0]) ,
             interpolation = cv2.INTER_NEAREST)
  
  cv_enet_model_output = ( (0.3 * video_frame) + (0.7 * mask_class_map) ).astype('uint8')

  cv2.imshow("Frame", cv_enet_model_output)

  # 파일에 저장.
  # 원본은 가로세로 픽셀이 짝수인데, 아웃풋은 홀수로 나오기 때문에, 이미지 라사이징 필요
  cv_enet_model_output = cv2.resize(cv_enet_model_output, ( frame_width , frame_height ),
                            fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
  out.write(cv_enet_model_output)  
  # print( ( frame_height, frame_width ))
  # print(cv_enet_model_output.shape)
  # print(cv_enet_model_output.max())
  # print(cv_enet_model_output.min())

  # 키보드에서 esc키를 누르면 exit 하라는 것.
  if cv2.waitKey(25) & 0xFF == 27 :
    break

#   print(cv_enet_model_output)

sv.release()

cv2.destroyAllWindows()