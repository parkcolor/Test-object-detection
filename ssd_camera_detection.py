import tensorflow as tf
import os
import pathlib

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import cv2
import time

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


# 내 로컬에 설치된 TFOD 경로
PATH_TO_LABELS = 'C:\\Users\\5-15\\Documents\\peter\\Tensorflow\\models\\research\\object_detection\\data\\mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
print(category_index)

def load_model(model_name):

    # http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz

    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)
    

    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))

    return model
# http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz
# mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.

# /download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model =  load_model(model_name)

print(  detection_model.signatures['serving_default'].inputs   )
print(detection_model.signatures['serving_default'].output_dtypes)
print( detection_model.signatures['serving_default'].output_shapes )

def run_inference_for_single_image(model, image):
  # 넘파이 어레이로 바꿔준다.
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}

  # print(output_dict)
  
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = tf.convert_to_tensor(output_dict['detection_masks'], dtype=tf.float32)
    output_dict['detection_boxes'] = tf.convert_to_tensor(output_dict['detection_boxes'], dtype=tf.float32)
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])  
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict


def show_inference(model, image_np):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.array(output_dict['detection_boxes']),
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed',None),
        use_normalized_coordinates=True,
        line_thickness=8)

    cv2.imshow("result", image_np)

    return image_np

## 이미지 경로에 있는 이미지를 실행
# PATH_TO_TEST_IMAGE_DIR = pathlib.Path('data\\images')
# TEST_IMAGE_PATH =   sorted(  list(PATH_TO_TEST_IMAGE_DIR.glob("*.jpg"))  )
# for image_path in TEST_IMAGE_PATH :
#     show_inference(detection_model, image_path)

## 비디오를 실행하는 코드
cap = cv2.VideoCapture('data/videos/video.mp4')

## 카메라의 영상을 실행하는 코드
# cap = cv2.VideoCapture(0)

if cap.isOpened() == False :
    print("Error opening video stream or file")

else :
    # 반복문이 필요한 이유?  비데오는 여러 사진으로 구성되어있으니까
    # 여러개니까!
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter('data/videos/output.avi', cv2.VideoWriter_fourcc(*'H264'),10,(frameWidth,frameHeight))
    while cap.isOpened() :
        # 사진을 1장씩 가져와서.
        ret, frame = cap.read()
        # 제대로 사진 가져왔으면, 화면에 표시!
        if ret == True:
            ### 이 부분을 모델 추론하고 화면에 보여주는 코드로 변경 
            # cv2.imshow("Frame", frame)

            start_time = time.time()
            img = show_inference(detection_model, frame )
            end_time = time.time()
            # print(end_time - start_time)
            out.write(img)
            print(img.shape)

            # 키보드에서 esc키를 누르면 exit 하라는 것.
            if cv2.waitKey(25) & 0xFF == 27 :
                break
        else : 
            break

    cap.release()
    cv2.destroyAllWindows()