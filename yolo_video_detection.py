import numpy as np
import cv2
from yolo.model.yolo_model import YOLO
import time

def process_image( img ) :
    ''' 이미지 리사이즈하고, 차원을 확장하는 함수
    img  : 원본 이미지
    결과 : (64,64,3) 으로 프로세싱된 이미지 반환 '''

    image_org = cv2.resize(img, (416,416), interpolation = cv2.INTER_CUBIC)

    image_org = image_org / 255.0
    image_org = np.expand_dims(image_org, axis=0)

    return image_org


def get_classess(file):
    '''   클래스의 이름을 리스트로 가져온다.   '''
    with open(file) as f :
        name_of_class = f.readlines()
    
    name_of_class = [  class_name.strip() for class_name in name_of_class  ]

    return name_of_class


def box_draw(image, boxes, scores, classes, all_classes):
    '''
    image   : 오리지날 이미지
    boxes   : 물체의 박스 ( ndarray )
    scores  : 오브젝트의 클래스 정보 ( ndarray )
    classes : 오브젝트의 확률 ( ndarray )
    all_classes : 모든 클래스의 이름
    '''
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()


def detect_image( image, yolo, all_classes ):
    '''
    image : 오리지날 이미지
    yolo  : 욜로 모델
    all_classes : 전체 클래스 이름

    변환된 이미지를 반환한다.
    '''
    pimage = process_image(image)

    image_boxes, image_classes, image_scores = yolo.predict(pimage, image.shape)

    if image_boxes is not None :
        box_draw( image, image_boxes, image_scores, image_classes, all_classes )

    return image

yolo = YOLO(0.6, 0.5)
all_classes = get_classess('yolo/data/coco_classes.txt')
print(all_classes)

cap = cv2.VideoCapture('data/videos/video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        start_time = time.time()
        result_image = detect_image(frame, yolo, all_classes)
        end_time = time.time()

        print(end_time - start_time)
        cv2.imshow('Frame', result_image)

        if cv2.waitKey(25) & 0xFF ==27:
            break

    else:
        break
cap.release()
cv2.destroyAllWindows()
