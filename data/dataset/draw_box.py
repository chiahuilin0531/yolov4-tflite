import numpy as np
import cv2
import os
from absl import app, flags, logging
from absl.flags import FLAGS

# Directory
directory = "check_box"
parent_dir = "/home/tsneg-aimm/Garmin/"
path = os.path.join(parent_dir, directory)
os.mkdir(path)

f = open("train_countdown_3cls.txt",'r')
# flags.DEFINE_string('file', 'train_mix_3cls.txt', 'xxxxxx.txt')

for line in f:
    s = line.split(' ')
    img_path = s[0]
    name = img_path.split('/')[-1]
    boxes = s[1:]

    img = cv2.imread(img_path)
    red_color = (0, 0, 255) # BGR
    for box in boxes:
        a = box.split(',')
        x1 = int(a[0])
        y1 = int(a[1])
        x2 = int(a[2])
        y2 = int(a[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), red_color, 3, cv2.LINE_AA)
        cv2.putText(img, 'object '+a[4][0], (x1,y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.6, red_color, 1)
    
    

    # cv2.imshow('My Image', img)

    # # 按下任意鍵則關閉所有視窗
    cv2.imwrite(path+'/'+name, img)