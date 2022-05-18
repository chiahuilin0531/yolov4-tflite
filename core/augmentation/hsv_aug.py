# !/usr/bin/python 
# coding:utf-8 
from PIL import Image , ImageEnhance
import cv2
import random
import os
import shutil
import numpy as np


def input_image(path):
    #image = Image.open(path)
    #image = image.convert('RGBA')  # using when paste pic
    #image=rotate_image(image)
    #image=dark_image(image)
    image= hls_image(path)
    return image

def dark_image(image):
    enh_bri = ImageEnhance.Brightness(image)
    brightness = random.uniform(0.5,1.5)  # 產生指定範圍內的隨機浮點數
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened

def hls_image(path):
    
    img=cv2.imread(path)

    hlsImg = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

     # 亮度調整
    hlsImg[:, :, 1] = random.uniform(0.5,1.0) * hlsImg[:, :, 1]
    # 飽和度調整
    hlsImg[:, :, 2] = random.uniform(0.2,1.5) * hlsImg[:, :, 2]
    # 顏色空間反轉換 HLS -> BGR 
    result_img = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)
    return result_img

def resize_image(image):
    width, height = image.size[:2]
    ratio_w =random.uniform(0.7,1.3)
    ratio_h =random.uniform(0.7,1.3)
    width = int(width * ratio_w)
    height = int(height * ratio_h)
    image = image.resize( (width, height), Image.BILINEAR )  #  or Image.NEAREST
    return image

def rotate_image(image):
    #angle=random.randint(0,360)
    #image = image.rotate( angle, Image.BILINEAR )
    image = image.transpose(random.randint(0,6))
    return image

# Main
if __name__=='__main__':
    case="hls"    # bri= brightness", res= resize
    path="/mnt/data0/Garmin/datasets/real_data/images/"
    aug_path="/mnt/data0/Garmin/datasets/aug_data/image_hsv/"
    label_path="/mnt/data0/Garmin/datasets/real_data/anno/train_real_4cls.txt"
    aug_label_path="/mnt/data0/Garmin/datasets/aug_data/anno/train_hsv_4cls"
    #allFilelist = os.listdir(path) 
    labels = open(label_path,'r')
    label = labels.readlines()
    aug_labels = open(aug_label_path,'w')
    copy = 10
    pse=[0,0,0,0]
    pre_pse=[0,0,0,0]
    count=0
    expect_num = [8000,8000,8000,8000]

    for line in label:
        info = line.split(' ')
        boxnum = len(info)-1
        for i in range(1,boxnum+1):
            data = info[i].split(',') 
            pre_pse[int(data[4])]+=1
            
    for line in label:
        
        info = line.split(' ')
        pic_path=info[0].split('/')
        title,ext = pic_path[-1].split('.')
        name = pic_path[-1]
        boxnum = len(info)-1
        aug_labels.writelines(line)
        if count%100 == 0:
            print(str(count)+"\n"+info[0]+"\n\n")
        count+=1
        image = Image.open(info[0])
        image.save(aug_path+pic_path[-1])
        for i in range(1,boxnum+1):
            data = info[i].split(',') 
            if data[4]=="0" :
                for j in range(1,2):
                    Motify_image=input_image(info[0])
                    cv2.imwrite(aug_path+title+"_"+str(j)+".jpg",Motify_image)
                    #Motify_image.save(aug_path+title+"_"+str(j)+".jpg")
                    aug_labels.writelines((info[0].split('.'))[0]+"_"+str(j)+".jpg")
                    for k in range(1,boxnum+1):
                        aug_labels.writelines(" "+info[k])
                    #aug_labels.writelines("\n")
                break
            """if data[4]=="1":
                for j in range(1,):
                    #Motify_image=input_image(info[0])
                    #cv2.imwrite(aug_path+title+"_"+str(j)+".jpg",Motify_image)
                    #Motify_image.save(aug_path+title+"_"+str(j)+".jpg")
                    aug_labels.writelines((info[0].split('.'))[0]+"_"+str(j)+".jpg")
                    for k in range(1,boxnum+1):
                        aug_labels.writelines(" "+info[k])
                    #aug_labels.writelines("\n")
                break"""
            if data[4]=="2":
                for j in range(1,25):
                    Motify_image=input_image(info[0])
                    cv2.imwrite(aug_path+title+"_"+str(j)+".jpg",Motify_image)
                    #Motify_image.save(aug_path+title+"_"+str(j)+".jpg")
                    aug_labels.writelines((info[0].split('.'))[0]+"_"+str(j)+".jpg")
                    for k in range(1,boxnum+1):
                        aug_labels.writelines(" "+info[k])
                    #aug_labels.writelines("\n")
                break

    aug_labels.close()
    aug_labels = open(aug_label_path,'r')
    auged = aug_labels.readlines()
    for line in auged:
        info = line.split(' ')
        boxnum = len(info)-1
        for i in range(1,boxnum+1):
            data = info[i].split(',') 
            pse[int(data[4])]+=1
            
    print("pse: "+str(pse))
    labels.close()
    aug_labels.close()

    """for img_name in allFilelist:
        title,ext = img_name.split('.') 
        Motify_image=input_image(path+'/'+img_name)
        # copy the txt from file to file
        #shutil.copy(label_path+title+".txt", 
        #            res_label_path+title+str("_")+case+".txt")
        # generate augment jpg in file    
        if case == "hls": 
            cv2.imwrite(res_path+title+str("_")+case+".jpg",Motify_image)
        elif case == "res":
            Motify_image.save(res_path+title+str("_")+case+".jpg")"""



