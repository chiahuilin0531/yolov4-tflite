# !/usr/bin/python 
# coding:utf-8 
from PIL import Image , ImageEnhance
import cv2
import random
import os
import shutil
import numpy as np
import math
import random
from scipy import signal

#from general import LOGGER, check_version, colorstr, resample_segments, segment2box
#from metrics import bbox_ioa

debug=False
def dprint(*args):
    if debug:
        print(*args)

"""def input_image(img):
    #image = Image.open(path)
    #image = image.convert('RGBA')  # using when paste pic
    #image=rotate_image(image)
    #image=dark_image(image)
    image= hls_image(path)
    return img"""

def dark_image(image):
    enh_bri = ImageEnhance.Brightness(image)
    brightness = random.uniform(0.5,1.5)  # 產生指定範圍內的隨機浮點數
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened
def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d *= (1/gkern2d.max())
    return gkern2d

def hls_image(img):
    
    #img=cv2.imread(path)

    hlsImg = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

     # 亮度調整
    hlsImg[:, :, 1] = random.uniform(0.5,1.0) * hlsImg[:, :, 1]
    # 飽和度調整
    hlsImg[:, :, 2] = random.uniform(0.2,1.5) * hlsImg[:, :, 2]
    # 顏色空間反轉換 HLS -> BGR 
    result_img = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)
    return result_img

def copy_paste_hls_image(img_cv2,aug_num): #aug_cls n*5 num,xyxy,cls
    H,W,C = img_cv2.shape
    len = 1 if aug_num.ndim == 1 else aug_num.shape[0]
    img = Image.fromarray(cv2.cvtColor(img_cv2,cv2.COLOR_BGR2RGB)) #轉image
    A = np.zeros((img_cv2.shape[0],img_cv2.shape[1]))
    add_labels = np.array([])
    # 先建立一個ones mask, 有紅綠燈的地方是1.0, 其他是0.0
    for i in range(len) :
        x=aug_num[i][1]
        y=aug_num[i][2]
        w=aug_num[i][3]-aug_num[i][1]
        h=aug_num[i][4]-aug_num[i][2]
        A[y:y+h, x:x+w] = np.ones(A[y:y+h, x:x+w].shape)
    # 
    for i in range(len) :
        
        clss = aug_num[i][5]
        x=aug_num[i][1]
        y=aug_num[i][2]
        w=aug_num[i][3]-aug_num[i][1]
        h=aug_num[i][4]-aug_num[i][2]
        num = aug_num[i][0]
        if num == 0:
            continue
        # 被剪下來的紅綠燈圖片
        crop_img_cv2 = img_cv2[y:y+h, x:x+w] #crop
        # 貼在圖片上面2/3
        for j in range(num):

            # paste_x = random.randint(0,W-w)
            #paste_y = random.randint(y-100,y+100)
            # paste_y = random.randint(min(max(0,y-100),min(H,y+100)),max(max(0,y-100),min(H,y+100)))

            #crop_img_cv2_hls = hls_image(crop_img_cv2) #hls轉換
            crop_img = Image.fromarray(cv2.cvtColor(crop_img_cv2,cv2.COLOR_BGR2RGB)) #轉image

            while True :
                
                paste_x = random.randint(0,W-w)
                #paste_y = random.randint(y-100,y+100)
                paste_y = random.randint(min(max(0,y-100),min(H,y+100)),max(max(0,y-100),min(H,y+100)))
                """print("W:"+str(W))
                print("H:"+str(H))
                print("x:"+str(x))
                print("y:"+str(y)+": "+str(0)+"~"+str(W*3/4))
                print("paste_x:"+str(paste_x)+": "+str(W*1/4)+"~"+str(W*3/4))
                print("paste_y:"+str(paste_y)+": "+str(H*1/4)+"~"+str(H*3/4))"""
                

                #外面變中間要縮小(0.3~0.5)
                if paste_x > W*1/4 and paste_x < W*3/4 and paste_y > H*1/4 and paste_y < H*3/4 and (x <= W*1/4 or x>=W*3/4 or y<=H*1/4 or y>=H*3/4):
                    crop_img_resize = resize_image(crop_img,0.4,0.7)
                    #print("shrink")
                    #中間變外面要放大(2~3)
                elif (paste_x <= W*1/4 or paste_x >= W*3/4 or paste_y <= H*1/4 or paste_y >= H*3/4) and (x > W*1/4 and x<W*3/4 and y>H*1/4 and y<H*3/4):
                    #print("upsize")
                    continue
                    #crop_img_resize = resize_image(crop_img,2,3)
                    print("upsize")
                else:
                    crop_img_resize = crop_img

                
                if (clss==1 or clss==0) and crop_img_resize.size[0]*crop_img_resize.size[1]>=768:
                    #print("w*h= "+str(crop_img_resize.size[0]*crop_img_resize.size[1]))
                    crop_img_resize = resize_image(crop_img_resize,0.7,0.9)
                

                if(not A[paste_y:paste_y+crop_img_resize.size[1],paste_x:paste_x+crop_img_resize.size[0]].any()):
                    break
                else:
                    print("exist one")
                
            img.paste(crop_img_resize,(paste_x,paste_y)) #貼上
            add_labels = np.hstack([add_labels,str(paste_x)+","+str(paste_y)+","+str(paste_x+crop_img_resize.size[0])+","+str(paste_y+crop_img_resize.size[1])+","+str(clss)])
            #print(paste_x)
            #print(paste_x+crop_img_resize.size[0])
            #print(A[paste_y:paste_y+crop_img_resize.size[1],paste_x:paste_x+crop_img_resize.size[0]].shape)
            A[paste_y:paste_y+crop_img_resize.size[1],paste_x:paste_x+crop_img_resize.size[0]] = np.ones(A[paste_y:paste_y+crop_img_resize.size[1],paste_x:paste_x+crop_img_resize.size[0]].shape)

        """img_cv2 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        image_filter = np.zeros((img_cv2.shape[0], img_cv2.shape[1]), dtype=np.float64)
        for x in range(int(data[0]),int(data[2])):
            y=int(data[1])
            kernel = gkern(61, 12)
            for x1 in range(kernel.shape[0]):
                for y1 in range(kernel.shape[1]):
                    if  (x - int(kernel.shape[0]/2) + x1) < 0 or (x - int(kernel.shape[0]/2) + x1) >= img_cv2.shape[1]:
                        continue
                    elif (y - int(kernel.shape[1]/2) + y1) < 0 or (y - int(kernel.shape[1]/2) + y1) >= img_cv2.shape[0]:
                        continue
                    else :
                        image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1] = max(image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1], kernel[x1][y1])
        for x in range(int(data[0]),int(data[2])):
            y=int(data[3])
            kernel = gkern(61, 12)
            for x1 in range(kernel.shape[0]):
                for y1 in range(kernel.shape[1]):
                    if  (x - int(kernel.shape[0]/2) + x1) < 0 or (x - int(kernel.shape[0]/2) + x1) >= img_cv2.shape[1]:
                        continue
                    elif (y - int(kernel.shape[1]/2) + y1) < 0 or (y - int(kernel.shape[1]/2) + y1) >= img_cv2.shape[0]:
                        continue
                    else :
                        image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1] = max(image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1], kernel[x1][y1])
        for y in range(int(data[1]),int(data[3])):
            x=int(data[0])
            kernel = gkern(61, 12)
            for x1 in range(kernel.shape[0]):
                for y1 in range(kernel.shape[1]):
                    if  (x - int(kernel.shape[0]/2) + x1) < 0 or (x - int(kernel.shape[0]/2) + x1) >= img_cv2.shape[1]:
                        continue
                    elif (y - int(kernel.shape[1]/2) + y1) < 0 or (y - int(kernel.shape[1]/2) + y1) >= img_cv2.shape[0]:
                        continue
                    else :
                        image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1] = max(image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1], kernel[x1][y1])
        for y in range(int(data[1]),int(data[3])):
            x=int(data[2])
            kernel = gkern(61, 12)
            for x1 in range(kernel.shape[0]):
                for y1 in range(kernel.shape[1]):
                    if  (x - int(kernel.shape[0]/2) + x1) < 0 or (x - int(kernel.shape[0]/2) + x1) >= img_cv2.shape[1]:
                        continue
                    elif (y - int(kernel.shape[1]/2) + y1) < 0 or (y - int(kernel.shape[1]/2) + y1) >= img_cv2.shape[0]:
                        continue
                    else :
                        image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1] = max(image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1], kernel[x1][y1])
        
        #print(image.shape)
        #print(image_filter.shape)
        #image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
        
        fimage = img_cv2.astype(np.float32) 
        fimage = fimage / 255.0
        image_hsv = cv2.cvtColor(fimage, cv2.COLOR_BGR2HLS)
        hlsCopy = np.copy(image_hsv)

        hlsCopy[:, :, 1] = (1 +  image_filter*0.5) * hlsCopy[:, :, 1]
        hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1  # 應該要介於 0~1，計算出
        #image_hsv[:, :, 1] = (1 + image_filter ) * image_hsv[:, :, 1]
        #image_hsv[:, :, 1][image_hsv[:, :, 1] > 200] = 200
        #image = cv2.cvtColor(image_hsv,cv2.COLOR_HLS2BGR)
        # print(image_filter.max())
        # cv2.imwrite(FLAGS.output+'/test_blur/'+i[:-4]+'.jpg', image_filter*255)
        result_img = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
        result_img = ((result_img * 255).astype(np.uint8))
        cv2.imwrite( "/mnt/data0/Garmin/datasets/real_data/gaussian_blur"+'/'+title+"_comb"+'.jpg', result_img)
        cv2.imwrite( "/mnt/data0/Garmin/datasets/real_data/gaussian_blur"+'/'+title+'.jpg', image_filter*255)"""
    
    #print(add_labels)
    #print(A)
    return img, add_labels

def paste_image(img0, img1):
    img0.paste(img1,(300,300))
    return img0


def resize_image(image,a,b):
    width, height = image.size[:2]
    #print("width " + str(width))
    #print("height " + str(height))
    ratio_w =random.uniform(a,b)
    ratio_h =random.uniform(a,b)
    width = int(width * ratio_w)
    height = int(height * ratio_h)
    #print("width aft" + str(width))
    #print("height aft" + str(height)+"\n")
    #print("resiez w,h " + str(width)+" "+str(height))
    image = image.resize( (width, height), Image.BILINEAR )  #  or Image.NEAREST
    return image

def rotate_image(image):
    #angle=random.randint(0,360)
    #image = image.rotate( angle, Image.BILINEAR )
    image = image.transpose(random.randint(0,6))
    return image


# Main
if __name__=='__main__':
    
    data_root = "night_dataset"
    path= f"/mnt/27619746-8786-4cc0-a696-2b6bd7349a04/data/WeiJie/yolov4-tflite-new/datasets/{data_root}/images/"# "/mnt/data0/Garmin/datasets/real_data/data_selection_3/"
    aug_path= f"/mnt/27619746-8786-4cc0-a696-2b6bd7349a04/data/WeiJie/yolov4-tflite-new/datasets/{data_root}/images_copy_paste/"

    label_path=f"/mnt/27619746-8786-4cc0-a696-2b6bd7349a04/data/WeiJie/yolov4-tflite-new/datasets/{data_root}/anno/train_3cls.txt"
    aug_label_path=f"/mnt/27619746-8786-4cc0-a696-2b6bd7349a04/data/WeiJie/yolov4-tflite-new/datasets/{data_root}/anno_aug/train_3cls_copy_paste.txt"
    scale_aware_path = f"/mnt/27619746-8786-4cc0-a696-2b6bd7349a04/data/WeiJie/yolov4-tflite-new/datasets/{data_root}/anno_aug/train_3cls_scale_aware.txt"

    os.makedirs(os.path.dirname(aug_path), exist_ok=True)
    os.makedirs(os.path.dirname(aug_label_path), exist_ok=True)


    labels = open(label_path,'r')
    label = labels.readlines()
    aug_labels = open(aug_label_path,'w')
    scale_aware = open(scale_aware_path,'w')

    copy = 10
    pse=[0,0,0,0]
    pre_pse=[0,0,0,0]
    count=0
    expect_num = [8000,8000,8000,8000]
    red,green,yellow = 0, 0, 0
    for line in label:
        info = line.split(' ')
        boxnum = len(info)-1
        for i in range(1,boxnum+1):
            data = info[i].split(',') 
            pre_pse[int(data[4])]+=1
            
    for line in label:
        
        info = line.split(' ')
        info[-1] = info[-1].strip()
        pic_path=info[0].split('/')
        title,ext = pic_path[-1].split('.')
        name = pic_path[-1].strip()
        boxnum = len(info)-1
        img = cv2.imread(path + name)
        #aug_labels.writelines(line)
        #image = Image.open(path + pic_path[-1])
        #image.save(aug_path+pic_path[-1])
        print(f'path: {path} name: {name}|||||')
        print(str(count) + f"< {path + name} >")
        count+=1
        aug_num = np.empty((0,6))
        
        for i in range(1,boxnum+1):
            data = info[i].split(',') 
            #paste燈號數量
            if data[4] == "0": # green 100% copy paste
                if red%3 ==0:
                    num_pos = np.hstack([np.array([1]),data[0:5]]) 
                else:
                    num_pos = np.hstack([np.array([1]),data[0:5]]) 
                red+=1
                aug_num = np.vstack([aug_num,num_pos])
            elif data[4] == "1": # red 66% copy paste
                if yellow%3 ==0:
                    num_pos = np.hstack([np.array([0]),data[0:5]]) 
                else:
                    num_pos = np.hstack([np.array([1]),data[0:5]]) 
                yellow+=1
                aug_num = np.vstack([aug_num,num_pos])
            elif data[4] == "2": # yello 100% copy paste
                num_pos = np.hstack([np.array([1]),data[0:5]]) 
                aug_num = np.vstack([aug_num,num_pos])
        dprint(f'{count}:{path + name}\n',aug_num)
        aug_num = aug_num.astype('int')
        if aug_num[:,0].any() :
            for i in range(1):

                img1,add_labels = copy_paste_hls_image(img,aug_num)
                img1.save(aug_path+title+"_"+str(i+1)+".jpg")
                aug_labels.write(aug_path+title+"_"+str(i+1)+".jpg")
                scale_aware.write(aug_path+title+"_"+str(i+1)+".jpg")

                for k in range(1,boxnum+1):
                        aug_labels.write(" "+info[k])

                for j in range(len(add_labels)):
                    aug_labels.write(" "+add_labels[j])
                    scale_aware.write(" "+add_labels[j])

                aug_labels.write("\n")
                scale_aware.write("\n")
        
        #hls
        """
        for i in range(1,boxnum+1):
            data = info[i].split(',') 
            if data[4]=="0" :
                for j in range(1,2):
                    Motify_image=input_image(path + pic_path[-1])
                    cv2.imwrite(aug_path+title+"_"+str(j)+".jpg",Motify_image)
                    
                    aug_labels.writelines((info[0].split('.'))[0]+"_"+str(j)+".jpg")
                    for k in range(1,boxnum+1):
                        aug_labels.writelines(" "+info[k])
                    
                break
            elif data[4]=="1":
                for j in range(1,4):
                    #Motify_image=input_image(path + pic_path[-1])
                    #cv2.imwrite(aug_path+title+"_"+str(j)+".jpg",Motify_image)
                    
                    aug_labels.writelines((info[0].split('.'))[0]+"_"+str(j)+".jpg")
                    for k in range(1,boxnum+1):
                        aug_labels.writelines(" "+info[k])
                    
                break
            if data[4]=="2":
                for j in range(1,10):
                    Motify_image=input_image(path + pic_path[-1])
                    cv2.imwrite(aug_path+title+"_"+str(j)+".jpg",Motify_image)
                    
                    aug_labels.writelines((info[0].split('.'))[0]+"_"+str(j)+".jpg")
                    for k in range(1,boxnum+1):
                        aug_labels.writelines(" "+info[k])
                    
                break"""

    aug_labels.close()
    aug_labels = open(aug_label_path,'r')
    auged = aug_labels.readlines()
    for line in auged:
        info = line.split(' ')
        boxnum = len(info)-1
        for i in range(1,boxnum+1):
            data = info[i].split(',') 
            pse[int(data[4])]+=1
            
    print("aug: "+str(pse))
    print("original: " + str(pre_pse) )
    print("total: " + str(np.array(pse) + np.array(pre_pse)))
    labels.close()
    aug_labels.close()
