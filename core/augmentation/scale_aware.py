import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt
import os
from absl import app, flags, logging
from absl.flags import FLAGS
import shutil
from tqdm import tqdm

flags.DEFINE_string('annotation', "/mnt/data0/Garmin/datasets/aug_data/anno/03_scale_aware.txt",
                    'path to annotation file')
flags.DEFINE_string('image', "/mnt/data0/Garmin/datasets/aug_data/images_3",
                    'path to image file')
flags.DEFINE_string('output', "/mnt/data0/Garmin/datasets/aug_data/03_images_sw/",
                    'path to output file')


def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gkern2d *= (1/gkern2d.max())
    return gkern2d


# def draw_kernel(img, x=0, y=0, kernlen=21, std=3):
#     kernel = gkern(kernlen, std)


"""def main(_argv):
    files = os.listdir(FLAGS.annotation)
    os.makedirs(FLAGS.output, exist_ok=True)
    os.makedirs(FLAGS.output+'/test_blur', exist_ok=True)
    os.makedirs(FLAGS.output+'/test_image', exist_ok=True)
    for i in tqdm(files):
        anno_data = open(FLAGS.annotation+'/'+i, 'r')
        image = cv2.imread(FLAGS.image+'/'+i[:-4]+'.jpg')   #read image shape
        shutil.copy2(FLAGS.image+'/'+i[:-4]+'.jpg', FLAGS.output+'/test_image')  #copy image to output path
        image_filter = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)
        for j in anno_data:
            anno = j.split(',')
            x = int(int(anno[0]) + int(anno[2])/2)
            y = int(int(anno[1]) + int(anno[3])/2)
            kernel = gkern(61, 12)
            for x1 in range(kernel.shape[0]):
                for y1 in range(kernel.shape[1]):
                    print(int(kernel.shape[0]))
                    if  (x - int(kernel.shape[0]/2) + x1) < 0 or (x - int(kernel.shape[0]/2) + x1) >= image.shape[1]:
                        continue
                    elif (y - int(kernel.shape[1]/2) + y1) < 0 or (y - int(kernel.shape[1]/2) + y1) >= image.shape[0]:
                        continue
                    else :
                        image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1] = max(image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1], kernel[x1][y1])
        # print(image_filter.max())
        # cv2.imwrite(FLAGS.output+'/test_blur/'+i[:-4]+'.jpg', image_filter*255)
        cv2.imwrite(FLAGS.output+'/'+i[:-4]+'.jpg', image_filter*255)
        # print(image_filter.max())
        break

    # print(image.shape)
    # print(image.shape[1])

    # cv2.imwrite('test.jpg', gkern(61, 12)*255)
    # plt.imshow(gkern(21), interpolation='none')
    # plt.imsave('test.png',gkern(61, 12))
"""
def main(_argv):
    anno_data = open(FLAGS.annotation)
    os.makedirs(FLAGS.output, exist_ok=True)
    for line in anno_data:
        print(line)
        info = line.split(' ')
        info[-1] = info[-1].replace("\n","")
        pic_path=info[0].split('/')
        title,ext = pic_path[-1].split('.')
        name = pic_path[-1]
        boxnum = len(info)-1
        
        image = cv2.imread(info[0])   #read image shape
        #shutil.copy2(FLAGS.image+'/'+i[:-4]+'.jpg', FLAGS.output+'/test_image')  #copy image to output path
        image_filter = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)
        for j in range(1,boxnum+1):
            
            data = info[j].split(',')
            for x in range(int(data[0]),int(data[2])):
                y=int(data[1])
                kernel = gkern(61, 12)
                for x1 in range(kernel.shape[0]):
                    for y1 in range(kernel.shape[1]):
                        if  (x - int(kernel.shape[0]/2) + x1) < 0 or (x - int(kernel.shape[0]/2) + x1) >= image.shape[1]:
                            continue
                        elif (y - int(kernel.shape[1]/2) + y1) < 0 or (y - int(kernel.shape[1]/2) + y1) >= image.shape[0]:
                            continue
                        else :
                            image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1] = max(image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1], kernel[x1][y1])
            for x in range(int(data[0]),int(data[2])):
                y=int(data[3])
                kernel = gkern(61, 12)
                for x1 in range(kernel.shape[0]):
                    for y1 in range(kernel.shape[1]):
                        if  (x - int(kernel.shape[0]/2) + x1) < 0 or (x - int(kernel.shape[0]/2) + x1) >= image.shape[1]:
                            continue
                        elif (y - int(kernel.shape[1]/2) + y1) < 0 or (y - int(kernel.shape[1]/2) + y1) >= image.shape[0]:
                            continue
                        else :
                            image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1] = max(image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1], kernel[x1][y1])
            for y in range(int(data[1]),int(data[3])):
                x=int(data[0])
                kernel = gkern(61, 12)
                for x1 in range(kernel.shape[0]):
                    for y1 in range(kernel.shape[1]):
                        if  (x - int(kernel.shape[0]/2) + x1) < 0 or (x - int(kernel.shape[0]/2) + x1) >= image.shape[1]:
                            continue
                        elif (y - int(kernel.shape[1]/2) + y1) < 0 or (y - int(kernel.shape[1]/2) + y1) >= image.shape[0]:
                            continue
                        else :
                            image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1] = max(image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1], kernel[x1][y1])
            for y in range(int(data[1]),int(data[3])):
                x=int(data[2])
                kernel = gkern(61, 12)
                for x1 in range(kernel.shape[0]):
                    for y1 in range(kernel.shape[1]):
                        if  (x - int(kernel.shape[0]/2) + x1) < 0 or (x - int(kernel.shape[0]/2) + x1) >= image.shape[1]:
                            continue
                        elif (y - int(kernel.shape[1]/2) + y1) < 0 or (y - int(kernel.shape[1]/2) + y1) >= image.shape[0]:
                            continue
                        else :
                            image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1] = max(image_filter[y - int(kernel.shape[1]/2) + y1][x - int(kernel.shape[0]/2) + x1], kernel[x1][y1])
            
        #print(image.shape)
        #print(image_filter.shape)
        #image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
        
        fimage = image.astype(np.float32) 
        fimage = fimage / 255.0
        image_hsv = cv2.cvtColor(fimage, cv2.COLOR_BGR2HLS)
        hlsCopy = np.copy(image_hsv)

        hlsCopy[:, :, 1] = (1 +  image_filter*0.3) * hlsCopy[:, :, 1]
        hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1  # 應該要介於 0~1，計算出
        #image_hsv[:, :, 1] = (1 + image_filter ) * image_hsv[:, :, 1]
        #image_hsv[:, :, 1][image_hsv[:, :, 1] > 200] = 200
        #image = cv2.cvtColor(image_hsv,cv2.COLOR_HLS2BGR)
        # print(image_filter.max())
        # cv2.imwrite(FLAGS.output+'/test_blur/'+i[:-4]+'.jpg', image_filter*255)
        result_img = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
        result_img = ((result_img * 255).astype(np.uint8))
        cv2.imwrite(FLAGS.output+'/'+title+'.jpg', result_img)
        #cv2.imwrite(FLAGS.output+'/'+title+'.jpg', image_filter*255)
        # print(image_filter.max())
        #break
        

    # print(image.shape)
    # print(image.shape[1])

    # cv2.imwrite('test.jpg', gkern(61, 12)*255)
    # plt.imshow(gkern(21), interpolation='none')
    # plt.imsave('test.png',gkern(61, 12))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass