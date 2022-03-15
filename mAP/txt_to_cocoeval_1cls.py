import os
import cv2
import json
from tqdm import tqdm
import argparse
import json
from pycocotools.coco import COCO
from cocoeval import COCOeval

parser = argparse.ArgumentParser()
parser.add_argument('--label_dir', '--gt', default='./mAP/ground-truth',type=str, help="label_dir which contains txts or label_json in coco format")
parser.add_argument('--pred_dir', '--dt', default='./mAP/predicted',type=str, help="prediction_dir which contains txts or prediction_json in coco format")
# parser.add_argument('--ct', '--conf-thres', type=float, default=0.25, help='object confidence threshold')
args = parser.parse_args()

def yolo2coco(label_path, pred_path):
    # id_dict = {}
    if label_path[-1] == '/':
        label_path = label_path[:-1]
    if pred_path[-1] == '/':
        pred_path = pred_path[:-1]
        
    if '.json' == label_path[-5:]:
        label_json = label_path
        print('Loading GT JSON file from '+label_json)
    else:
        files = os.listdir(label_path)

        labels = []
        images = []
        classes = [{"supercategory": "tl","id": 0,"name": "traffic light"}]
        i = 0
        print('Ground Truth yolo txt to coco json...')
        for file in tqdm(files):
            if '.txt' not in file:
                continue
            # img = cv2.imread('imgs/'+file[:-4]+'.jpg')

            # W = img.shape[1]
            # H = img.shape[0]

            W = 3000
            H = 3000
            images.append({'file_name':file[:-4]+'.jpg', 'height': H, 'width': W, 'id':int(file[:-4])})

            f1 = open(label_path+'/'+file, 'r')
            for line in f1.readlines():
                c, x1, y1, x2, y2 = line.split()
                x1 = float(x1)
                y1 = float(y1)
                w = float(x2) - x1
                h = float(y2) - y1
                bb = [x1, y1, w, h]
                labels.append({'id':i, 'image_id':int(file[:-4]), 'category_id':0, 'bbox':[round(b, 3) for b in bb], 'area':bb[2]*bb[3], 'iscrowd':0})
                i += 1
            f1.close()

        label_json = os.path.join(os.path.dirname(label_path), label_path.split('/')[-1])+'.json'
        print('Saving JSON file to '+label_json)
        with open(label_json, 'w') as f:
            json.dump({'annotations':labels, 'images':images, 'categories':classes}, f)
    if '.json' == pred_path[-5:]:
        pred_json = pred_path
        print('Loading DT JSON file from '+pred_json)
    else:
        files = os.listdir(pred_path)

        predictions = []
        print('Predictions yolo txt to coco json...')
        for file in tqdm(files):
            if '.txt' not in file:
                continue

            f2 = open(pred_path+'/'+file, 'r')
            for line in f2.readlines():
                c, s, x1, y1, x2, y2 = line.split()
                # if float(s) < args.ct:
                #     continue
                x1 = float(x1)
                y1 = float(y1)
                w = float(x2) - x1
                h = float(y2) - y1
                bb = [x1, y1, w, h]
                predictions.append({'image_id':int(file[:-4]), 'category_id':0, 'bbox':[round(b, 3) for b in bb], 'score':round(float(s), 5)})
            f2.close()
        pred_json = os.path.join(os.path.dirname(pred_path), pred_path.split('/')[-1])+'.json'
        print('Saving JSON file to '+pred_json)
        with open(pred_json, 'w') as f:
            json.dump(predictions, f)
    return label_json, pred_json

label_path = args.label_dir
pred_path = args.pred_dir
l_json, p_json = yolo2coco(label_path,pred_path)
cocoGt=COCO(l_json)
cocoDt=cocoGt.loadRes(p_json)
cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
# cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 1022], [1022, 3809], [3809, 1e5 ** 2]]
cocoEval.params.areaRng = [[123, 1e5 ** 2], [123, 341], [341, 768],  [768, 1e5 ** 2]]
cocoEval.params.areaRngLbl=['all','small', 'medium', 'large']
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
