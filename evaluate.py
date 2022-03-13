from builtins import NotImplementedError
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
from core.yolov4 import filter_boxes,decode
from tensorflow.python.saved_model import tag_constants
import core.utils as utils
from tqdm import tqdm
from core.config import cfg

flags.DEFINE_string('weights', './data/yolov4.weights',
                    'path to weights file')
flags.DEFINE_string('framework', 'tf', 'select model type in (tf, tflite, tf_ckpt, trt)'
                    'path to weights file')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('input_size', 416, 'resize images to')
flags.DEFINE_string('annotation_path', "./data/dataset/val2017.txt", 'annotation path')
flags.DEFINE_boolean('draw_image', False, 'write image path')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

def main(_argv):
    INPUT_SIZE = FLAGS.input_size
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)

    predicted_dir_path = './mAP/predicted'
    ground_truth_dir_path = './mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

    # Build Model According to different model weight format
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # elif FLAGS.framework == 'tf_ckpt':
    #     infer = tf.keras.models.load_model(FLAGS.weights)
    #     infer.summary()
    elif FLAGS.framework == 'tf':
        # saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        # infer = saved_model_loaded.signatures['serving_default']
        infer = tf.keras.models.load_model(FLAGS.weights)
    else:
        raise NotImplementedError(f"no such frame detected: {FLAGS.framework}")

    # Number of object in annotation path
    num_lines = sum(1 for line in open(FLAGS.annotation_path))
    with open(FLAGS.annotation_path, 'r') as annotation_file:
        lines = annotation_file.readlines()
        print(FLAGS.annotation_path, '  #########################################')
        with tqdm(total=len(lines), ncols=150, desc="Evaluating: ") as pbar:
            for num, line in enumerate(lines):
                # Load Image From Annotation File
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Process Annotation in each line 
                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])
                if len(bbox_data_gt) == 0:
                    bboxes_gt = []
                    classes_gt = []
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

                # print('=> ground truth of %s:' % image_name)
                num_bbox_gt = len(bboxes_gt)
                # Write ground truth  Of each image and write them to another folder
                with open(ground_truth_path, 'w') as f:
                    for i in range(num_bbox_gt):
                        class_name = CLASSES[classes_gt[i]]
                        xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                        bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        # print('\t' + str(bbox_mess).strip())

                # print('=> predict result of %s:' % image_name)
                predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
                # Predict Process
                image_size = image.shape[:2]
                image_data = cv2.resize(np.copy(image), (INPUT_SIZE, INPUT_SIZE))
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)

                # Inference Code for different format of model
                if FLAGS.framework == 'tflite':
                    interpreter.set_tensor(input_details[0]['index'], image_data)
                    interpreter.invoke()
                    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                    if FLAGS.model == 'yolov4' and FLAGS.tiny == True:
                        boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape = tf.constant([INPUT_SIZE,INPUT_SIZE]))
                    else:
                        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape = tf.constant([INPUT_SIZE,INPUT_SIZE]))
                # elif FLAGS.framework == 'tf_ckpt':
                #     batch_data = tf.constant(image_data)
                #     pred_bbox = infer(batch_data)
                #     boxes = pred_bbox[0] / np.array([INPUT_SIZE,INPUT_SIZE,INPUT_SIZE,INPUT_SIZE], dtype='float32')
                #     boxes = tf.concat([
                #         (boxes[..., :2] - boxes[..., 2:] / 2.0)[...,::-1],
                #         (boxes[..., :2] + boxes[..., 2:] / 2.0)[...,::-1],
                #     ], axis=-1)
                #     pred_conf = pred_bbox[1]
                elif FLAGS.framework == 'tf':
                    batch_data = tf.constant(image_data)
                    pred_bbox = infer(batch_data)
                    boxes = pred_bbox[..., 0:4]
                    pred_conf = pred_bbox[..., 4:]
                    # print('pred_bbox', pred_bbox)
                    # for key, value in pred_bbox.items():
                    #     boxes = value[:, :, 0:4]
                    #     pred_conf = value[:, :, 4:]
                else:
                    raise NotImplementedError()

                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=FLAGS.iou,
                    score_threshold=FLAGS.score
                )

                boxes, scores, classes, valid_detections = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()] 

                if cfg.TEST.DECTECTED_IMAGE_PATH is not None and FLAGS.draw_image:
                    image_result = utils.draw_bbox(np.copy(image), [boxes, scores, classes, valid_detections])
                    cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH + image_name, image_result)

                with open(predict_result_path, 'w') as f:
                    image_h, image_w, _ = image.shape
                    for i in range(valid_detections[0]):
                        if int(classes[0][i]) < 0 or int(classes[0][i]) > NUM_CLASS: continue
                        coor = boxes[0][i]
                        coor[0] = int(coor[0] * image_h)
                        coor[2] = int(coor[2] * image_h)
                        coor[1] = int(coor[1] * image_w)
                        coor[3] = int(coor[3] * image_w)

                        score = scores[0][i]
                        class_ind = int(classes[0][i])
                        class_name = CLASSES[class_ind]
                        score = '%.4f' % score
                        ymin, xmin, ymax, xmax = list(map(str, coor))
                        bbox_mess = f'{class_name:6s} {xmin:>4s} {ymin:>4s} {xmax:>4s} {ymax:>4s} {score}'
                        f.write(' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n')
                        
                pbar.set_postfix({
                    'detected_obj': f'{valid_detections[0]:2d}',
                    'image_path': f"{image_path}",
                })
                pbar.update(1)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


