from core.common import MishLayer,BatchNormalization
from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset_tiny import Dataset
from datetime import datetime
# from core.dataset import Dataset


from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from tensorflow.keras import layers

flags.DEFINE_string('models', 'yolov4,yolov4', 'models separated by commas')
flags.DEFINE_string('weights', None, 'pretrained weights')
flags.DEFINE_string('tinys', 'False,False', 'yolo or yolo-tiny list separated by commas')
flags.DEFINE_string('save_path','DML','save path for the models')
    
def main(_argv):
    print('Saving in', "./checkpoints/DML/" + str(FLAGS.save_path))

    models = FLAGS.models.split(',')
    tinys = np.array(FLAGS.tinys.split(',')) == "True" 
    number_models = len(models)

    trainset = Dataset(models[0], tinys[0], is_training=True)
    testset = Dataset(models[0], tinys[0], is_training=False)
    logdir = "./data/log"
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch

    # Define all the models

    keras_models = []

    for i in range(number_models):
        
        input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(models[i], tinys[i])
        IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

        freeze_layers = utils.load_freeze_layer(models[i], tinys[i])

        feature_maps = YOLO(input_layer, NUM_CLASS, models[i], tinys[i])
        if tinys[i]:
            print("---------------------- Student tiny", tinys[i])
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                if i == 0:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                else:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
        else:
            print("---------------------- Student NOT tiny")
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                if i == 0:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                elif i == 1:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                else:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)

        keras_models.append(tf.keras.Model(input_layer, bbox_tensors))

    optimizer = tf.keras.optimizers.Adam()
    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)




    class Distiller(keras.Model):
        def __init__(self):
            super(Distiller, self).__init__()

        def compile(
            self,
            optimizer,
            metrics,
            distillation_loss_fn,
            alpha=0.1,
            temperature=3
        ):
            super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
            self.distillation_loss_fn = distillation_loss_fn
            self.alpha = alpha
            self.temperature = temperature

        # define training step function
        # @tf.function
        def train_step(self, image_data, target, pred_results, current_model):
            with tf.GradientTape() as tape:

                k = current_model
                model = keras_models[k]
                pred_result = model(image_data, training=True)
                giou_loss = conf_loss = prob_loss = 0

                # optimizing process
                for i in range(len(freeze_layers)):
                    conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                    loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                    giou_loss += loss_items[0]
                    conf_loss += loss_items[1]
                    prob_loss += loss_items[2]

                self_loss = giou_loss + conf_loss + prob_loss
                kl_loss = 0
                for j in range(number_models):
                    if k != j:
                        for p in range(len(pred_results[k])):
                            kl_loss += self.distillation_loss_fn(
                                                tf.nn.log_softmax(pred_results[k][p], axis=1),
                                                tf.nn.softmax(pred_results[j][p], axis=1)
                                )

                total_loss = self_loss + kl_loss / (number_models - 1)

                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                if global_steps%200==0 : 


                    now = datetime.now()

                    current_time = now.strftime("%H:%M:%S")
                    print("Current Time =", current_time)
                    tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                         "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(),
                                                                   giou_loss, conf_loss,
                                                                   prob_loss, total_loss))
                if k == number_models - 1:
                        
                    # update learning rate
                    global_steps.assign_add(1)
                    if global_steps < warmup_steps:
                        lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
                    else:
                        lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                            (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                        )
                    optimizer.lr.assign(lr.numpy())


        # define test step function
        # @tf.function        
        def test_step(self, image_data, target):
            with tf.GradientTape() as tape:


                pred_result = []
                for model in keras_models:
                    pred_result.append(model(image_data, training=True))

                for k in range(number_models):
                    model = keras_models[k]
                    giou_loss = conf_loss = prob_loss = 0
                    
                    # optimizing process
                    for i in range(len(freeze_layers)):
                        conv, pred = pred_result[k][i * 2], pred_result[k][i * 2 + 1]
                        loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                        giou_loss += loss_items[0]
                        conf_loss += loss_items[1]
                        prob_loss += loss_items[2]

                    total_loss = giou_loss + conf_loss + prob_loss
                    if global_steps%200==0 : 
                        tf.print("=> TEST STEP %4d Model#%4d giou_loss: %4.2f   conf_loss: %4.2f   "
                             "prob_loss: %4.2f   total_loss: %4.2f" % (k, global_steps, giou_loss, conf_loss,
                                                                       prob_loss, total_loss))

    distiller = Distiller()
    distiller.compile(
        optimizer=optimizer,
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )

    for epoch in range(first_stage_epochs + second_stage_epochs):
        if epoch < first_stage_epochs:
            if not isfreeze:
                isfreeze = True
                for name in freeze_layers:
                    contador = 0
                    for model in keras_models:
                        new_name = name.split("_")[0] + "_" + str((int)(name.split("_")[1]) + contador)
                        freeze = model.get_layer(new_name)
                        freeze_all(freeze)
                        contador += 21
        elif epoch >= first_stage_epochs:
            if isfreeze:
                isfreeze = False
                print(freeze_layers)
                for name in freeze_layers:
                    contador = 0
                    for model in keras_models:
                        model.summary()
                        new_name = name.split("_")[0] + "_" + str((int)(name.split("_")[1]) + contador)
                        freeze = model.get_layer(new_name)
                        freeze_all(freeze)
                        contador += 21
            
        for image_data, target in trainset:
            pred_results = []
            for model in keras_models:
                pred_results.append(model(image_data, training=True))
            for i in range(number_models):
                distiller.train_step(image_data, target, pred_results, i)
        for image_data, target in testset:
            distiller.test_step(image_data, target)


        # original 
        for i in range(number_models):
            keras_models[i].save_weights("./checkpoints/DML/" + str(FLAGS.save_path) + "_" + str(i))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass