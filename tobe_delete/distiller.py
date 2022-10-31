from core.common import MishLayer,BatchNormalization
from absl import app, flags, logging
from absl.flags import FLAGS
import os, sys
from core.accumulator import Accumulator
import shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset_tiny import tfDataset
from datetime import datetime
from tqdm import tqdm
# from core.dataset import Dataset


from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all, init_shared_variable
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from tensorflow.keras import layers

flags.DEFINE_string('models', 'yolov4,yolov4', 'models separated by commas')
flags.DEFINE_string('model', 'yolov4', 'models separated by commas')
flags.DEFINE_string('tinys', 'True,True', 'yolo or yolo-tiny list separated by commas')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny list separated by commas')
flags.DEFINE_string('weights', None, 'pretrained weights')
flags.DEFINE_string('save_dir','checkpoints/DML','save path for the models')
flags.DEFINE_boolean('qat', False, 'train w/ or w/o quatize aware')
# tf.config.optimizer.set_jit(True)

def apply_quantization(layer):
    # if isinstance(layer, tf.python.keras.engine.base_layer.TensorFlowOpLayer):
    if 'tf_op' in layer.name or 'lambda' in layer.name or \
        'tf.' in layer.name or 'activation' in layer.name or \
            'multiply' in layer.name:
        return layer
    return tfmot.quantization.keras.quantize_annotate_layer(layer)

def qa_train(model):
    # qa_train part
    quantize_model = tfmot.quantization.keras.quantize_model
    quantize_scope = tfmot.quantization.keras.quantize_scope
    
    annotated_model = tf.keras.models.clone_model(model,
        clone_function=apply_quantization,
    )

    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    return quant_aware_model

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.rmtree(d, ignore_errors=True)
            shutil.copy2(s, d)


def main(_argv):
    print('Saving in', str(FLAGS.save_dir) )
    init_shared_variable()

    models = FLAGS.models.split(',')
    tinys = np.array(FLAGS.tinys.split(',')) == "True" 
    number_models = len(models)

    filtered_full_HD_area = 123
    filtered_area = filtered_full_HD_area / (1920 / cfg.TRAIN.INPUT_SIZE) ** 2
    print('[Info] Filtered Instance Area', filtered_area)
    filtered_area = max(filtered_area, 49)
    trainDataLoader = tfDataset(FLAGS, cfg, is_training=True, filter_area=filtered_area, use_imgaug=False)
    trainDataLoader.reset_epoch()
    trainset = trainDataLoader.dataset_gen()
    testset = tfDataset(FLAGS, cfg, is_training=False, filter_area=filtered_area, use_imgaug=False).dataset_gen()

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.save_dir, 'config'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.save_dir, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.save_dir, 'pic'), exist_ok=True)

    copytree('./core', os.path.join(FLAGS.save_dir, 'config'))
    shutil.copy2(sys.argv[0], os.path.join(FLAGS.save_dir, 'config', os.path.basename(sys.argv[0])))
    with open(os.path.join(FLAGS.save_dir, 'command.txt'), 'w') as f:
        f.writelines(' '.join(sys.argv))
    f.close()

    logdir = os.path.join(FLAGS.save_dir, 'logs')
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_epochs = (first_stage_epochs + second_stage_epochs)
    total_steps = total_epochs * steps_per_epoch
    # Define all the models

    keras_models = []

    for i in range(number_models):
        
        input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
        # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(models[i], tinys[i])
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS, cfg)

        IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

        freeze_layers = utils.load_freeze_layer(models[i], tinys[i])

        feature_maps = YOLO(input_layer, NUM_CLASS, models[i], tinys[i], dc_head_type=0)
        if tinys[i]:
            print("---------------------- Student tiny", tinys[i])
            bbox_tensors = []
            for j, fm in enumerate(feature_maps):
                if j == 0:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, j, XYSCALE)
                else:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, j, XYSCALE)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
        else:
            print("---------------------- Student NOT tiny")
            bbox_tensors = []
            for j, fm in enumerate(feature_maps):
                if j == 0:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, j, XYSCALE)
                elif j == 1:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, j, XYSCALE)
                else:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, j, XYSCALE)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(input_layer, bbox_tensors)
        if i == 0 :
            if FLAGS.weights == None:
                print("Training from scratch ......................")
            else:
                if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
                    utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
                else:
                    model.load_weights(FLAGS.weights)
                print('Restoring weights from: %s ... ' % FLAGS.weights)
        #####################################################################################################
        if (FLAGS.qat):
            model = qa_train(model)
            print("Training in Quatization Aware ................. ")
        #####################################################################################################
    
        keras_models.append(model)

    # optimizers = [tf.keras.optimizers.Adam() for _ in range(len(tinys))]
    optimizer = tf.keras.optimizers.Adam()

    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)
    distillation_loss_fn=keras.losses.KLDivergence()
    alpha = 0.1,
    temperature = 10,

    # @tf.function
    def train_step(image_data, target, pred_results, current_model):
        # global keras_models
        # global optimizers
        model_idx = current_model
        # model = keras_models[model_idx]
        with tf.GradientTape() as tape:
            pred_result = keras_models[current_model](image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0
            ############################################################
            # Train Normal Object Detection Loss
            ############################################################
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, \
                    NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]
            det_loss = giou_loss + conf_loss + prob_loss
            ############################################################
            # Train KL Divergence Loss
            ############################################################
            kl_loss = 0
            for j in range(number_models):
                if model_idx != j:
                    for p in range(len(pred_results[model_idx])):
                        kl_loss += distillation_loss_fn(
                            tf.nn.log_softmax(pred_results[model_idx][p], axis=1),
                            tf.nn.softmax(pred_results[j][p], axis=1)
                        )
            ############################################################
            # Calculate Loss and Gradient
            ############################################################
            total_loss = det_loss + kl_loss/ (number_models - 1)
            # tf.print('total_loss', total_loss)
            # tf.print('current_model', current_model)
            # tf.print('keras_models', keras_models)
            gradients = tape.gradient(total_loss, keras_models[current_model].trainable_variables)
            # optimizers[model_idx].apply_gradients(zip(gradients, keras_models[current_model].trainable_variables))
            optimizer.apply_gradients(zip(gradients, keras_models[current_model].trainable_variables))

            # writing summary data
            with writer.as_default():
                # tf.summary.scalar("lr", optimizers[model_idx].lr, step=global_steps)
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
                tf.summary.scalar("loss/kl_loss", kl_loss, step=global_steps)
            writer.flush()
            return {
                'giou_loss': giou_loss, 
                'conf_loss': conf_loss, 
                'prob_loss': prob_loss,
                'kl_loss': kl_loss,
                'total_loss': total_loss
            }

    @tf.function        
    def test_step(image_data, target):
        pred_result = []
        for model in keras_models:
            pred_result.append(model(image_data, training=False))
        
        giou_loss = conf_loss = prob_loss = total_loss = 0
        for k in range(number_models):
            model = keras_models[k]
            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[k][i * 2], pred_result[k][i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, \
                    NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]
                tf.print('loss_items', loss_items)
        total_loss += (giou_loss + conf_loss + prob_loss)
        result = {
            'giou_loss': giou_loss, 
            'conf_loss': conf_loss, 
            'prob_loss': prob_loss,
            'total_loss': total_loss
        }
        return result

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
        giou_loss_counter = Accumulator()
        conf_loss_counter = Accumulator()
        prob_loss_counter = Accumulator()
        for iter_idx, data_item in tqdm(enumerate(trainset)):
            if iter_idx == 5: break
            image_data=data_item['images']
            target=[
                [data_item['label_bboxes_m'], data_item['bboxes_m']], 
                [data_item['label_bboxes_l'], data_item['bboxes_l']], 
            ]

            pred_results = []
            for model in keras_models:
                pred_results.append(model(image_data, training=True))
            for i in range(number_models):
                loss_dict = train_step(image_data, target, pred_results, i)

            # update learning rate
            global_steps.assign_add(1)
            for i in range(len(tinys)):
                if global_steps < warmup_steps:
                    lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
                else:
                    lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                        (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                    )
                # optimizers[i].lr.assign(lr.numpy())
                optimizer.lr.assign(lr.numpy())


            if global_steps%200==0 : 
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print("Current Time =", current_time)
                tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                    "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, optimizers[0].lr.numpy(),
                                                                   loss_dict['giou_loss'], loss_dict['conf_loss'], loss_dict['prob_loss'], loss_dict['total_loss']))


        if True:
            giou_loss_counter.reset()
            conf_loss_counter.reset()
            prob_loss_counter.reset()
            with tqdm(total=len(testset), ncols=150, desc=f"{'Test':<13}") as pbar:
                batch_size = image_data.shape[0]
                for data_item in testset:
                    image_data=data_item['images']
                    target=[
                        [data_item['label_bboxes_m'], data_item['bboxes_m']], 
                        [data_item['label_bboxes_l'], data_item['bboxes_l']], 
                    ]

                    batch_size = image_data.shape[0]
                    loss_dict = test_step(image_data, target)

                    giou_loss_counter.update(loss_dict['giou_loss'], batch_size)
                    conf_loss_counter.update(loss_dict['conf_loss'], batch_size)
                    prob_loss_counter.update(loss_dict['prob_loss'], batch_size)
                    total = giou_loss_counter.get_average() \
                        + conf_loss_counter.get_average() \
                        + prob_loss_counter.get_average()

                    pbar.set_postfix({
                        'giou_loss': f"{giou_loss_counter.get_average():6.4f}",
                        'conf_loss': f"{conf_loss_counter.get_average():6.4f}",
                        'prob_loss': f"{prob_loss_counter.get_average():6.4f}",
                        'total': f"{total: 6.4f}"
                    })
                    pbar.update(1)

        trainDataLoader.step_epoch()        
        # original 
        if (epoch + 1) % 5 == 0 or epoch > total_epochs - 5:
            for i in range(number_models):
                keras_models[i].save_weights(os.path.join(FLAGS.save_dir, f'model{i}_{epoch}.ckpt'))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass