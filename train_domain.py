import sys
import cv2
import os
import numpy as np
import random
import tensorflow as tf
global_random_seed = random.randint(0, 65535) # 10
operand_random_seed = random.randint(0, 65535) # 100
tf.keras.utils.set_random_seed(global_random_seed)
# # tf.config.experimental.enable_op_determinism()
# tf.random.set_seed(global_random_seed)
# np.random.seed(global_random_seed)
# random.seed(global_random_seed)

from absl import app, flags, logging
from absl.flags import FLAGS
from core.accumulator import Accumulator, AreaCounter
import os, shutil
from core.yolov4 import YOLO, compute_loss, compute_domain_loss, decode_train
from core.iayolo import CNNPP, DIP_FilterGraph

# from core.dataset_tiny import Dataset, tfDataset
# from core.config import cfg

from core.dataset_tiny import tfAdversalDataset as tfDataset
from core.config_adverserial import cfg


from core import utils
from core.utils import draw_bbox, freeze_all, unfreeze_all, read_class_names, get_shared_variable, init_shared_variable
from tqdm import tqdm
import tensorflow_model_optimization as tfmot
import tensorflow_addons as tfa
import time

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_string('weights', './data/yolov4-tiny.weights', 'pretrained weights')
flags.DEFINE_string('cnnpp_weights', None, 'pretrained dip weights')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_boolean('qat', False, 'train w/ or w/o quatize aware')
flags.DEFINE_string('save_dir', 'checkpoints/yolov4_tiny', 'save model dir')
flags.DEFINE_float('repeat_times', 1.0, 'repeat of dataset')
flags.DEFINE_boolean('iayolo', False, 'use IAYOLO or not')
flags.DEFINE_boolean('da', True, 'use Domain Adaptation or not')
flags.DEFINE_boolean('instance_level', False, 'use instance_level Domain Adaptation or not')


tf.config.optimizer.set_jit(True)

def apply_quantization(layer):
    # if 'relu' in layer.name:
    #     return tfmot.quantization.keras.quantize_annotate_layer(layer)
    # if isinstance(layer, tf.python.keras.engine.base_layer.TensorFlowOpLayer):
    ####3
    # if 'tf_op' in layer.name or 'lambda' in layer.name or \
    #     'tf.' in layer.name or 'activation' in layer.name or \
    #         'multiply' in layer.name:
    if 'tf_op' in layer.name or 'lambda' in layer.name or \
        'tf.' in layer.name or isinstance(layer, tfa.layers.InstanceNormalization) or \
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
    assert(FLAGS.da)
    print(f'Global Seed: {global_random_seed} Operation Seed: {operand_random_seed}')
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    init_shared_variable()
    # trainset = Dataset(FLAGS, is_training=True, filter_area=123)
    # testset = Dataset(FLAGS, is_training=False, filter_area=123)
    #########################################################
    # use_imgaug augmentation would lead to unknown performance drop
    # this issue should be resolved in the future.
    filtered_full_HD_area = 123
    filtered_area = filtered_full_HD_area / (1920 / cfg.TRAIN.INPUT_SIZE) ** 2
    print('[Info] Filtered Instance Area', filtered_area)
    # filtered_area = max(filtered_area, 12)
    trainDataLoader = tfDataset(FLAGS, cfg, is_training=True, filter_area=filtered_area, use_imgaug=False, operand_seed=operand_random_seed)
    trainDataLoader.reset_epoch()
    trainset = trainDataLoader.dataset_gen(repeat_times=int(FLAGS.repeat_times))
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

    info_file = open(os.path.join(FLAGS.save_dir, 'info.txt'), 'w')
    info_file.writelines(f'Global Seed: {global_random_seed} Operation Seed: {operand_random_seed}\n') 
    info_file.flush()    

    logdir = os.path.join(FLAGS.save_dir, 'logs')
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps, global_epochs = get_shared_variable()
    global_steps.assign(1)
    global_epochs.assign(1)

    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    if FLAGS.iayolo:
        resized_input = tf.image.resize(input_layer, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
        filter_parameters = CNNPP(resized_input)
        yolo_input, processed_list = DIP_FilterGraph(input_layer, filter_parameters)
    else:
        yolo_input = input_layer
        

    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS, cfg)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
    classes_name = read_class_names(cfg.YOLO.CLASSES)

    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)

    output_maps = YOLO(yolo_input, NUM_CLASS, FLAGS.model, FLAGS.tiny, nl=cfg.YOLO.NORMALIZATION, dc_head_type=1)
    feature_maps = output_maps[:2]
    da_maps = output_maps[2:]
    
    # Decoding YOLOv4 Output
    if FLAGS.tiny:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
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
    output_dict = {
        'raw_bbox_m': bbox_tensors[0],          # tensor size of feature map
        'bbox_m': bbox_tensors[1],
        'da_m': da_maps[0],
        'raw_bbox_l': bbox_tensors[2],
        'bbox_l': bbox_tensors[3],
        'da_l': da_maps[1],
    }
    if FLAGS.iayolo: 
        output_dict.update({
            'dip_img':              yolo_input,
            'filter_parameters':    filter_parameters,
            'ImprovedWhiteBalance': processed_list[0], 
            'Gamma':                processed_list[1], 
            'Tone':                 processed_list[2], 
            'Contrast':             processed_list[3], 
            'Usm':                  processed_list[4],
        })
    model = tf.keras.Model(input_layer, output_dict)
        
    if FLAGS.cnnpp_weights == None:
        print("Training from scratch ......................")
    else:
        model.load_weights(FLAGS.cnnpp_weights)
        print('Restoring CNNPP weights from: %s ... ' % FLAGS.weights)
        
    if FLAGS.weights == None:
        print("Training from scratch ......................")
    else:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny, load_batch_weight=cfg.YOLO.NORMALIZATION=='BatchNorm')
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)
    
    #####################################################################################################
    if (FLAGS.qat):
        model = qa_train(model)
        print("Training in Quatization Aware ................. ")
    #####################################################################################################
    
    model.summary()

    optimizer = tf.keras.optimizers.Adam(clipvalue=10.0)
    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    @tf.function
    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            dict_result = model(image_data, training=True)
            pred_result = [
                dict_result['raw_bbox_m'], 
                dict_result['bbox_m'], 
                dict_result['raw_bbox_l'],
                dict_result['bbox_l'],
            ]
            
            da_reuslt = [
                dict_result['da_m'],
                dict_result['da_l'],
            ]
            if FLAGS.instance_level:
                mask_da_m = tf.expand_dims(tf.cast(tf.reduce_any(target[0][0] > 0, axis=(3,4)), dtype=tf.float32), axis=-1)
                mask_da_l = tf.expand_dims(tf.cast(tf.reduce_any(target[1][0] > 0, axis=(3,4)), dtype=tf.float32), axis=-1)
            else:
                mask = tf.cast(tf.reduce_any(image_data != tfDataset.get_padding_val(), axis=-1, keepdims=True), dtype=tf.float32)
                mask_da_m = tf.image.resize(mask, tf.shape(target[0][0])[1:3])
                mask_da_l = tf.image.resize(mask, tf.shape(target[1][0])[1:3])
            maskes=[
                mask_da_m, 
                mask_da_l
            ]
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layers)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, 
                NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i, IOU_LOSS=utils.bbox_diou)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]
            da_loss = compute_domain_loss(da_reuslt, target[2], mask=maskes)
            
            da_loss = cfg.TRAIN.ADVERSERIAL_CONST * da_loss
            total_loss = giou_loss + conf_loss + prob_loss + da_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            if FLAGS.iayolo:
                for idx, (g, v) in enumerate(zip(gradients, model.trainable_variables)):
                    gradients[idx] = tf.clip_by_norm(g, 5)
                    if  'ex_conv6' in v.name.split(':')[0]: break
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
                tf.summary.scalar("loss/da_loss", da_loss, step=global_steps)
                
                if FLAGS.iayolo:
                    iayolo_value = tf.reduce_mean(dict_result['filter_parameters'], axis=0)
                    tf.summary.scalar("ia-yolo/1", iayolo_value[0], step=global_steps)
                    tf.summary.scalar("ia-yolo/2", iayolo_value[1], step=global_steps)
                    tf.summary.scalar("ia-yolo/3", iayolo_value[2], step=global_steps)
                    tf.summary.scalar("ia-yolo/4", iayolo_value[3], step=global_steps)
                    tf.summary.scalar("ia-yolo/5", iayolo_value[4], step=global_steps)
                    tf.summary.scalar("ia-yolo/6", iayolo_value[5], step=global_steps)
                    tf.summary.scalar("ia-yolo/7", iayolo_value[6], step=global_steps)
                    tf.summary.scalar("ia-yolo/8", iayolo_value[7], step=global_steps)
                    tf.summary.scalar("ia-yolo/9", iayolo_value[8], step=global_steps)
                    tf.summary.scalar("ia-yolo/10", iayolo_value[9], step=global_steps)
                    tf.summary.scalar("ia-yolo/11", iayolo_value[10], step=global_steps)
                    tf.summary.scalar("ia-yolo/12", iayolo_value[11], step=global_steps)
                    tf.summary.scalar("ia-yolo/13", iayolo_value[12], step=global_steps)
                    tf.summary.scalar("ia-yolo/14", iayolo_value[13], step=global_steps)
                      
            writer.flush()
            return {
                'giou_loss': giou_loss, 
                'conf_loss': conf_loss, 
                'prob_loss': prob_loss,
                'da_loss': da_loss
            }

    @tf.function
    def test_step(image_data, target):
        dict_result = model(image_data, training=False)
        pred_result = [
            dict_result['raw_bbox_m'], 
            dict_result['bbox_m'], 
            dict_result['raw_bbox_l'],
            dict_result['bbox_l'],
        ]
        
        da_reuslt = [
            dict_result['da_m'],
            dict_result['da_l'],
        ]
        if FLAGS.instance_level:
            mask_da_m = tf.expand_dims(tf.cast(tf.reduce_any(target[0][0] > 0, axis=(3,4)), dtype=tf.float32), axis=-1)
            mask_da_l = tf.expand_dims(tf.cast(tf.reduce_any(target[1][0] > 0, axis=(3,4)), dtype=tf.float32), axis=-1)
        else:
            mask = tf.cast(tf.reduce_any(image_data != tfDataset.get_padding_val(), axis=-1, keepdims=True), dtype=tf.float32)
            mask_da_m = tf.image.resize(mask, tf.shape(target[0][0])[1:3])
            mask_da_l = tf.image.resize(mask, tf.shape(target[1][0])[1:3])
        maskes=[
            mask_da_m, 
            mask_da_l
        ]
            
        giou_loss = conf_loss = prob_loss = 0
        # optimizing process
        for i in range(len(freeze_layers)):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, 
            NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]
        da_loss = compute_domain_loss(da_reuslt, target[2], mask=maskes)
            
        da_loss = cfg.TRAIN.ADVERSERIAL_CONST * da_loss
        total_loss = giou_loss + conf_loss + prob_loss

        return {
            'giou_loss': giou_loss, 
            'conf_loss': conf_loss, 
            'prob_loss': prob_loss,
            'da_loss': da_loss,
            'total_loss': total_loss
        }

    model.save_weights(os.path.join(FLAGS.save_dir, 'ckpt', f'{0:04d}.ckpt'))
    best_loss = np.inf
    ranges=[123, 195, 267, 341, 489, 637, 786, 1e5]
    bboxes_counter = AreaCounter(ranges)
    total_cnt = np.copy(bboxes_counter.get_cnt())
    info_file.writelines(bboxes_counter.get_title())
    for epoch in range(1, 1+first_stage_epochs + second_stage_epochs):
        if epoch <= first_stage_epochs:
            if not isfreeze:
                isfreeze = True
                for name in freeze_layers:
                    if FLAGS.qat==True:
                        freeze = model.get_layer('quant_'+name)
                    else:
                        freeze = model.get_layer(name)
                    freeze_all(freeze)
        elif epoch > first_stage_epochs:
            if isfreeze:
                isfreeze = False
                for name in freeze_layers:
                    if FLAGS.qat==True:
                        freeze = model.get_layer('quant_'+name)
                    else:
                        freeze = model.get_layer(name)
                    unfreeze_all(freeze)
        
        giou_loss_counter = Accumulator()
        conf_loss_counter = Accumulator()
        prob_loss_counter = Accumulator()
        da_loss_counter = Accumulator()
        
        tmp=time.time()
        with tqdm(total=len(trainset), ncols=200) as pbar:
            for iter_idx, data_item in enumerate(trainset):
                image_data=data_item['images']
                target=[
                    [data_item['label_bboxes_m'], data_item['bboxes_m']], 
                    [data_item['label_bboxes_l'], data_item['bboxes_l']], 
                    data_item['domain']
                ]

                data_time=time.time()-tmp
                batch_size = image_data.shape[0]
                tmp=time.time()
                loss_dict = train_step(image_data, target)
                model_time = time.time()-tmp

                giou_loss_counter.update(loss_dict['giou_loss'], batch_size)
                conf_loss_counter.update(loss_dict['conf_loss'], batch_size)
                prob_loss_counter.update(loss_dict['prob_loss'], batch_size)
                da_loss_counter.update(loss_dict['da_loss'], batch_size)

                # update learning rate
                global_steps.assign_add(1)
                if global_steps < warmup_steps:
                    lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
                else:
                    lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                        (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
                optimizer.lr.assign(lr.numpy())

                if (iter_idx + 1) % 200 == 0:
                    #############################
                    # Visualize Detection Part
                    #############################
                    dict_result = model(image_data[:4], training=False)
                    
                    if FLAGS.instance_level:
                        mask_da_m = tf.expand_dims(tf.cast(tf.reduce_any(target[0][0] > 0, axis=(3,4)), dtype=tf.float32), axis=-1)
                        mask_da_l = tf.expand_dims(tf.cast(tf.reduce_any(target[1][0] > 0, axis=(3,4)), dtype=tf.float32), axis=-1)
                    else:
                        mask = tf.cast(tf.reduce_any(image_data != tfDataset.get_padding_val(), axis=-1, keepdims=True), dtype=tf.float32)
                        mask_da_m = tf.image.resize(mask, tf.shape(target[0][0])[1:3])
                        mask_da_l = tf.image.resize(mask, tf.shape(target[1][0])[1:3])
                    maskes=[ mask_da_m, mask_da_l]
                    
                    if FLAGS.iayolo:  display_images = dict_result['dip_img']
                    else: display_images = image_data[:4]
                    processed_bboxes = utils.raw_output_to_bbox([
                        dict_result['bbox_m'],
                        dict_result['bbox_l'],
                    ], tf.shape(image_data).numpy()[1:3])
                    
                    display_images = (display_images*255.0).numpy().astype(np.uint8)
                    display_list = []

                    for i in range(4):
                        image = draw_bbox(display_images[i], [
                            item[i:i+1] for item in processed_bboxes
                        ], classes_name)
                        display_list.append(image)
                    top_img = np.concatenate(display_list[:2], axis=1)
                    bot_img = np.concatenate(display_list[2:], axis=1)
                    full_img = np.concatenate([top_img, bot_img], axis=0)
                    # Create Display Mask m
                    mask_m_bool = tf.cast(maskes[0], tf.uint8).numpy() * 255
                    top_mask_m = np.concatenate(mask_m_bool[:2], axis=1) 
                    bot_mask_m = np.concatenate(mask_m_bool[2:4], axis=1) 
                    full_mask_m = np.concatenate([top_mask_m, bot_mask_m], axis=0).repeat(3, axis=-1)
                    # Create Display Mask l
                    mask_l_bool = tf.cast(maskes[1], tf.uint8).numpy() * 255
                    top_mask_l = np.concatenate(mask_l_bool[:2], axis=1) 
                    bot_mask_l = np.concatenate(mask_l_bool[2:4], axis=1) 
                    full_mask_l = np.concatenate([top_mask_l, bot_mask_l], axis=0).repeat(3, axis=-1)
                    

                    cv2.imwrite(os.path.join(FLAGS.save_dir, 'pic', f'{epoch}_{iter_idx+1}_img.jpg'), full_img[..., ::-1])
                    cv2.imwrite(os.path.join(FLAGS.save_dir, 'pic', f'{epoch}_{iter_idx+1}_msk_m.jpg'), full_mask_m)
                    cv2.imwrite(os.path.join(FLAGS.save_dir, 'pic', f'{epoch}_{iter_idx+1}_msk_l.jpg'), full_mask_l)
                    #############################
                    # Visualize DIP Part
                    #############################
                    if FLAGS.iayolo:
                        processed_key_words = ['ImprovedWhiteBalance', 'Gamma', 'Tone', 'Contrast', 'Usm']
                        for batch_idx in range(2):
                            processed_images = [image_data[batch_idx]]
                            for key in processed_key_words:
                                processed_images.append(dict_result[key][batch_idx])
                            top_img = np.concatenate(processed_images[:3], axis=1)
                            bot_img = np.concatenate(processed_images[3:], axis=1)
                            full_img = np.concatenate([top_img, bot_img], axis=0)
                            full_img = np.clip(full_img*255.0, 0, 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(FLAGS.save_dir, 'pic', f'{epoch}_{iter_idx+1}_dip{batch_idx}.jpg'), full_img[..., ::-1])
                    
                total = giou_loss_counter.get_average() \
                    + conf_loss_counter.get_average() \
                    + prob_loss_counter.get_average() \
                    + da_loss_counter.get_average()

                stats_bboxes = tf.concat([data_item['bboxes_m'], data_item['bboxes_l']], axis=0)
                stats_bboxes = stats_bboxes.numpy()
                bboxes_counter.update(stats_bboxes) 
                
                pbar.set_postfix({
                    'lr': f"{lr.numpy():6.4f}",
                    'giou_loss': f"{giou_loss_counter.get_average():6.4f}",
                    'conf_loss': f"{conf_loss_counter.get_average():6.4f}",
                    'prob_loss': f"{prob_loss_counter.get_average():6.4f}",
                    'da_loss': f"{prob_loss_counter.get_average():6.4f}",
                    'total': f"{total: 6.4f}",
                    # 'data_time': f'{data_time:8.6f}',
                    # 'model_time': f'{model_time:8.6f}'
                })
                total_epoch = first_stage_epochs + second_stage_epochs
                pbar.set_description(f"Epoch {epoch:3d}/{total_epoch:3d}")
                pbar.update(1)

                tmp=time.time()
        trainDataLoader.step_epoch()
        
        info_file.writelines(bboxes_counter.get_info())
        info_file.flush()

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
                        data_item['domain']
                    ]

                    batch_size = image_data.shape[0]
                    loss_dict = test_step(image_data, target)

                    giou_loss_counter.update(loss_dict['giou_loss'], batch_size)
                    conf_loss_counter.update(loss_dict['conf_loss'], batch_size)
                    prob_loss_counter.update(loss_dict['prob_loss'], batch_size)
                    da_loss_counter.update(loss_dict['da_loss'], batch_size)
                    
                    total = giou_loss_counter.get_average() \
                        + conf_loss_counter.get_average() \
                        + prob_loss_counter.get_average() \
                        + da_loss_counter.get_average()

                    pbar.set_postfix({
                        'giou_loss': f"{giou_loss_counter.get_average():6.4f}",
                        'conf_loss': f"{conf_loss_counter.get_average():6.4f}",
                        'prob_loss': f"{prob_loss_counter.get_average():6.4f}",
                        'da_loss': f"{da_loss_counter.get_average():6.4f}",
                        'total': f"{total: 6.4f}"
                    })
                    pbar.update(1)
            
            # writing summary data
            with writer.as_default():
                tf.summary.scalar("val/total_loss", total, step=epoch)
                tf.summary.scalar("val/giou_loss", giou_loss_counter.get_average(), step=epoch)
                tf.summary.scalar("val/conf_loss", conf_loss_counter.get_average(), step=epoch)
                tf.summary.scalar("val/prob_loss", prob_loss_counter.get_average(), step=epoch)
                tf.summary.scalar("val/da_loss", da_loss_counter.get_average(), step=epoch)
                
        if total < best_loss:
            best_loss = total
            print('[Info] Save Best Model')
            model.save_weights(os.path.join(FLAGS.save_dir, 'ckpt', f'best.ckpt'))
        if epoch % 5 == 0 or (1+first_stage_epochs + second_stage_epochs- epoch < 10):
            model.save_weights(os.path.join(FLAGS.save_dir, 'ckpt', f'{epoch:04d}.ckpt'))
        total_cnt += bboxes_counter.get_cnt()
        bboxes_counter.reset()
    
    bboxes_counter.cnt = total_cnt
    info_file.writelines(bboxes_counter.get_info(with_title=True))
    model.save_weights(os.path.join(FLAGS.save_dir, 'ckpt', 'final.ckpt'))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
