python ./train.py --tiny --save_dir ./checkpoints/test --weights ./data/yolov4-tiny.weights
python ./save_model.py --weights ./checkpoints/test/ckpt/final.ckpt --output ./checkpoints/test/save_model_final --tiny --input_size 608
python ./evaluate_map.py --weights ./checkpoints/test/save_model_final/ --framework tf --input_size 608 --annotation_path ./datasets/data_selection_mix/anno/val_1cls_filter_small.txt
python ./evaluate_map.py --weights ./checkpoints/test/save_model_final/ --framework tf --input_size 608 --annotation_path ./datasets/night_dataset/anno/val_well_split_1cls.txt




python ./evaluate_map_v3.py --weights ./checkpoints/test/save_model_final/ --framework tf --input_size 608 --annotation_path ./datasets/night_dataset/anno/val_3cls.txt --class_path ./data/classes/3cls.names
python ./convert_tflite.py --weights ./checkpoints_da/instance_level_alignment/save_model_final/ --output ./checkpoints_da/instance_level_alignment/tflite_final/ --input_size 608 --quantize_mode int8 --dataset ./datasets/day_night_mix/anno/train.txt