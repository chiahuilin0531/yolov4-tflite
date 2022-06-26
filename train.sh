python ./train.py --tiny --save_dir ./checkpoints/test --weights ./data/yolov4-tiny.weights
python ./save_model.py --weights ./checkpoints/test/ckpt/final.ckpt --output ./checkpoints/test/save_model_final --tiny --input_size 608
python ./evaluate_map.py --weights ./checkpoints/test/save_model_final/ --framework tf --input_size 608 --annotation_path ./datasets/data_selection_mix/anno/val_1cls_filter_small.txt
python ./evaluate_map.py --weights ./checkpoints/test/save_model_final/ --framework tf --input_size 608 --annotation_path ./datasets/night_dataset/anno/val_well_split_1cls.txt




python ./evaluate_map_v3.py --weights ./checkpoints/test/save_model_final/ --framework tf --input_size 608 --annotation_path ./datasets/night_dataset/anno/val_3cls.txt --class_path ./data/classes/3cls.names
python ./convert_tflite.py --weights ./checkpoints_da/instance_level_alignment/save_model_final/ --output ./checkpoints_da/instance_level_alignment/tflite_final/ --input_size 608 --quantize_mode int8 --dataset ./datasets/day_night_mix/anno/train.txt

python ./convert_tflite.py --weights ./checkpoints/tw_coco_0616_qat/save_model_final/ --output ./checkpoints/int8_final.tflite --input_size 608 --quantize_mode int8 --dataset ./datasets/day_night_mix/anno/train.txt
python ./evaluate_map_v3.py --weights ./checkpoints/tw_coco_0616_qat/int8_final.tflite --framework tflite --input_size 608 --annotation_path ./datasets/night_dataset/anno/val_3cls.txt --class_path ./data/classes/3cls.names



=========================
conda deactivate; conda activate WJtf
python ./save_model.py --weights ./checkpoints/day_tw_qat_v2/ckpt/0058.ckpt --output ./checkpoints/day_tw_qat_v2/save_model_0058_tflite --tiny --qat --framework tflite
python ./convert_tflite.py  --output ./checkpoints/day_tw_qat_v2/tflite_0058/float32.tflite --quantize_mode float32  --dataset ./datasets/data_selection_mix/anno/val_3cls.txt --weights ./checkpoints/day_tw_qat_v2/save_model_0058_tflite
python ./convert_tflite.py  --output ./checkpoints/day_tw_qat_v2/tflite_0058/float16.tflite --quantize_mode float16  --dataset ./datasets/data_selection_mix/anno/val_3cls.txt --weights ./checkpoints/day_tw_qat_v2/save_model_0058_tflite
python ./convert_tflite.py  --output ./checkpoints/day_tw_qat_v2/tflite_0058/int8.tflite --quantize_mode int8  --dataset ./datasets/data_selection_mix/anno/val_3cls.txt --weights ./checkpoints/day_tw_qat_v2/save_model_0058_tflite

