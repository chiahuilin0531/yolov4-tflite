export CUDA_VISIBLE_DEVICES=7
python ./train.py --tiny --qat --save_dir ./checkpoints/0719_test2
python ./save_model.py --tiny --qat --weights ./checkpoints/0719_test2/ckpt/final.ckpt --output ./checkpoints/0719_test2/save_model_final_tflite --framework tflite
python ./quantize_model.py
python ./evaluate_map_v3.py --weights ./tflite_exp_v2/selective_int8_model_33_layer_st88_end121.tflite --framework tflite