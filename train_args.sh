#!/bin/bash

args=$(getopt -u -l "save_dir:,weights:,cnnpp_weights:,tiny:,qat:,iayolo:" --options "" -- "$@")

echo "$args"
eval set -- "$args"

################################
# initialize arguments
################################
# specify arguments
save_dir=""
weights=""
cnnpp_weights=""
# on-off arguments 
tiny=""
qat=""
iayolo=""


while [ $# -ge 1 ]; do
        echo  "$2"
        case "$1" in
                # --)
                #     # No more options left.
                #     shift
                #     break
                #    ;;
                --save_dir)
                    save_dir="$2"
                    shift
                    ;;
                --weights)
                    weights="$2"
                    shift
                    ;;
                --cnnpp_weights)
                    cnnpp_weights="$2"
                    shift
                    ;;
                --tiny)
                    if [ "$2" = "True" ] || [ "$2" = "true" ]
                    then
                            tiny="--tiny"
                    elif [ "$2" = "False" ] || [ "$2" = "false" ]
                    then
                            tiny=""
                    else
                            echo "Satisfactory";
                    fi
                    shift
                    ;;
                --qat)  
                    if [ "$2" = "True" ] || [ "$2" = "true" ]
                    then
                            qat="--qat"
                    elif [ "$2" = "False" ] || [ "$2" = "false" ]
                    then
                            qat=""
                    else
                            echo "Satisfactory";
                    fi
                    shift
                    ;;
                --iayolo)  
                    if [ "$2" = "True" ] || [ "$2" = "true" ]
                    then
                            iayolo="--iayolo"
                    elif [ "$2" = "False" ] || [ "$2" = "false" ]
                    then
                            iayolo=""
                    else
                            echo "Satisfactory";
                    fi
                    shift
                    ;;
                -h)
                    echo "Display some help"
                    exit 0
                    ;;
        esac

        shift
done

# echo "save_dir: $save_dir"
# echo "weights: $weights"
# echo "tiny: $tiny"
# echo "qat: $qat"
# echo "iayolo: $iayolo"
# echo "cnnpp_weights: $cnnpp_weights"
# echo "remaining args: $*"
ckpt_dir="$save_dir/ckpt/final.ckpt"
save_model_dir="$save_dir/save_model_final/"

# echo "save_model_dir: $save_model_dir"
# echo "ckpt_dir: $ckpt_dir"
echo ./train.py  --save_dir "$save_dir" --weights "$weights"   " $iayolo"  " $tiny" " $qat"  
python ./train.py  --save_dir "$save_dir" --weights "$weights"   " $iayolo"  " $tiny" " $qat"  

echo ./save_model.py " $tiny" " $qat"  " $iayolo" --weights "$ckpt_dir" --output "$save_model_dir"
python ./save_model.py " $tiny" " $qat"  " $iayolo" --weights "$ckpt_dir" --output "$save_model_dir"

echo ./evaluate_map_v3.py --weights "$save_model_dir" --show_ia
python ./evaluate_map_v3.py --weights "$save_model_dir" --show_ia