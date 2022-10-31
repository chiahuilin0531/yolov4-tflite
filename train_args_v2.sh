#!/bin/bash
while getopts ":a:p:" opt; do
  case $opt in
    a) arg_1="$OPTARG";;
    p) p_out="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

printf "Argument p_out is %s\n" "$p_out"
printf "Argument arg_1 is %s\n" "$arg_1"

# environment=${environment:-production}
# school=${school:-is out}

# while [ $# -gt 0 ]; do

#    if [[ $1 == *"--"* ]]; then
#         param="${1/--/}"
#         declare $param="$2"
#         # echo $1 $2 // Optional to see the parameter:value result
#    fi

#   shift
# done

echo $environment $school