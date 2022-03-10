import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--label_path', '--path', default=None,type=str, help="labeln_dir")
parser.add_argument('--out_path', '--output', default=None,type=str, help="labeln_dir")

args = parser.parse_args()

with open(args.label_path) as f:
    lines=f.readlines()
    lines.sort()
    with open(args.out_path,'w') as f1:
        for i in lines:
            f1.write(i)