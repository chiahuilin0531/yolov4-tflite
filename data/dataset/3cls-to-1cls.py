import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', '--in', default=None,type=str, help="labeln_dir")
parser.add_argument('--out_path', '--out', default=None,type=str, help="labeln_dir")

args = parser.parse_args()

f1 = open(args.in_path,'r')
f2 = open(args.out_path,'w')

for line in f1:
    line = line.replace('\n','')
    s = line.split(' ')
    new_line = s[0]  #image path

    boxes = s[1:]
    for box in boxes:
        new_box = box[0:-1]+'0'
        new_line += (' '+new_box)
    
    # print(new_line)
    f2.write(new_line+'\n')