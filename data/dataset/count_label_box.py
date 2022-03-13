import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--label_path', '--path', default=None,type=str, help="labeln_dir")
args = parser.parse_args()

f = open(args.label_path,'r')

cnt_g = 0
cnt_r = 0
cnt_y = 0
# cnt_u = 0

for line in f:

    line = line.replace('\n','')
    boxes = line.split(' ')[1:]
    for box in boxes:
        cls = box[-1]
        if cls == '0':cnt_g+=1
        elif cls == '1':cnt_r+=1
        elif cls == '2':cnt_y+=1

print('File :', args.label_path)
print('Green :',cnt_g)
print('Red :',cnt_r)
print('Yellow :',cnt_y)