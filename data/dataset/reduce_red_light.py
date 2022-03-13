import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--label_path', '--path', default=None,type=str, help="labeln_dir")
args = parser.parse_args()

f = open(args.label_path,'r')
f1 = open("train_new_countdown_3cls.txt",'w')

for line in f:
    line = line.replace('\n','')
    s = line.split(' ')
    boxes = s[1:]

    flag = False
    for box in boxes:
        if box[-1]=='0' or box[-1]=='2':
            flag=True
    
    if flag==True:
        f1.write(line+'\n')
