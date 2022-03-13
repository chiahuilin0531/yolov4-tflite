import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', '--in', default=None,type=str, help="labeln_dir")
parser.add_argument('--out_path', '--out', default=None,type=str, help="labeln_dir")

args = parser.parse_args()

f1 = open(args.in_path,'r')
f2 = open(args.out_path,'w')
# all = 0
for line in f1:

    line = line.replace('\n','')
    s = line.split(' ')
    cnt = 0
    new_line = s[0]  #image path

    boxes = s[1:]
    for box in boxes:
        points = box.split(',')
        x1 = int(points[0])
        y1 = int(points[1])
        x2 = int(points[2])
        y2 = int(points[3])
        area = (x2-x1)*(y2-y1)
        # print(area, end='   ')

        if area > 123:
            new_line += (' '+box)
            cnt += 1
        # if area < 123:
            # all += 1
    # if cnt>0 :print(new_line)
    if cnt>0 :f2.write(new_line+'\n')
# print(all)