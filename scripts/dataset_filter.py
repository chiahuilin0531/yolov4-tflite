import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--anno', help="annotation path")
parser.add_argument('--output', help="output file")
args = parser.parse_args()


if __name__=='__main__':
    lines = open(args.anno).readlines()
    large_instance_image_list=[]
    for line in lines:
        line_split = line.split()
        filename = line_split[0]
        bboxes = line_split[1:]
        max_area = 0
        info = None
        for bbox in bboxes:
            x1,y1,x2,y2,_ = map(int, bbox.split(','))
            w = (x2-x1)
            h = (y2-y1)
            area = w * h
            if area > max_area and w > h: 
                max_area = area
                info = bbox
        large_instance_image_list.append((filename, max_area, bbox))
    dtype = [('name', 'S300'), ('area', float), ('info','S100')]

    area_list = np.array(large_instance_image_list, dtype=dtype)
    area_list = np.sort(area_list, order='area')[::-1]

    with open(args.output, 'w') as f:
        for i in range(500):
            filename = area_list[i][0]
            area = area_list[i][1]
            f.writelines(f'{filename.decode("utf-8")} {area} {area_list[i][2]}\n')
    f.close()