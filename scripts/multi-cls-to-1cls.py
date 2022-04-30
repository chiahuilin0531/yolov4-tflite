import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--anno', help='annotation file path')
parser.add_argument('--output', help='output file path')
args = parser.parse_args()


if __name__=='__main__':
    with open(args.anno, 'r') as f:
        lines=f.readlines()
    lines = [line.split() for line in lines]


    with open(args.output, 'w') as f:
        for line in lines:
            write_line=line[0]+' '
            for bbox in line[1:]:
                x1y1x2y2cls = bbox.split(',')
                x1y1x2y2cls[-1] = '0'
                write_line += ','.join(x1y1x2y2cls) + ' '
            write_line += '\n'
            f.write(write_line)
