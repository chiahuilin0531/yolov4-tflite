
import numpy as np

class Accumulator:
    def __init__(self):
        self.cnt = 0
        self.value = 0

    def update(self, data, num=None):
        self.value += data
        if num is None:
            self.cnt += 1
        else:
            self.cnt += num

    def get_average(self):
        return self.value / self.cnt

    def reset(self):
        self.value = 0
        self.cnt = 0
    
    
class AreaCounter:
    def __init__(self, ranges=[123, 195, 267, 341, 489, 637, 786, 1e5], downsample_size=(1920 / 608) ** 2):
        """
        ranges: range in FHD
        downsample_size: bbox in downsample_size (area)
        """
        self.cnt = np.zeros((len(ranges) - 1,))
        self.downsample_size = downsample_size
        self.ranges = ranges
        self.data = np.zeros((0,), dtype=np.int32)
        
        # (1, k)
        self.less    = np.array(ranges[:-1]).reshape(1, -1) / downsample_size
        self.greater = np.array(ranges[ 1:]).reshape(1, -1) / downsample_size

    def update(self, data):
        """
        data: (b, n, 4), in x1y1x2y2
        """
        area = (data[..., 3] * data[..., 2])
        area = np.reshape(area, (-1, 1))

        # (n, 1)
        area = area[area > 1]
        area = area.reshape((-1,1))
        # (n, k)
        less_bool    = area >= self.less
        greater_bool = area < self.greater
        
        in_range = np.logical_and(less_bool, greater_bool)
        not_in_range = np.all(~in_range, axis=1)
        # (n, k)
        cnt = np.sum(in_range.astype(np.int32), axis=0)
        unknown_cnt = np.sum(not_in_range)
        
        assert(cnt.shape == self.cnt.shape)
        self.cnt += cnt
        
        # self.data = np.concatenate([self.data, area.reshape((-1,))], axis=0)
        

    def get_cnt(self):
        return np.copy(self.cnt)
    
    def get_data(self):
        return self.data

    def reset(self):
        self.data = np.zeros((0,), dtype=np.int32)
        self.cnt = self.cnt * 0
        
        
    def get_info(self, with_title=False):
        if with_title:
            info_string = self.get_title()
        else:
            info_string = ''
            
        info_string += f'| {"count".rjust(10, " ")} '
        total_cnt = np.sum(self.cnt)
        for c in self.cnt:
            # info_string += f"| {int(c):17d} "
            info_string += f"| {int(c):10d}({c/total_cnt*100.0:5.2f}) "
        info_string += f'| {int(total_cnt):10d} |\n'
        
        return info_string
    
    def get_title(self):
        return_string = f'| {"FHD BBoxes".rjust(10, " ")} '
        for h, l in zip(self.ranges[:-1], self.ranges[1:]):
            return_string += f"| {int(h):8d}~{int(l):8d} "
        return_string += f'| {"Total".rjust(10, " ")} |\n'
        
        return_string += f'| {"Img BBoxes".rjust(10, " ")} '
        for h, l in zip(self.ranges[:-1], self.ranges[1:]):
            h = h / self.downsample_size
            l = l/ self.downsample_size
            return_string += f"| {int(h):8d}~{int(l):8d} "
        return_string += f'| {" " * 10} |\n'
        
        return return_string