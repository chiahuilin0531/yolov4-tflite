
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