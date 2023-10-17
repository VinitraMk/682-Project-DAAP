class Accumulator():
    def __init__(self):
        self.sum = 0
        self.num = 0

    def update(self, val):
        self.sum += val
        self.num += 1

    def get_avg(self):
        if self.num == 0:
            return 0
        else:
            return self.sum / self.num
    
    def __str__(self):
        return "{:.2f}%".format(100*self.get_avg())