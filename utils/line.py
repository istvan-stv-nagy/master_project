class Line:
    def __init__(self, x1, y1, x2, y2):
        self.min_x = min(x1, x2)
        self.max_x = max(x1, x2)
        self.min_y = min(y1, y2)
        self.max_y = max(y1, y2)
        self.m = (y2 - y1) / (x2 - x1)
        self.n = y1 - self.m * x1

    def contains(self, x, y):
        return (self.min_x <= x <= self.max_x) and (self.min_y <= y <= self.max_y)

    def get_x(self, y):
        return int((y - self.n) / self.m)