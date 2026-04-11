import os


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def get_path(*paths):
    full_path = os.path.join(*paths)
    dir_path = os.path.dirname(full_path)
    os.makedirs(dir_path, exist_ok=True)
    return full_path


def makedirs(directory):
    os.makedirs(os.path.dirname(directory), exist_ok=True)
