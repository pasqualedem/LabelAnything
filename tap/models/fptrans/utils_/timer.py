import time


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.elapsed = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.spc = 0.           # Seconds per call
        self.cps = 0.           # Calls per second

        self.total_time = 0.    # Not affected by self.reset()
        self.total_calls = 0    # Not affected by self.reset()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.elapsed += self.diff
        self.total_time += self.diff
        self.calls += 1
        self.total_calls += 1
        self.spc = self.elapsed / self.calls
        self.cps = self.calls / self.elapsed
        return self.diff

    def reset(self):
        self.elapsed = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.spc = 0.
        self.cps = 0.

    def start(self):
        return self

    def __enter__(self):
        self.tic()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc()
