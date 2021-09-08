class Counter():
    def __init__(self, patience=5, descending=False):
        self.patience = patience
        self.descending = descending
        self.reset()

    def step(self, x):
        if (self.descending and x <= self._best) or (not self.descending and x >= self._best):
            self._best = x
            self._counter = 0
            return False
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.reset()
                return True

    def reset(self):
        self._counter = 0
        self._best = 0
