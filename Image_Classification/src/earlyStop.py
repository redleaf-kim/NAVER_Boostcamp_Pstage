import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, type_="min"):
        self.type_ = type_
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.target_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, target):
        if self.type_ == "min": score = -target
        else: score = target

        if self.best_score is None:
            self.best_score = score
            return False
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
            return False
        else:
            self.best_score = score
            self.counter = 0
            return False