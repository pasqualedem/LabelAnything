
class PolyLR:
    def __init__(self, power):
        self.poly_exp = power

    def perform_scheduling(self, initial_lr, epoch, max_epoch, **kwargs):
        """
        Perform polyLR learning rate scheduling
        """
        return initial_lr * pow((1.0 - epoch / max_epoch), self.poly_exp)


class ExpLR:
    def __init__(self, gamma, last_epoch_decay=None, min_lr=None):
        self.gamma = gamma
        self.last_epoch_decay = last_epoch_decay
        self.min_lr = min_lr

    def perform_scheduling(self, initial_lr, epoch, **kwargs):
        """
        Perform polyLR learning rate scheduling
        """
        if (
            self.min_lr is not None
            and initial_lr * pow(self.gamma, epoch) < self.min_lr
        ):
            return self.min_lr
        if self.last_epoch_decay is not None and epoch >= self.last_epoch_decay:
            return initial_lr * pow(self.gamma, self.last_epoch_decay)
        return initial_lr * pow(self.gamma, epoch)
