from super_gradients.training.utils.callbacks import LRCallbackBase, Phase


class PolyLR(LRCallbackBase):
    """
    A callback that performs polynomial learning rate scheduling.
    """

    def __init__(self, poly_exp, max_epochs, **kwargs):
        super().__init__(Phase.TRAIN_BATCH_STEP, **kwargs)
        self.max_epochs = max_epochs
        self.poly_exp = poly_exp

    def perform_scheduling(self, context):
        effective_epoch = context.epoch - self.training_params.lr_warmup_epochs
        effective_max_epochs = (
            self.max_epochs
            - self.training_params.lr_warmup_epochs
            - self.training_params.lr_cooldown_epochs
        )
        current_iter = (
            self.train_loader_len * effective_epoch + context.batch_idx
        ) / self.training_params.batch_accumulate
        max_iter = (
            self.train_loader_len
            * effective_max_epochs
            / self.training_params.batch_accumulate
        )
        self.lr = self.initial_lr * pow(
            (1.0 - (current_iter / max_iter)), self.poly_exp
        )
        self.update_lr(context.optimizer, context.epoch, context.batch_idx)

    def is_lr_scheduling_enabled(self, context):
        post_warmup_epochs = (
            self.training_params.max_epochs - self.training_params.lr_cooldown_epochs
        )
        return (
            self.training_params.lr_warmup_epochs <= context.epoch < post_warmup_epochs
        )
