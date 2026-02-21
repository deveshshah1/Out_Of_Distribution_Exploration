from torch.optim.lr_scheduler import _LRScheduler


class NoamScheduler(_LRScheduler):
    """
    Noam learning rate scheduler from "Attention is All You Need".
    Formula: lr = base_lr * sqrt(warmup_steps) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps (typical: 4000)
        last_epoch: The index of last epoch (default: -1)
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)  # step starts from 1 not 0
        scale = (self.warmup_steps ** (0.5)) * min(
            step ** (-0.5), step * (self.warmup_steps ** (-1.5))
        )
        return [base_lr * scale for base_lr in self.base_lrs]
