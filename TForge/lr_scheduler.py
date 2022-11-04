import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class TForgeScheduler(_LRScheduler):
    """ This scheduler ignores the initial learning rate in the optimizer
    and overrides it with the calculated value.
    """
    def __init__(self, 
                 optimizer: Optimizer,
                 embed_dim: int=512,
                 warmup_steps: int=4000,
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:

        self.embed_dim = embed_dim
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> float:
        lr = self.calc_lr(self._step_count, self.embed_dim, self.warmup_steps)
        return [lr] * self.num_param_groups

    def calc_lr(self, step, dim_embed, warmup_steps):
        return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))