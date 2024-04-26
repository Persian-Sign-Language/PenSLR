import torch.nn as nn
import torch

class RegularizationModule(nn.Module):

    def __init__(self):
        super().__init__()

    def regularization_loss(self):
        return torch.tensor(0)

class L2RegularizationModule(RegularizationModule):

    def __init__(self):
        super().__init__()
    
    def regularization_loss(self):
        params = [p for p in self.parameters() if p.requires_grad]
        s = 0
        for param in params:
            s += torch.pow(param, 2).sum()
        return s
    
class L1RegularizationModule(RegularizationModule):

    def __init__(self):
        super().__init__()
    
    def regularization_loss(self):
        params = [p for p in self.parameters() if p.requires_grad]
        s = 0
        for param in params:
            s += torch.norm(param, 1)
        return s
        

