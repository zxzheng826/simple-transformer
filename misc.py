
from typing import Optional

from torch import Tensor

class NestedTensor(object):
    def __init__(self, tensors, mask:Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
    
    def to(self, device):
        """
        transfer between host and device
        """
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)
    
    def decompose(self):
        """
        seperate NestedTensor to tensors
        """
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)