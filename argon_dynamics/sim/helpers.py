import numpy as np
import math
from tinygrad.tensor import Tensor

def gen_grid_r(N: int, box_size: float):
  gx, gy = np.meshgrid(np.linspace(1.0, box_size, math.ceil(math.sqrt(N))), np.linspace(1.0, box_size, math.ceil(math.sqrt(N))))
  grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)[:N].astype(np.float32)
  return Tensor(grid)