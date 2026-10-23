import numpy as np
from numpy.typing import NDArray

sigma = 3.405
epsilon = 0.01032
mass = 39.948
cutoff = 8.51

class Argon:
  def __init__(self, pos: NDArray):
    self.pos = pos