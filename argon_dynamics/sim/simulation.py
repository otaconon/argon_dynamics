import numpy as np
from tinygrad.tensor import Tensor

sigma = 3.405
epsilon = 0.01032
mass = 39.948
m_inv = 1/mass
cutoff = 8.51

class Simulation:
  def __init__(self, n, dt, steps, w, h):
    self.n = n
    self.dt = dt
    self.steps = steps
    self.w = w
    self.h = h
    
  def run(self, r) -> list[float]:
    return self._velocity_verlet(r)
  
  def _velocity_verlet(self, r):
    v = Tensor.zeros_like(r).realize()
    r = r.realize()
    F = self._force(r).realize()
    positions = []

    for _ in range(self.steps):
      a = F * m_inv
      r = (r + v * self.dt + 0.5 * a * self.dt * self.dt).realize()
      
      F_new = self._force(r)
      a_new = F_new * m_inv
      v = v + 0.5 * (a + a_new) * self.dt
      
      F = F_new
      Tensor.realize(v, F)
      positions.append(r.numpy().copy())
    return positions
  
  def _force(self, r):
    return -self._lennard_jones(r).gradient(r)[0]
  
  def _lennard_jones(self, r):
    dr = r.unsqueeze(1) - r.unsqueeze(0)
    d2 = (dr * dr).sum(axis=-1)
    eye = Tensor.eye(r.shape[0])
    k = (sigma**2 / (d2 + eye))**3
    U = 4*epsilon*(k*k - k) * (1 - eye)
    return U.sum() / 2


  