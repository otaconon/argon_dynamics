import numpy as np
from tinygrad.tensor import Tensor
from .argon import epsilon, sigma, mass

m_inv = 1/mass

def lennard_jones(r):
  dr = r.unsqueeze(1) - r.unsqueeze(0)
  d2 = (dr * dr).sum(axis=-1)
  eye = Tensor.eye(r.shape[0])
  k = (sigma**2 / (d2 + eye))**3
  U = 4*epsilon*(k*k - k) * (1 - eye)
  return U.sum() / 2

def force(r):
  return -lennard_jones(r).gradient(r)[0]

def velocity_verlet(r, dt, steps):
  v = Tensor.zeros_like(r).realize()
  r = r.realize()
  F = force(r).realize()
  positions = []

  for _ in range(steps):
    a = F * m_inv
    r = (r + v * dt + 0.5 * a * dt * dt).realize()
    
    F_new = force(r)
    a_new = F_new * m_inv
    v = v + 0.5 * (a + a_new) * dt
    
    F = F_new
    Tensor.realize(v, F)
    positions.append(r.numpy().copy())
  return positions