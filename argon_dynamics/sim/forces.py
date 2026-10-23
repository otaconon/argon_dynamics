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
  v = Tensor.zeros_like(r)
  F = force(r)
  positions = []
  for _ in range(steps):
    a = F * m_inv
    r += v*dt + 0.5*a*dt*dt
    F_new = force(r)
    a_new = F*m_inv
    v += 0.5*(a + a_new)*dt
    positions.append(r.numpy().copy())
    F = F_new
  return positions