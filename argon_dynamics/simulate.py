from argon_dynamics import sim
from argon_dynamics import vis
from tinygrad.tensor import Tensor
import numpy as np
from vispy import plot as vp
import vispy
import math

vispy.use("pyside6")
def main():
  N = 16
  dt = 0.01
  steps = 1000
  gx, gy = np.meshgrid(np.linspace(1.0, 7.0, math.ceil(math.sqrt(N))), np.linspace(1.0, 7.0, math.ceil(math.sqrt(N))))
  grid = np.stack([gx.ravel(), gy.ravel()], axis=-1)[:N].astype(np.float32)
  r = Tensor(grid)
  print(r.numpy())
  positions = sim.velocity_verlet(r, dt, steps)
  print(positions[0])
  print(positions[1])
  print(positions[-1])
  
  visualizer = vis.Visual(w=800, h=800, dim=(8, 8))
  visualizer.run(positions)

if __name__ == "__main__":
  main()