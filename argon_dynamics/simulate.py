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
  w, h = 7, 7
  
  r = sim.gen_grid_r(N, w, h)
  simulation = sim.Simulation(n=N, dt=0.01, steps=1000, w=w, h=h)
  
  positions = simulation.run(r) 
  
  visualizer = vis.Visual(w=800, h=800, dim=(8, 8))
  visualizer.run(positions)

if __name__ == "__main__":
  main()