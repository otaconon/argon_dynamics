from argon_dynamics import sim
from argon_dynamics import vis
from tinygrad.tensor import Tensor
import vispy

vispy.use("pyside6")

def main():
  N = 16
  box_size = N**2/2
  
  r = sim.gen_grid_r(N, box_size)
  simulation = sim.Simulation(n=N, dt=0.03, steps=1000, box_size=box_size)
  
  positions = simulation.run(r)
  
  visualizer = vis.Visual(w=800, h=800, dim=(box_size, box_size))
  visualizer.run(positions)

if __name__ == "__main__":
  main()