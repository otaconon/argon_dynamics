from argon_dynamics import sim
from tinygrad.tensor import Tensor

def main():
  r = Tensor.rand(1000, 2) * 8
  print(r.numpy())
  positions = sim.velocity_verlet(r, 0.01, 100)
  print(positions[0])
  print(positions[1])
  print(positions[-1])
  
if __name__ == "__main__":
  main()