import os 
from feature_matching.recon_3d import run_recon_all

## replace main_dir with the path to your dataset. Should be same dataset as you used with Pouya's script

if __name__ == "__main__":
  main_dir = "zind_subset"

  run_recon_all(main_dir)
