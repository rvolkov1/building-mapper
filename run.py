from feature_matching.feature_match import get_3d_pt_cloud
from feature_matching.feature_matching_dl import get_dl_recon

if __name__ == "__main__":
  path = "/Users/rvolkov/Documents/uni/5561/building-mapper/zind_subset/0528/2d_views/room 01/corres_0"
  #get_3d_pt_cloud(path)
  #all_opencv(path)

  get_dl_recon(path)
