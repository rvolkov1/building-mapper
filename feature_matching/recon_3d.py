import os, sys, time
import argparse
import importlib

import numpy as np
import torch
import cv2
from tqdm.notebook import tqdm
from imageio import imread
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go

from feature_matching.HoHoNet.lib.config import config
from feature_matching.top_down_view import top_down_view, top_down_border_min_rect, project_openings_on_rectangle

from pathlib import Path

def run_recon_all(main_dir):
  def is_4_digit_number(name):
    return name.isdigit() and len(name) == 4

  dirs = [file for file in sorted(os.listdir(main_dir)) if is_4_digit_number(file)]

  for _dir in tqdm(dirs):
    pth = main_dir + "/" + _dir

    panos = [fname for fname in sorted(os.listdir(pth + "/" + "panos")) if fname.endswith(".jpg")]

    os.makedirs(pth + "/room_recons", exist_ok=True)

    for pano in panos:
      pano_path = pth + "/panos/" + pano

      img, depth = predict_dl_pano_depth(pano_path, viz=False)
      xyz, xyzrgb = reproject(img, depth, viz=False)

      saves = {}
      saves["img"] = img
      saves["depth"] = depth
      saves["xyz"] = xyz
      saves["xyzrgb"] = xyzrgb

      name = Path(pano).stem

      save_fname = pth + "/room_recons" + "/" + name + ".npz"

      np.savez(save_fname, **saves)

PRETRAINED_PTH = "/building-mapper/feature_matching/HoHoNet/ckpt/mp3d_depth_HOHO_depth_dct_efficienthc_TransEn1_hardnet/ep60.pth"

#if not os.path.exists(PRETRAINED_PTH):
#    os.makedirs(os.path.split(PRETRAINED_PTH)[0], exist_ok=True)
#    !gdown 'https://drive.google.com/uc?id=1kZFPwdo36Uk7qP96yYUyQebZtGjsEabL' -O $PRETRAINED_PTH
#
#if not os.path.exists('assets/pano_asmasuxybohhcj.png'):
#    !gdown 'https://drive.google.com/uc?id=1CXl6RPK6yPRFXxsa5OisHV9KwyRcejHu' -O 'assets/pano_asmasuxybohhcj.png'

#path = '../building-mapper/zind_subset/0528/panos/floor_01_partial_room_14_pano_21.jpg'

config.defrost()
config.merge_from_file('feature_matching/HoHoNet/config/mp3d_depth/HOHO_depth_dct_efficienthc_TransEn1_hardnet.yaml')
config.model.kwargs['backbone_config']['kwargs']['pretrained'] = False
config.freeze()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_file = "feature_matching.HoHoNet.lib.model.hohonet"

model_file = importlib.import_module(model_file)
model_class = getattr(model_file, config.model.modelclass)
net = model_class(**config.model.kwargs)
net.load_state_dict(torch.load(PRETRAINED_PTH, map_location=device))
net = net.eval().to(device)

def depth_comparison_viz(rgb, pred_depth):
  plt.figure(figsize=(15,6))

  plt.subplot(121)
  plt.imshow(rgb[80:-80])
  plt.axis('off')
  plt.title('RGB')

  plt.subplot(122)
  plt.imshow(
      pred_depth.squeeze()[80:-80].cpu().numpy(),
      cmap='inferno_r', vmin=0.1, vmax=10)
  plt.axis('off')
  plt.title('depth prediction')

  plt.show()

def predict_dl_pano_depth(path, viz=True):
  #path = "/Users/rvolkov/Documents/uni/5561/building-mapper/zind_subset/0528/panos/floor_01_partial_room_09_pano_94.jpg"
  rgb = imread(path)

  rgb = cv2.resize(rgb, (1024, 512), interpolation=cv2.INTER_LINEAR)

  if (viz):
    plt.imshow(rgb)
    plt.show()

  x = torch.from_numpy(rgb).permute(2,0,1)[None].float() / 255.
  if x.shape[2:] != config.dataset.common_kwargs.hw:
      x = torch.nn.functional.interpolate(x, config.dataset.common_kwargs.hw, mode='area')
  x = x.to(device)

  with torch.no_grad():
      ts = time.time()
      pred_depth = net.infer(x)
      if not torch.is_tensor(pred_depth):
          pred_depth = pred_depth.pop('depth')
      if torch.cuda.is_available():
          torch.cuda.synchronize()
      print(f'Eps time: {time.time() - ts:.2f} sec.')

  if viz:
    depth_comparison_viz(rgb, pred_depth)

  return rgb, pred_depth

def reproject(rgb, pred_depth, seg=None, viz=True):
    def get_uni_sphere_xyz(H, W):
        j, i = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        u = (i+0.5) / W * 2 * np.pi
        v = ((j+0.5) / H - 0.5) * np.pi
        z = -np.sin(v)
        c = np.cos(v)
        y = c * np.sin(u)
        x = c * np.cos(u)
        return np.stack([x, y, z], -1)           # (H,W,3)

    d = pred_depth.squeeze().cpu().unsqueeze(-1).numpy()      # (H,W,1)
    xyz = d * get_uni_sphere_xyz(*pred_depth.shape[2:])         # (H,W,3)

    xyzrgb_full = np.concatenate([xyz, rgb/255.0], axis=-1)     # (H,W,6)

    xyzrgb_flat = xyzrgb_full[80:-80][::2, ::2].reshape(-1, 6)  # (N0,6)
    if seg is not None:
        seg_flat = seg[80:-80][::2, ::2].reshape(-1, 1)         # (N0,1)

    keep = xyzrgb_flat[:,2] < 1.5                                # boolean mask
    xyzrgb = xyzrgb_flat[keep]                                   # (N,6)
    x_seg  = seg_flat[keep] if seg is not None else None         # (N,1)

    if viz:
        fig = go.Figure(
            data=[go.Scatter3d(
                x=xyzrgb[:,0], y=xyzrgb[:,1], z=xyzrgb[:,2],
                mode='markers',
                marker=dict(size=1, color=xyzrgb[:,3:]),
            )],
            layout=dict(scene=dict(
                xaxis=dict(visible=False, range=[-3,3]),
                yaxis=dict(visible=False, range=[-3,3]),
                zaxis=dict(visible=False, range=[-3,3]),
            ))
        )
        fig.show()

    return xyz, xyzrgb, x_seg



# Example usage

if __name__ == "__main__":
    import argparse
    import os
    import numpy as np
    import cv2
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("floor_id", help="Floor ID (e.g., 0035)")
    parser.add_argument("--pano_root", default="panos", help="Root directory of panorama images")
    parser.add_argument("--seg_root", default="segs", help="Root directory of segmentation outputs")
    parser.add_argument("--out_root", default="room_2d", help="Output directory for top-down and layout files")
    args = parser.parse_args()

    floor_id = args.floor_id
    pano_dir = Path(args.pano_root) / floor_id
    seg_dir = Path(args.seg_root) / floor_id
    out_dir = Path(args.out_root) / floor_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for pano_path in sorted(pano_dir.glob("*.jpg")):
        pano_name = pano_path.stem  # e.g., pano_43
        seg_path = seg_dir / f"{pano_name}.npy"

        if not seg_path.exists():
            print(f"[WARN] Missing segmentation: {seg_path}")
            continue

        seg_raw = np.load(seg_path)
        seg_resized = cv2.resize(seg_raw, (1024, 512),
                                 interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        rgb, depth = predict_dl_pano_depth(str(pano_path), viz=False)
        xyz, xyzrgb, x_seg = reproject(rgb, depth, seg=seg_resized, viz=False)

        # Save top-down view
        top_down_view(
            xyzrgb,
            x_seg,
            out_png=out_dir / f"{pano_name}_top_down.pdf",
        )

        # Save rectangle bounding layout
        rect = top_down_border_min_rect(
            xyzrgb,
            x_seg,
            keep_ratio=0.95,
            out_npy=out_dir / f"{pano_name}_rect.npy"
        )

        # Project doors/windows
        project_openings_on_rectangle(
            xyzrgb,
            x_seg,
            rect,
            door_label=2,
            window_label=1,
            out_png=out_dir / f"{pano_name}_openings_on_rect.pdf",
            out_npy=out_dir / f"{pano_name}_openings_on_rect.npy"
        )