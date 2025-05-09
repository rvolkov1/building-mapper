import os 
import re

import matplotlib.pyplot as plt

from feature_matching.feature_match import get_3d_pt_cloud
from feature_matching.feature_matching_dl import get_dl_recon

def parse_correspondence_file(path):
  print(path)
  with open(path, 'r') as file:
    content = file.read()

  # Extract general info
  target_match = re.search(r"Target point:\s*\[([-\d.]+),\s*([-\d.]+)\]", content)
  fov_match = re.search(r"Field of view:\s*([-\d.]+)°", content)
  total_views_match = re.search(r"Total views:\s*(\d+)", content)

  result = {
      "target_point": [float(target_match.group(1)), float(target_match.group(2))],
      "field_of_view": float(fov_match.group(1)),
      "total_views": int(total_views_match.group(1)),
      "views": []
  }

  # Extract each view block
  view_blocks = re.findall(
      r"View (\d+):\s+Camera position: \[([-\d.]+), ([\-\d.]+)\]\s+Camera rotation: ([\-\d.]+)°\s+Yaw angle: ([\-\d.]+)°\s+Distance to target: ([\-\d.]+)m",
      content
  )

  for view in view_blocks:
      view_data = {
          "index": int(view[0]),
          "camera_position": [float(view[1]), float(view[2])],
          "camera_rotation": float(view[3]),
          "yaw_angle": float(view[4]),
          "distance_to_target": float(view[5])
      }
      result["views"].append(view_data)

  return result

def parse_room(room_path):
  corres = sorted(os.listdir(room_path))[1:]

  target_pts = []

  for corr in corres:
    corr_path = room_path + "/" + corr

    fname = "correspondence_info.txt"

    info_file = corr_path + "/" + fname

    res = parse_correspondence_file(info_file)

    target_pts.append(res["target_point"])

  print(target_pts)

  x_vals, y_vals = zip(*target_pts)

  # Plotting
  plt.figure(figsize=(6, 6))
  plt.scatter(x_vals, y_vals, color='blue', marker='o')
  plt.plot(x_vals, y_vals, linestyle='--', color='gray', alpha=0.5)  # optional: connect points
  for i, (x, y) in enumerate(target_pts):
    plt.text(x + 0.05, y + 0.05, str(i), fontsize=9, color='red')

  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title('2D Scatter Plot of Points')
  plt.grid(True)
  plt.axis('equal')  # ensures equal scaling on both axes
  plt.show()


if __name__ == "__main__":
  path = "/Users/rvolkov/Documents/uni/5561/building-mapper/zind_subset/0528/2d_views/room 01/corres_0"
  #get_3d_pt_cloud(path)
  #all_opencv(path)

  room_path = "/Users/rvolkov/Documents/uni/5561/building-mapper/zind_subset/0528/2d_views/room 07"

  parse_room(room_path)

  #get_dl_recon(path)
