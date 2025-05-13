import argparse, glob, json, os, sys, re
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.validation import make_valid


def four_corners_of_polygon(poly_xy: np.ndarray) -> np.ndarray:
    poly = make_valid(Polygon(poly_xy)).convex_hull
    rect = poly.minimum_rotated_rectangle
    corners = np.array(rect.exterior.coords[:-1])  # drop duplicate last
    c = corners.mean(axis=0)
    order = np.argsort(np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0]))
    return corners[order]


def gt_vertices(json_path, pano_id):
    with open(json_path, "r") as f: data = json.load(f)
    for floor in data["merger"].values():
        for comp in floor.values():
            for part in comp.values():
                for key, pano in part.items():
                    if key == f"pano_{pano_id}" and isinstance(pano, dict):
                        return np.array(pano["layout_visible"]["vertices"])
    raise KeyError(f"pano_{pano_id} not found")


def load_pred_rect(pred_dir, pano_id):
    files = glob.glob(os.path.join(pred_dir, f"*pano_{pano_id}_rect.npy"))
    if not files:
        raise FileNotFoundError(f"No *_pano_{pano_id}_rect.npy in {pred_dir}")
    if len(files) > 1:
        print(f"[WARN] multiple predictions; using {files[0]}")
    return np.load(files[0])


def normalise_pred(pred, gt_centroid):
    x_min, x_max = pred[:, 0].min(), pred[:, 0].max()
    half_w = (x_max - x_min) / 2.0
    if half_w == 0:
        raise ValueError("Prediction rectangle has zero width.")
    return (pred) + (gt_centroid - pred.mean(axis=0))


def iou(poly_a, poly_b):
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    return inter / union if union else 0.0


def corner_stats(pred, gt, tol):
    if not len(pred) or not len(gt): return 0, 0, 0
    used = np.zeros(len(gt), bool); match = 0
    for p in pred:
        d = np.linalg.norm(gt - p, axis=1)
        j = np.argmin(d)
        if d[j] <= tol and not used[j]:
            used[j] = True; match += 1
    P = match / len(pred)
    R = match / len(gt)
    F = 0 if P + R == 0 else 2 * P * R / (P + R)
    return P, R, F


def extract_all_pano_ids(predicted_root):
    pano_ids = set()
    for floor_dir in sorted(Path(predicted_root).iterdir()):
        if floor_dir.is_dir():
            for file in floor_dir.glob("*_pano_*_rect.npy"):
                try:
                    # Extract pano ID from: floor_XX_partial_room_YY_pano_ZZ_rect.npy
                    stem_parts = file.stem.split("_")
                    pano_index = stem_parts.index("pano")
                    pano_id = stem_parts[pano_index + 1]  # the number after 'pano'
                    pano_ids.add((floor_dir.name, pano_id, file.stem))  # (floor_id, pano_id, filename without ext)
                except Exception as e:
                    print(f"[WARN] Failed to parse {file.name}: {e}")
    return sorted(pano_ids)

from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("predicted_dir", help="Root of room_2d predictions")
    p.add_argument("--pano_root", default="/building-mapper/panos", help="Root directory for pano folders")
    p.add_argument("--save-vis-dir", help="Directory to save overlay images", default=None)
    a = p.parse_args()

    pano_index = extract_all_pano_ids(a.predicted_dir)
    print(f"[INFO] Found {len(pano_index)} predicted pano rectangles.\n")

    iou_list = []

    for floor_id, pano_id, file_stem in pano_index:
        try:
            json_path = Path(a.pano_root) / floor_id / "zind_data.json"
            if not json_path.exists():
                print(f"[WARN] Skipping {floor_id}/{pano_id}: missing zind_data.json at {json_path}")
                continue

            gt = gt_vertices(json_path, pano_id)
            pred_file = Path(a.predicted_dir) / floor_id / f"{file_stem}.npy"
            pred = np.load(pred_file)

            gt_corners_raw = four_corners_of_polygon(gt)
            gt_centroid = gt_corners_raw.mean(axis=0)
            pred_norm = normalise_pred(pred, gt_centroid)

            poly_gt = make_valid(Polygon(gt))
            poly_pred = make_valid(Polygon(pred_norm))

            iou2d = iou(poly_pred, poly_gt)
            P, R, F = corner_stats(pred_norm, gt, tol=0.02)

            print(f"[FLOOR {floor_id}] Panorama {pano_id}")
            print(f"2‑D IoU (shared frame) : {iou2d:.4f}")
            print(f"Corner precision       : {P:.4f}")
            print(f"Corner recall          : {R:.4f}")
            print(f"Corner F‑score         : {F:.4f}\n")

            if iou2d > 0:
                iou_list.append(iou2d)

            if a.save_vis_dir:
                save_vis_path = Path(a.save_vis_dir) / floor_id
                save_vis_path.mkdir(parents=True, exist_ok=True)

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.fill(*poly_gt.exterior.xy, facecolor="palegreen", edgecolor="green", alpha=0.3, lw=2, label="GT (dataset)")
                ax.plot(*poly_pred.exterior.xy, "-r", lw=2, label="Prediction")
                ax.scatter(gt[:, 0], gt[:, 1], c="green", marker="o", s=40)
                ax.scatter(pred_norm[:, 0], pred_norm[:, 1], c="red", marker="x", s=50)
                ax.set_aspect("equal", adjustable="box")
                ax.invert_yaxis()
                ax.grid(ls="--", alpha=0.4)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(f"Pano {pano_id}   IoU={iou2d:.3f}")
                ax.legend(loc="lower left")
                plt.tight_layout()
                fig.savefig(save_vis_path / f"vis_{pano_id}.pdf", dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"[INFO] saved overlay → {save_vis_path}/vis_{pano_id}.pdf")

        except Exception as e:
            print(f"[ERROR] Floor {floor_id} / Pano {pano_id} → {e}")

    if iou_list:
        avg_iou = sum(iou_list) / len(iou_list)
        print(f"\n[RESULT] Average 2‑D IoU over {len(iou_list)} panoramas: {avg_iou:.4f}")
    else:
        print("\n[WARNING] No valid IoUs were computed.")