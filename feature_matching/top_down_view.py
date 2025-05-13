import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def top_down_view(xyz, x_seg, out_png=None):
    keep   = x_seg[:, 0] != -1
    pts    = xyz[keep]
    labels = x_seg[keep, 0]

    cmap = {0: "yellow", 1: "red", 2: "green", 3: "blue"}
    colours = [cmap.get(int(l), "gray") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(pts[:, 0], pts[:, 1], s=1, c=colours)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Top‑down semantic view")
    ax.invert_yaxis()

    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color="yellow", label="Outer background"),
        mpatches.Patch(color="red", label="Windows"),
        mpatches.Patch(color="green", label="Door"),
        mpatches.Patch(color="blue", label="Walls"),
        mpatches.Patch(color="gray", label="Background inside"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize="small")

    plt.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union


def top_down_border_min_rect(
        xyz, x_seg,
        keep_labels=(1, 2, 3),
        x_step=0.5,
        smooth_window=5,
        gap_bins=5,
        keep_ratio=0.95,         # keep this fraction of *nearest* outline pts
        out_png=None,
        out_npy=None,
    ):
    """
    Fit a minimum‑area rotated rectangle that encloses the central
    *keep_ratio* fraction of the smoothed outline points.

    keep_ratio = 1.0 → original behaviour (no trimming)
    keep_ratio < 1.0 → trims radial outliers before rectangle fitting
    """

    # ---------------------------------------------------------------- STEP 1 – collect outline (unchanged)
    keep = np.isin(x_seg[:, 0], keep_labels)
    pts_xy = xyz[keep, :2]
    if pts_xy.size == 0:
        raise ValueError("No points left after label filtering")

    x_vals = pts_xy[:, 0]
    bins = np.arange(x_vals.min(), x_vals.max() + x_step, x_step)
    idx_pts = np.digitize(x_vals, bins)

    y_lo = np.full(len(bins),  np.inf)
    y_hi = np.full(len(bins), -np.inf)
    for idx, (_, y) in zip(idx_pts, pts_xy):
        y_lo[idx] = min(y_lo[idx], y)
        y_hi[idx] = max(y_hi[idx], y)

    valid = np.isfinite(y_lo) & np.isfinite(y_hi)
    for start in range(len(bins)):
        if valid[start]:
            continue
        end = start
        while end < len(bins) and not valid[end]:
            end += 1
        gap = end - start
        if (
            start > 0 and end < len(bins) and gap <= gap_bins
            and valid[start - 1] and valid[end]
        ):
            y_lo[start:end] = np.linspace(y_lo[start - 1], y_lo[end], gap + 2)[1:-1]
            y_hi[start:end] = np.linspace(y_hi[start - 1], y_hi[end], gap + 2)[1:-1]
            valid[start:end] = True

    idx = np.flatnonzero(valid)
    x_cent = bins[idx] + x_step / 2
    y_lo, y_hi = y_lo[idx], y_hi[idx]

    # smooth
    if smooth_window >= 3 and smooth_window % 2 == 1:
        k = np.ones(smooth_window) / smooth_window
        pad = smooth_window // 2
        y_lo = np.convolve(np.pad(y_lo, pad, mode="edge"), k, mode="valid")
        y_hi = np.convolve(np.pad(y_hi, pad, mode="edge"), k, mode="valid")

    lower = np.column_stack([x_cent, y_lo])
    upper = np.column_stack([x_cent, y_hi])
    outline = np.vstack([lower, upper[::-1]])          # (N,2)

    # ---------------------------------------------------------------- STEP 2 – radial trimming
    if keep_ratio < 1.0:
        centroid = outline.mean(axis=0)
        dists = np.linalg.norm(outline - centroid, axis=1)
        thresh = np.quantile(dists, keep_ratio)
        outline_trim = outline[dists <= thresh]
        if len(outline_trim) >= 4:
            outline = outline_trim
        # else: too few points left, fall back to all points

    # ---------------------------------------------------------------- STEP 3 – minimum‑area rectangle
    poly_outline = Polygon(outline).convex_hull
    min_rect_poly = poly_outline.minimum_rotated_rectangle
    rect_coords = np.array(min_rect_poly.exterior.coords[:-1])  # (4,2)

    # sort vertices clockwise
    centroid = rect_coords.mean(axis=0)
    angles = np.arctan2(rect_coords[:, 1] - centroid[1],
                        rect_coords[:, 0] - centroid[0])
    rect_sorted = rect_coords[np.argsort(angles)]

    rect_closed = np.vstack([rect_sorted, rect_sorted[0]])

    # ---------------------------------------------------------------- STEP 4 – visualise
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(*outline.T, color="lightgray", lw=0.8, label="trimmed outline")
    ax.plot(*rect_closed.T, "-k", lw=2, label="min‑area rectangle")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"keep_ratio = {keep_ratio} → oriented min‑area rectangle")
    ax.legend()
    plt.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_npy:
        np.save(out_npy, rect_sorted)

    plt.show()
    return rect_sorted


def project_openings_on_rectangle(
        xyz: np.ndarray,
        x_seg: np.ndarray,
        rect: np.ndarray,                 # (4, 2) – corners from previous step
        door_label=1,
        window_label=2,
        door_col="green",
        window_col="red",
        wall_col="black",
        out_png: str | None = None,
        out_npy: str | None = None,
    ):
    """
    Projects door/window points onto the nearest rectangle edge and draws them.

    Parameters
    ----------
    rect : (4, 2) ndarray
        Rectangle corners in clockwise order:
        [LL, LR, UR, UL].  (Output of top_down_border_rectangle.)
    """

    # ----------------------------------------------------------- STEP 0  helpers
    def project_to_edge(p, a, b):
        """Project point *p* onto edge a‑>b (clamped to segment)."""
        ab = b - a
        t  = np.clip(np.dot(p - a, ab) / (ab @ ab), 0.0, 1.0)
        proj = a + t * ab
        dist = np.linalg.norm(p - proj)
        return proj, dist

    edges = [(rect[i], rect[(i + 1) % 4]) for i in range(4)]   # four edges

    # ----------------------------------------------------------- STEP 1  gather
    door_pts   = xyz[x_seg[:, 0] == door_label, :2]
    window_pts = xyz[x_seg[:, 0] == window_label, :2]

    proj_doors, proj_windows = [], []

    for cloud, store in [(door_pts, proj_doors), (window_pts, proj_windows)]:
        for p in cloud:
            # find closest edge
            best_proj, best_dist = None, np.inf
            for a, b in edges:
                proj, dist = project_to_edge(p, a, b)
                if dist < best_dist:
                    best_proj, best_dist = proj, dist
            store.append(best_proj)

    proj_doors   = np.array(proj_doors)   if proj_doors   else np.empty((0, 2))
    proj_windows = np.array(proj_windows) if proj_windows else np.empty((0, 2))

    # ----------------------------------------------------------- STEP 2  plot
    fig, ax = plt.subplots(figsize=(8, 4))

    # rectangle outline
    rect_closed = np.vstack([rect, rect[0]])
    ax.plot(*rect_closed.T, "-k", linewidth=2, color=wall_col, label="walls")

    # projected doors / windows
    if proj_doors.size:
        ax.scatter(proj_doors[:, 0], proj_doors[:, 1],
                   c=door_col, marker="s", s=25, label="doors")
    if proj_windows.size:
        ax.scatter(proj_windows[:, 0], proj_windows[:, 1],
                   c=window_col, marker="o", s=25, label="windows")

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Doors & windows projected to rectangle")
    ax.legend(loc="lower left")
    plt.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")

    if out_npy:
        # save projected coordinates: [x, y, type]  where type = 1 door, 2 window
        data = []
        if proj_doors.size:
            data.append(np.column_stack([proj_doors, np.full(len(proj_doors), 1)]))
        if proj_windows.size:
            data.append(np.column_stack([proj_windows, np.full(len(proj_windows), 2)]))
        np.save(out_npy, np.vstack(data) if data else np.empty((0, 3)))

    plt.show()
    return proj_doors, proj_windows