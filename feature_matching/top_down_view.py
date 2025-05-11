import numpy as np
import cv2
import matplotlib.pyplot as plt

# ----------------------------------------------
# user‑tweakable parameters
# ----------------------------------------------
RES_MM        = 25     # grid resolution (25 mm ≈ 1″)
MORPH_KERNEL  = 5      # size (px) of closing kernel
BORDER_THICK  = 3      # contour stroke width

# label → BGR colour (OpenCV’s channel order!)
COLOR_BGR = {
    1: (  255,   0, 0),   # windows – red     (B, G, R)
    2: (0, 255, 0),   # doors   – white
    3: (0,   0,   255),   # walls   – blue
}

# ----------------------------------------------
# helper : rasterise points of one label
# ----------------------------------------------
def rasterise_label(pts_xy, res):
    """Return binary grid (uint8 0/255) for the given 2‑D points."""
    if pts_xy.size == 0:
        return None, 0, 0
    cols = (pts_xy[:,0] / res).astype(int)
    rows = (pts_xy[:,1] / res).astype(int)
    h, w = rows.max()+1, cols.max()+1
    grid = np.zeros((h, w), dtype=np.uint8)
    grid[rows, cols] = 255
    return grid, h, w

# ----------------------------------------------
# main function
# ----------------------------------------------
def top_down_border(xyz, x_seg, out_png=None):
    # 0. keep only labels of interest (1,2,3)
    mask      = np.isin(x_seg[:,0], [1,2,3])
    pts_all   = xyz[mask, :2]
    labels_all= x_seg[mask, 0]

    # shift so origin is bottom‑left
    pts_all  -= pts_all.min(axis=0)

    # resolution in metres
    res = RES_MM / 1_000.0

    # 1. prepare blank RGB canvas at full extent
    wh = (pts_all / res).astype(int).max(axis=0) + 1
    canvas = np.full((*wh[::-1], 3), 255, dtype=np.uint8)   # HxWx3, white

    # 2. process each semantic label separately
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL, MORPH_KERNEL))
    for lbl, colour in COLOR_BGR.items():

        pts_lbl = pts_all[labels_all == lbl]
        grid, h, w = rasterise_label(pts_lbl, res)
        if grid is None:          # label absent
            continue

        # – close tiny holes / cracks
        grid_closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)

        # – find outer contours
        contours, _ = cv2.findContours(grid_closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # – draw borders onto canvas
        cv2.drawContours(canvas, contours, -1, colour, BORDER_THICK)

    # 3. display (origin at bottom‑left like floor plans)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cv2.flip(canvas, 0))   # vertical flip so +y is up
    ax.set_axis_off()
    plt.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()

    return canvas