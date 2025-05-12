import os, glob, math
import numpy as np
import matplotlib.pyplot as plt

def load_rooms(dir_path):
    rooms = {}
    for rect_file in glob.glob(os.path.join(dir_path, "*_rect.npy")):
        if "_openings_on_" in rect_file:
            continue
        name = os.path.basename(rect_file)[:-9]
        rect = np.load(rect_file)
        open_file = os.path.join(dir_path, f"{name}_openings_on_rect.npy")
        if not os.path.exists(open_file):
            print(f"Missing: {name}")
            continue
        open_xy = np.load(open_file)
        doors = open_xy[open_xy[:, 2] == 1][:, :2]
        wins  = open_xy[open_xy[:, 2] == 2][:, :2]
        w = rect[:, 0].ptp()
        h = rect[:, 1].ptp()
        area = w * h
        rooms[name] = {"rect": rect, "doors": doors, "windows": wins, "w": w, "h": h, "area": area}
    return rooms

def grid_pack(rooms):
    target_w = math.sqrt(sum(r["area"] for r in rooms.values()))
    placed = {}
    cur_x = row_y = row_h = 0.0
    for name, data in rooms.items():
        w, h, rect = data["w"], data["h"], data["rect"]
        if cur_x > 0 and cur_x + w > target_w:
            row_y += row_h
            cur_x = row_h = 0.0
        shift = np.array([cur_x - rect[:, 0].min(), row_y - rect[:, 1].min()])
        placed[name] = {
            "rect": rect + shift,
            "doors": data["doors"] + shift if data["doors"].size else data["doors"],
            "windows": data["windows"] + shift if data["windows"].size else data["windows"],
        }
        cur_x += w
        row_h = max(row_h, h)
    return placed

def plot_grid(placed, out_png="grid_floor_plan.png"):
    fig, ax = plt.subplots(figsize=(12, 10))
    for p in placed.values():
        r = np.vstack([p["rect"], p["rect"][0]])
        ax.plot(*r.T, "-k", lw=2)
        if p["doors"].size:
            ax.scatter(p["doors"][:, 0], p["doors"][:, 1], c="green", marker="s", s=35)
        if p["windows"].size:
            ax.scatter(p["windows"][:, 0], p["windows"][:, 1], c="red", marker="o", s=25)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Grid floor plan")
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved", out_png)

if __name__ == "__main__":
    rooms = load_rooms("/building-mapper/room_2d")
    if not rooms:
        exit()
    placed = grid_pack(rooms)
    plot_grid(placed)
