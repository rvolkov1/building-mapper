# Dataset Utilities

This directory contains utilities for analyzing, visualizing and organizing the ZIND dataset, specifically focusing on extracting floors in which every room has 2+ panorama images.

## Scripts Overview

### 1. `count_multi_pano_rooms.py`

This script analyzes the dataset to find and count rooms and floors that have multiple panorama images.

#### Usage
```bash
python count_multi_pano_rooms.py [--dataset_path PATH] [--quiet]
```

#### Arguments
- `--dataset_path`: Path to the main dataset directory (default: 'dataset')
- `--quiet`: Suppress detailed output and only show final counts

#### Output
In normal mode, the script provides:
- Detailed information about each scene with multi-pano rooms
- List of panorama files for each room
- Information about floors where all rooms have multiple panoramas

In quiet mode (`--quiet`), only shows summary statistics:
- Total number of rooms with multiple panoramas
- Total number of complete multi-pano floors
- Total number of scenes analyzed

### 2. `organize_multi_pano_rooms.py`

This script creates an organized subset of the dataset containing scenes that have at least one floor where all rooms have multiple panoramas. It completely excludes floors that don't meet this criteria, keeping only panorama images from qualifying floors.

#### Usage
```bash
python organize_multi_pano_rooms.py [--dataset_path PATH] [--output_dir DIR] [--quiet]
```

#### Arguments
- `--dataset_path`: Path to the main dataset directory (default: 'dataset')
- `--output_dir`: Directory to store the organized data (default: 'zind_subset')
- `--quiet`: Suppress detailed output

#### Output Structure
```
output_dir/
├── scene_index.json           # Contains metadata about all organized scenes
└── {scene_id}/               # One directory per scene
    ├── panos/                # Directory with symlinks only to panoramas from qualifying floors
    ├── floor_plans/          # Symlink to original floor plans
    └── zind_data.json        # Symlink to original metadata file
```

#### Scene Index File
The `scene_index.json` file contains:
- Total number of organized scenes
- Total number of qualifying floors (where all rooms have multiple panoramas)
- Absolute paths to dataset and output directories
- For each scene:
  - Total number of floors in the original scene
  - Number of included floors (those with all rooms having multiple panoramas)
  - Details for only the included floors:
    - Total number of rooms
    - Details about each room (room ID, number of panoramas)
    - List of panorama files
  - Absolute path to source scene directory

### 3. `visualize_topdown.py`

This script generates a top-down visualization of a room layout including doors, windows, and openings (WDO) from the ZIND dataset.

#### Usage
```bash
python visualize_topdown.py <json_path> <pano_id> [--output-dir OUTPUT_DIR]
```

#### Arguments
- `json_path`: Path to the zind_data.json file (required)
- `pano_id`: ID of the panorama to visualize (required)
- `--output-dir`: Directory to save the output visualization (optional)

#### Output
The script generates:
- A visualization plot showing:
  - Room outline with filled area
  - Doors (red), Windows (green), and Openings (blue)
  - Labels for each WDO element
  - Room type and WDO count information
- Saves the visualization as a PNG file named `top_down_view_pano_{pano_id}.png`
- Displays statistics about the room including:
  - Room type
  - Number of vertices
  - Count of doors, windows, and openings

#### Example Usage
```bash
# Basic usage
python visualize_topdown.py zind_subset/0035/zind_data.json 72

# With custom output directory
python visualize_topdown.py zind_subset/0035/zind_data.json 72 --output-dir ./output
```

## Example Usage

1. Count multi-pano rooms in the default dataset:
```bash
python count_multi_pano_rooms.py
```

2. Count multi-pano rooms in a specific dataset location with minimal output:
```bash
python count_multi_pano_rooms.py --dataset_path /path/to/dataset --quiet
```

3. Organize scenes with multi-pano floors:
```bash
python organize_multi_pano_rooms.py
```

4. Organize scenes with custom paths and minimal output:
```bash
python organize_multi_pano_rooms.py --dataset_path /path/to/dataset --output_dir /path/to/output --quiet
```

## Notes

- Both scripts use absolute paths for symlinks to ensure they work correctly regardless of the working directory
- The organization script includes scenes that have at least one floor where all rooms have multiple panoramas
- Floors where not all rooms have multiple panoramas are completely excluded from the output
- The `panos` directory in each scene contains only panorama images from floors where all rooms have multiple panoramas
- Each scene retains its complete floor plans and metadata information
- Both scripts support quiet mode for automated processing

## Extract 2D Views (`extract_2d_views.py`)

This script extracts 2D perspective views from panoramic images in the ZInD dataset. For each room, it generates pairs of views from two different panoramas, with both cameras looking at common points in the room.

### Features

- Extracts perspective views from equirectangular panoramas
- Uses room geometry (when available) to generate meaningful viewpoints
- Automatically adjusts field of view based on camera distances
- Generates multiple view pairs per room with cameras looking at common points
- Saves detailed metadata for each view pair

### Prerequisites

```bash
pip install numpy opencv-python matplotlib scikit-learn
```

### Input Directory Structure

```
zind_subset/
├── scene_id/
│   ├── panos/
│   │   ├── room_01_pano_0.jpg
│   │   └── room_01_pano_1.jpg
│   └── zind_data.json
```

### Output Directory Structure

```
zind_subset/
├── scene_id/
│   ├── 2d_views/
│   │   └── room 01/
│   │       ├── pair_0/
│   │       │   ├── view1.jpg
│   │       │   ├── view2.jpg
│   │       │   └── target_info.txt
│   │       └── pair_1/
│   │           └── ...
```

### Usage

```bash
python extract_2d_views.py [options]
```

#### Options

- `--base_dir`: Root directory containing scene folders (default: 'zind_subset')
- `--num_pairs`: Number of view pairs to generate per room (default: 10)
- `--width`: Width of output perspective views (default: 512)
- `--height`: Height of output perspective views (default: 512)
- `--base_fov`: Base field of view in degrees (default: 90)
- `--min_fov`: Minimum field of view in degrees (default: 60)
- `--scene`: Process specific scene (optional)

#### Examples

Process all scenes with default settings:
```bash
python extract_2d_views.py
```

Process a specific scene with custom parameters:
```bash
python extract_2d_views.py --scene 0035 --num_pairs 15 --width 1024 --height 1024 --base_fov 80
```

### Output Files

For each pair of views, the script generates:

1. `view1.jpg` and `view2.jpg`: The extracted perspective views
2. `target_info.txt`: Metadata file containing:
   - Target point coordinates
   - Camera positions and rotations
   - Yaw angles for both views
   - Field of view used 