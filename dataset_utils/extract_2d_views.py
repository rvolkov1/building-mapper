import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import re
import json
import math

def equirectangular_to_perspective(pano, fov=90, theta=0, phi=0, width=512, height=512):
    """
    Convert an equirectangular panorama to a perspective view.
    
    Args:
        pano: Input panorama (BGR numpy array)
        fov: Field of view in degrees
        theta: Yaw angle in degrees
        phi: Pitch angle in degrees
        width, height: Output image dimensions
    
    Returns:
        Perspective image (RGB)
    """
    h, w, _ = pano.shape
    fov_rad = np.deg2rad(fov)
    
    # Create pixel coordinate grid
    x = np.linspace(-np.tan(fov_rad/2), np.tan(fov_rad/2), width)
    y = np.linspace(-np.tan(fov_rad/2), np.tan(fov_rad/2), height)
    xv, yv = np.meshgrid(x, -y)
    
    # Compute direction vectors
    zv = np.ones_like(xv)
    d = np.sqrt(xv**2 + yv**2 + zv**2)
    xv, yv, zv = xv/d, yv/d, zv/d
    
    # Apply rotations for yaw and pitch
    theta_rad, phi_rad = np.deg2rad(theta), np.deg2rad(phi)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(phi_rad), -np.sin(phi_rad)],
                    [0, np.sin(phi_rad), np.cos(phi_rad)]])
    R_y = np.array([[np.cos(theta_rad), 0, np.sin(theta_rad)],
                    [0, 1, 0],
                    [-np.sin(theta_rad), 0, np.cos(theta_rad)]])
    
    vectors = np.stack([xv, yv, zv], axis=-1)
    rotated = np.dot(np.dot(vectors, R_x.T), R_y.T)
    
    # Convert to equirectangular coordinates
    lon = np.arctan2(rotated[..., 0], rotated[..., 2])
    lat = np.arcsin(rotated[..., 1])
    x_pano = ((lon / np.pi) + 1) * (w / 2)
    y_pano = ((-lat / (np.pi / 2)) + 1) * (h / 2)
    
    # Sample the panorama
    perspective = cv2.remap(pano, x_pano.astype(np.float32), y_pano.astype(np.float32),
                            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return perspective

def load_camera_data(scene_dir):
    """
    Load camera rotation and translation metadata from zind_data.json.
    
    Args:
        scene_dir: Path to the scene directory
    
    Returns:
        Dictionary mapping pano_file -> {"rotation": degrees, "translation": [x, y, z]}
    """
    json_path = os.path.join(scene_dir, "zind_data.json")
    camera_data = {}
    
    if not os.path.exists(json_path):
        print(f"Warning: No zind_data.json found in {scene_dir}")
        return camera_data
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Process merger section, which has the floor, room, and pano structure
        merger = data.get("merger", {})
        
        # Process each floor
        for floor_key, floor_data in merger.items():
            # Process each complete room
            for room_key, room_data in floor_data.items():
                if room_key.startswith("complete_room_"):
                    # Process each partial room
                    for partial_room_key, partial_room_data in room_data.items():
                        if partial_room_key.startswith("partial_room_"):
                            # Process each panorama
                            for pano_key, pano_data in partial_room_data.items():
                                if pano_key.startswith("pano_"):
                                    # Extract rotation and translation data
                                    floor_plan_trans = pano_data.get("floor_plan_transformation", {})
                                    rotation = floor_plan_trans.get("rotation", 0)
                                    
                                    # Extract translation (may be stored in different formats)
                                    translation = floor_plan_trans.get("translation", [0, 0, 0])
                                    if isinstance(translation, dict):
                                        # Handle case where translation is stored as {x: val, y: val, z: val}
                                        translation = [
                                            translation.get("x", 0),
                                            translation.get("y", 0),
                                            translation.get("z", 0)
                                        ]
                                    elif isinstance(translation, list) and len(translation) == 2:
                                        # Handle case where translation is [x, y] only
                                        translation = [translation[0], translation[1], 0]
                                    
                                    image_path = pano_data.get("image_path", "")
                                    if image_path:
                                        pano_file = os.path.basename(image_path)
                                        camera_data[pano_file] = {
                                            "rotation": rotation,
                                            "translation": translation
                                        }
        
        print(f"Loaded {len(camera_data)} camera data entries from {json_path}")
        return camera_data
    
    except Exception as e:
        print(f"Error loading zind_data.json: {e}")
        return camera_data

def calculate_optimal_view_parameters(camera1_data, camera2_data, base_yaw, distance_scale=1.0):
    """
    Calculate optimal viewing angles and FOV for two cameras to look at a common point.
    
    Args:
        camera1_data: Dict with rotation and translation for camera 1
        camera2_data: Dict with rotation and translation for camera 2
        base_yaw: Base yaw angle in degrees (0-360)
        distance_scale: Scale factor for distance-based adjustments
    
    Returns:
        (yaw1, yaw2, fov1, fov2): Optimal yaw angles and FOVs for both cameras
    """
    rotation1 = camera1_data.get("rotation", 0)
    rotation2 = camera2_data.get("rotation", 0)
    
    translation1 = camera1_data.get("translation", [0, 0, 0])
    translation2 = camera2_data.get("translation", [0, 0, 0])
    
    # Convert to numpy arrays for vector operations
    pos1 = np.array(translation1[:2])  # Use only x, y coordinates
    pos2 = np.array(translation2[:2])
    
    # Calculate distance between cameras
    distance = np.linalg.norm(pos2 - pos1)
    
    # Synchronize the cameras to point in the same world direction
    # The base_yaw is now a global direction in the world
    # Each camera needs to adapt its own yaw to point in this direction
    
    # Global rotation direction for this pair
    world_direction_yaw = base_yaw
    
    # Apply camera-specific rotations to point in the same world direction
    # For each camera, we need to subtract its own rotation to get its local yaw
    yaw1 = world_direction_yaw - rotation1
    yaw2 = world_direction_yaw - rotation2
    
    # If distance is very small, we're at essentially the same position
    # In this case, apply a small offset to get slightly different perspectives
    if distance < 0.1:
        angle_offset = 5  # Smaller offset to maintain better overlap
        yaw1 += angle_offset
        yaw2 -= angle_offset
    else:
        # For cameras that are further apart, we need to account for their relative positions
        # Calculate the angle between cameras from camera1's perspective
        if np.linalg.norm(pos2 - pos1) > 0:
            angle_between = np.rad2deg(np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0]))
            
            # If the base_yaw is approximately in the direction of camera2 from camera1,
            # apply a slight adjustment to ensure overlap
            relative_angle = (angle_between - world_direction_yaw) % 360
            if relative_angle < 90 or relative_angle > 270:
                # When looking roughly toward the other camera, adjust slightly
                angle_factor = np.cos(np.deg2rad(relative_angle))
                adjustment = 10 * angle_factor  # Max 10 degree adjustment
                yaw1 += adjustment
                yaw2 -= adjustment
    
    # Calculate FOV based on distance between cameras
    # As distance increases, decrease FOV to maintain overlap
    base_fov = 90
    
    # If cameras are far apart, use a narrower FOV to maintain overlap
    if distance > 0.1:
        # Map distance to FOV range: close = 90°, far = 70°
        distance_factor = 1.0 / (1.0 + distance * distance_scale)
        fov = max(70, min(90, base_fov * distance_factor + 70))
    else:
        # For very close cameras, use the full FOV
        fov = base_fov
    
    return yaw1, yaw2, fov, fov

def main():
    base_dir = "zind_subset"  # Root directory
    num_pairs = 10  # Number of pairs per room
    distance_scale = 0.1  # Scale factor for distance-based FOV adjustments

    # Process each scene
    for scene in list(sorted(os.listdir(base_dir))):
        scene_dir = os.path.join(base_dir, scene)
        if not os.path.isdir(scene_dir):
            continue

        pano_dir = os.path.join(scene_dir, "panos")
        if not os.path.exists(pano_dir):
            continue

        # Load camera data (rotations and translations)
        camera_data = load_camera_data(scene_dir)

        # List panorama files
        pano_files = [f for f in os.listdir(pano_dir) if f.endswith((".jpg", ".png"))]

        # Group panoramas by room
        rooms = {}
        for pano_file in pano_files:
            match = re.search(r'room_(\d+)', pano_file)
            if match:
                room_id = match.group(1)  # e.g., "01"
                rooms.setdefault(room_id, []).append(pano_file)

        # Process each room
        for room_id, pano_files in rooms.items():
            if len(pano_files) < 2:
                print(f"Skipping room {room_id} in scene {scene}: fewer than 2 panoramas")
                continue

            # Sort and select first two panoramas
            pano_files.sort()
            pano1_file = pano_files[0]
            pano2_file = pano_files[1]
            pano1_path = os.path.join(pano_dir, pano1_file)
            pano2_path = os.path.join(pano_dir, pano2_file)

            # Get camera data (or use default if not found)
            camera1_data = camera_data.get(pano1_file, {"rotation": 0, "translation": [0, 0, 0]})
            camera2_data = camera_data.get(pano2_file, {"rotation": 0, "translation": [0, 0, 0]})
            
            rotation1 = camera1_data.get("rotation", 0)
            rotation2 = camera2_data.get("rotation", 0)
            translation1 = camera1_data.get("translation", [0, 0, 0])
            translation2 = camera2_data.get("translation", [0, 0, 0])
            
            rotation_diff = (rotation2 - rotation1) % 360
            
            # Calculate distance between cameras (for logging)
            pos1 = np.array(translation1[:2])
            pos2 = np.array(translation2[:2])
            distance = np.linalg.norm(pos2 - pos1)
            
            print(f"Cameras for room {room_id} in scene {scene}: "
                  f"rot: {rotation1:.2f}° and {rotation2:.2f}° (diff: {rotation_diff:.2f}°), "
                  f"distance: {distance:.2f}")

            # Load panoramas
            pano1 = cv2.imread(pano1_path)
            pano2 = cv2.imread(pano2_path)
            if pano1 is None or pano2 is None:
                print(f"Error loading panoramas: {pano1_path} or {pano2_path}")
                continue
            pano1 = cv2.cvtColor(pano1, cv2.COLOR_BGR2RGB)
            pano2 = cv2.cvtColor(pano2, cv2.COLOR_BGR2RGB)

            # Create output directory (e.g., "room 01")
            output_dir = os.path.join(scene_dir, "2d_views", f"room {room_id}")
            os.makedirs(output_dir, exist_ok=True)

            # Extract pairs with both cameras rotating together through 360 degrees
            for i in range(num_pairs):
                # Calculate base yaw for even distribution around the panorama (0-360)
                base_yaw = i * (360.0 / num_pairs)
                
                # Calculate optimal viewing parameters - both cameras will look in similar directions
                yaw1, yaw2, fov1, fov2 = calculate_optimal_view_parameters(
                    camera1_data, camera2_data, base_yaw, distance_scale
                )
                
                pair_dir = os.path.join(output_dir, f"pair_{i}")
                os.makedirs(pair_dir, exist_ok=True)

                # Extract views with the calculated viewing parameters
                view1 = equirectangular_to_perspective(pano1, fov=fov1, theta=yaw1, phi=0, width=512, height=512)
                view2 = equirectangular_to_perspective(pano2, fov=fov2, theta=yaw2, phi=0, width=512, height=512)

                # Save the pair
                cv2.imwrite(os.path.join(pair_dir, "view1.jpg"), cv2.cvtColor(view1, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(pair_dir, "view2.jpg"), cv2.cvtColor(view2, cv2.COLOR_RGB2BGR))
                print(f"Saved pair_{i} for room {room_id} in scene {scene} with "
                      f"yaw1={yaw1:.1f}°, yaw2={yaw2:.1f}° and FOV={fov1:.1f}°")

if __name__ == "__main__":
    main()