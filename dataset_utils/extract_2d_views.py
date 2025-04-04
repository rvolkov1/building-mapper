import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import re
import json
import math
import argparse

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

def load_camera_data_and_room_vertices(scene_dir):
    """
    Load camera data and room vertices from zind_data.json.
    
    Args:
        scene_dir: Path to the scene directory
    
    Returns:
        camera_data: Dictionary mapping pano_file -> {"rotation": degrees, "translation": [x, y, z]}
        room_vertices: Dictionary mapping room_id -> list of [x, y] vertices
    """
    json_path = os.path.join(scene_dir, "zind_data.json")
    camera_data = {}
    room_vertices = {}
    
    if not os.path.exists(json_path):
        print(f"Warning: No zind_data.json found in {scene_dir}")
        return camera_data, room_vertices
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Process merger section for camera data
        merger = data.get("merger", {})
        
        # Process each floor
        for floor_key, floor_data in merger.items():
            floor_num = floor_key.split('_')[-1] if '_' in floor_key else '00'
            
            # Process each room to extract vertices
            for room_key, room_data in floor_data.items():
                if room_key.startswith("complete_room_"):
                    room_id = room_key.split('_')[-1]
                    
                    # Extract room vertices
                    vertices = []
                    if 'vertices' in room_data:
                        for vertex in room_data['vertices']:
                            # Only use x, y coordinates (floor is assumed to be flat)
                            if isinstance(vertex, list) and len(vertex) >= 2:
                                vertices.append([vertex[0], vertex[1]])
                    
                    # Store vertices if found
                    if vertices:
                        room_vertices[room_id] = vertices
                    
                    # Process each partial room for camera data
                    for partial_room_key, partial_room_data in room_data.items():
                        if partial_room_key.startswith("partial_room_"):
                            # Process each panorama
                            for pano_key, pano_data in partial_room_data.items():
                                if pano_key.startswith("pano_"):
                                    # Extract rotation and translation data
                                    floor_plan_trans = pano_data.get("floor_plan_transformation", {})
                                    rotation = floor_plan_trans.get("rotation", 0)
                                    
                                    # Extract translation
                                    translation = floor_plan_trans.get("translation", [0, 0, 0])
                                    if isinstance(translation, dict):
                                        translation = [
                                            translation.get("x", 0),
                                            translation.get("y", 0),
                                            translation.get("z", 0)
                                        ]
                                    elif isinstance(translation, list) and len(translation) == 2:
                                        translation = [translation[0], translation[1], 0]
                                    
                                    image_path = pano_data.get("image_path", "")
                                    if image_path:
                                        pano_file = os.path.basename(image_path)
                                        camera_data[pano_file] = {
                                            "rotation": rotation,
                                            "translation": translation,
                                            "room_id": room_id
                                        }
        
        # If there are no vertices in the merger section, try to find them in the main structure
        if not room_vertices:
            # Try to extract from floors
            for floor_key, floor_data in data.items():
                if floor_key.startswith("floor_"):
                    for room_key, room_data in floor_data.items():
                        if room_key.startswith("room_"):
                            room_id = room_key.split('_')[-1]
                            
                            # Extract room vertices
                            vertices = []
                            if 'vertices' in room_data:
                                for vertex in room_data['vertices']:
                                    if isinstance(vertex, list) and len(vertex) >= 2:
                                        vertices.append([vertex[0], vertex[1]])
                            
                            # Store vertices if found
                            if vertices:
                                room_vertices[room_id] = vertices
        
        print(f"Loaded {len(camera_data)} camera data entries from {json_path}")
        print(f"Loaded {len(room_vertices)} room vertex sets from {json_path}")
        return camera_data, room_vertices
    
    except Exception as e:
        print(f"Error loading zind_data.json: {e}")
        return camera_data, room_vertices

def generate_room_points(room_vertices, num_points=10):
    """
    Generate evenly distributed points within a room from its vertices.
    
    Args:
        room_vertices: List of [x, y] vertices defining the room's boundary
        num_points: Number of points to generate
    
    Returns:
        List of [x, y] points within the room
    """
    if not room_vertices or len(room_vertices) < 3:
        # If no vertices or not enough to form a polygon, return empty list
        return []
    
    # Convert vertices to numpy array
    vertices = np.array(room_vertices)
    
    # Find bounding box of the room
    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)
    
    # Create a polygon object from the room vertices
    from matplotlib.path import Path
    room_polygon = Path(vertices)
    
    # Generate candidate points across the room
    # We'll generate more points than needed and select a subset
    oversample_factor = 5
    points = []
    attempts = 0
    max_attempts = 1000
    
    # Try to find points inside the room
    while len(points) < num_points * oversample_factor and attempts < max_attempts:
        # Generate random points within the bounding box
        x = np.random.uniform(min_x, max_x, num_points)
        y = np.random.uniform(min_y, max_y, num_points)
        candidates = np.column_stack([x, y])
        
        # Check which points are inside the room polygon
        mask = room_polygon.contains_points(candidates)
        
        # Add valid points to the list
        for point in candidates[mask]:
            points.append(point)
        
        attempts += 1
    
    # If we couldn't find enough points inside the room, use the vertices themselves
    if len(points) < num_points:
        print(f"Warning: Could not generate {num_points} points inside the room. Using vertices instead.")
        # Use the vertices and their midpoints
        all_points = list(vertices)
        
        # Add midpoints of edges
        for i in range(len(vertices)):
            midpoint = (vertices[i] + vertices[(i+1) % len(vertices)]) / 2
            all_points.append(midpoint)
        
        # If still not enough, add the centroid
        if len(all_points) < num_points:
            centroid = np.mean(vertices, axis=0)
            all_points.append(centroid)
        
        # Use as many points as available
        points = all_points[:min(num_points, len(all_points))]
    else:
        # If we have many points, ensure they're well distributed by clustering
        from sklearn.cluster import KMeans
        if len(points) > num_points:
            kmeans = KMeans(n_clusters=num_points, random_state=0).fit(points)
            points = kmeans.cluster_centers_
    
    return points[:num_points]

def calculate_camera_angle_to_point(camera_pos, target_point, camera_rotation):
    """
    Calculate the angle needed for a camera to look at a specific point.
    
    Args:
        camera_pos: [x, y, z] position of the camera
        target_point: [x, y] point to look at
        camera_rotation: Camera's rotation in degrees
    
    Returns:
        yaw: Yaw angle in degrees
    """
    # We only use x, y coordinates for yaw calculation
    camera_x, camera_y = camera_pos[:2]
    target_x, target_y = target_point
    
    # Calculate the relative vector from camera to target
    dx = target_x - camera_x
    dy = target_y - camera_y
    
    # Calculate the angle in world coordinates
    world_angle = np.rad2deg(np.arctan2(dy, dx))
    
    # Convert to camera's local angle by subtracting camera's rotation
    yaw = world_angle - camera_rotation
    
    return yaw

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract 2D perspective views from panoramic images.')
    parser.add_argument('--base_dir', type=str, default='zind_subset',
                      help='Root directory containing scene folders (default: zind_subset)')
    parser.add_argument('--num_pairs', type=int, default=10,
                      help='Number of view pairs to generate per room (default: 10)')
    parser.add_argument('--width', type=int, default=512,
                      help='Width of output perspective views (default: 512)')
    parser.add_argument('--height', type=int, default=512,
                      help='Height of output perspective views (default: 512)')
    parser.add_argument('--base_fov', type=float, default=90,
                      help='Base field of view in degrees (default: 90)')
    parser.add_argument('--min_fov', type=float, default=60,
                      help='Minimum field of view in degrees (default: 60)')
    parser.add_argument('--scene', type=str, default=None,
                      help='Process specific scene (optional)')
    
    args = parser.parse_args()

    # Process each scene
    scenes = [args.scene] if args.scene else sorted(os.listdir(args.base_dir))
    for scene in scenes:
        scene_dir = os.path.join(args.base_dir, scene)
        if not os.path.isdir(scene_dir):
            continue

        pano_dir = os.path.join(scene_dir, "panos")
        if not os.path.exists(pano_dir):
            continue

        # Load camera data and room vertices
        camera_data, room_vertices = load_camera_data_and_room_vertices(scene_dir)

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
            camera1_data = camera_data.get(pano1_file, {"rotation": 0, "translation": [0, 0, 0], "room_id": room_id})
            camera2_data = camera_data.get(pano2_file, {"rotation": 0, "translation": [0, 0, 0], "room_id": room_id})
            
            rotation1 = camera1_data.get("rotation", 0)
            rotation2 = camera2_data.get("rotation", 0)
            translation1 = camera1_data.get("translation", [0, 0, 0])
            translation2 = camera2_data.get("translation", [0, 0, 0])
            
            # Calculate distance between cameras (for FOV adjustment and logging)
            pos1 = np.array(translation1[:2])
            pos2 = np.array(translation2[:2])
            distance = np.linalg.norm(pos2 - pos1)
            
            print(f"Cameras for room {room_id} in scene {scene}: "
                  f"rot: {rotation1:.2f}° and {rotation2:.2f}°, "
                  f"distance: {distance:.2f}m")

            # Generate room points to look at
            room_points = []
            
            # Try to get vertices for this room
            vertices = room_vertices.get(room_id, [])
            
            if vertices:
                # Generate room points based on room geometry
                room_points = generate_room_points(vertices, args.num_pairs)
                print(f"Generated {len(room_points)} points from room vertices")
            
            # If no vertices or point generation failed, create a fallback grid of points
            if not room_points:
                print(f"No room vertices found for room {room_id}, generating surrogate points")
                
                # Calculate midpoint between cameras as reference
                midpoint = (pos1 + pos2) / 2
                
                # Create a circular arrangement of points around the midpoint
                radius = max(2.0, distance * 2)  # Ensure points are reasonably distant
                angles = np.linspace(0, 2*np.pi, args.num_pairs, endpoint=False)
                
                room_points = []
                for angle in angles:
                    x = midpoint[0] + radius * np.cos(angle)
                    y = midpoint[1] + radius * np.sin(angle)
                    room_points.append([x, y])

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

            # Extract pairs with both cameras looking at the same room point
            for i, target_point in enumerate(room_points[:args.num_pairs]):
                # Calculate yaw angles for cameras to look at the target point
                yaw1 = calculate_camera_angle_to_point(translation1, target_point, rotation1)
                yaw2 = calculate_camera_angle_to_point(translation2, target_point, rotation2)
                
                # Calculate distance from each camera to the target point
                dist1 = np.linalg.norm(np.array(target_point) - np.array(translation1[:2]))
                dist2 = np.linalg.norm(np.array(target_point) - np.array(translation2[:2]))
                
                # Adjust field of view based on distance to the target
                # Closer points need wider FOV, farther points need narrower FOV
                if max(dist1, dist2) > 2.0:
                    # For distant targets, reduce FOV to zoom in
                    fov_factor = min(1.0, 2.0 / max(dist1, dist2))
                    fov = max(args.min_fov, args.base_fov * fov_factor)
                else:
                    fov = args.base_fov
                
                # Create directory for this pair
                pair_dir = os.path.join(output_dir, f"pair_{i}")
                os.makedirs(pair_dir, exist_ok=True)

                # Extract views with cameras looking at the target point
                view1 = equirectangular_to_perspective(pano1, fov=fov, theta=yaw1, phi=0, 
                                                     width=args.width, height=args.height)
                view2 = equirectangular_to_perspective(pano2, fov=fov, theta=yaw2, phi=0, 
                                                     width=args.width, height=args.height)

                # Save the pair
                cv2.imwrite(os.path.join(pair_dir, "view1.jpg"), cv2.cvtColor(view1, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(pair_dir, "view2.jpg"), cv2.cvtColor(view2, cv2.COLOR_RGB2BGR))
                
                # Save target point coordinates for reference
                with open(os.path.join(pair_dir, "target_info.txt"), "w") as f:
                    f.write(f"Target point: [{target_point[0]:.2f}, {target_point[1]:.2f}]\n")
                    f.write(f"Camera 1 position: [{translation1[0]:.2f}, {translation1[1]:.2f}], rotation: {rotation1:.2f}°\n")
                    f.write(f"Camera 2 position: [{translation2[0]:.2f}, {translation2[1]:.2f}], rotation: {rotation2:.2f}°\n")
                    f.write(f"Yaw angles: {yaw1:.2f}° and {yaw2:.2f}°\n")
                    f.write(f"FOV: {fov:.1f}°\n")
                
                print(f"Saved pair_{i} for room {room_id} in scene {scene} with "
                      f"yaw1={yaw1:.1f}°, yaw2={yaw2:.1f}° and FOV={fov:.1f}°")

if __name__ == "__main__":
    main()