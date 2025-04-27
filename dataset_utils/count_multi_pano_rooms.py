import os
import re
import argparse
from collections import defaultdict

def analyze_panoramas(dataset_dir):
    """
    Analyze panorama images to find rooms with multiple panoramas.
    
    Parameters:
    - dataset_dir: path to a specific scene directory (e.g., dataset/0000)
    
    Returns:
    - Tuple containing:
        1. Dictionary with multi-pano rooms and their files
        2. Dictionary with all rooms and their files (for complete floor analysis)
    """
    # Path to panos directory
    panos_dir = os.path.join(dataset_dir, 'panos')
    
    if not os.path.exists(panos_dir):
        print("path does not exist")
        return {}, {}
    
    # Dictionary to store room counts and files
    # Key: (floor_id, room_id), Value: list of panorama files
    room_panos = defaultdict(list)
    
    # Regular expression pattern to extract floor, room, and pano IDs
    pattern = r'floor_(\d+)_partial_room_(\d+)_pano_(\d+)\.jpg'
    
    # Iterate through all files in the panos directory
    for filename in os.listdir(panos_dir):
        if filename.endswith('.jpg'):
            match = re.match(pattern, filename)
            if match:
                floor_id, room_id, pano_id = match.groups()
                room_key = (floor_id, room_id)
                room_panos[room_key].append(filename)
    
    # Filter rooms with more than one panorama
    multi_pano_rooms = {k: v for k, v in room_panos.items() if len(v) > 1}
    
    return multi_pano_rooms, room_panos

def find_complete_multi_pano_floors(room_panos):
    """
    Find floors where all rooms have multiple panoramas.
    
    Parameters:
    - room_panos: Dictionary with all rooms and their panorama files
    
    Returns:
    - Dictionary with floors where all rooms have multiple panoramas
    """
    # Group rooms by floor
    floor_rooms = defaultdict(list)
    for (floor_id, room_id) in room_panos.keys():
        floor_rooms[floor_id].append((floor_id, room_id))
    
    # Find floors where all rooms have multiple panoramas
    complete_multi_pano_floors = {}
    for floor_id, rooms in floor_rooms.items():
        all_multi_pano = all(len(room_panos[room_key]) > 1 for room_key in rooms)
        if all_multi_pano:
            complete_multi_pano_floors[floor_id] = {
                room_key: room_panos[room_key]
                for room_key in rooms
            }
    
    return complete_multi_pano_floors

def analyze_all_scenes(dataset_path, verbose=True):
    """
    Analyze all scenes in the dataset directory.
    
    Parameters:
    - dataset_path: path to the main dataset directory
    - verbose: whether to print detailed information
    
    Returns:
    - Tuple containing:
        1. Total number of multi-pano rooms
        2. Total number of complete multi-pano floors
        3. Number of scenes analyzed
        4. Total number of rooms in complete multi-pano floors
    """
    # Get all scene directories (they should be numbered folders)
    scene_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d.isdigit()]
    if verbose:
        print("scene_dirs: ", scene_dirs)
    scene_dirs.sort()  # Sort numerically
    
    total_multi_pano_rooms = 0
    total_complete_multi_pano_floors = 0
    total_rooms_in_multi_pano_floors = 0
    
    if verbose:
        print("\nAnalyzing all scenes in the dataset...")
        print("=" * 70)
    
    for scene_dir in scene_dirs:
        scene_path = os.path.join(dataset_path, scene_dir)
        multi_pano_rooms, all_room_panos = analyze_panoramas(scene_path)
        complete_multi_pano_floors = find_complete_multi_pano_floors(all_room_panos)
        
        if multi_pano_rooms or complete_multi_pano_floors:
            if verbose:
                print(f"\nScene {scene_dir}:")
                print("-" * 50)
            
            if multi_pano_rooms:
                if verbose:
                    print(f"Number of rooms with multiple panoramas: {len(multi_pano_rooms)}")
                    for (floor_id, room_id), pano_files in sorted(multi_pano_rooms.items()):
                        print(f"\n  Floor {floor_id}, Room {room_id}:")
                        print(f"  Number of panoramas: {len(pano_files)}")
                        print("  Panorama files:")
                        for file in sorted(pano_files):
                            print(f"    - {file}")
                
                total_multi_pano_rooms += len(multi_pano_rooms)
            
            if complete_multi_pano_floors:
                if verbose:
                    print("\nFloors with ALL rooms having multiple panoramas:")
                    print("-" * 50)
                    for floor_id, rooms in complete_multi_pano_floors.items():
                        print(f"\n  Floor {floor_id}:")
                        print(f"  Number of rooms: {len(rooms)}")
                        for (_, room_id), pano_files in sorted(rooms.items()):
                            print(f"    Room {room_id}: {len(pano_files)} panoramas")
                
                total_complete_multi_pano_floors += len(complete_multi_pano_floors)
                # Count total rooms in multi-pano floors
                for floor_rooms in complete_multi_pano_floors.values():
                    total_rooms_in_multi_pano_floors += len(floor_rooms)
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"Total number of rooms with multiple panoramas across all scenes: {total_multi_pano_rooms}")
        print(f"Total number of floors where ALL rooms have multiple panoramas: {total_complete_multi_pano_floors}")
        print(f"Total number of rooms in complete multi-pano floors: {total_rooms_in_multi_pano_floors}")
        print(f"Number of scenes analyzed: {len(scene_dirs)}")
    
    return total_multi_pano_rooms, total_complete_multi_pano_floors, len(scene_dirs), total_rooms_in_multi_pano_floors

def main():
    parser = argparse.ArgumentParser(
        description='Analyze panorama images to find rooms and floors with multiple panoramas.'
    )
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        default='dataset',
        help='Path to the main dataset directory (default: dataset)'
    )
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Suppress detailed output and only show final counts'
    )
    
    args = parser.parse_args()
    
    total_rooms, total_floors, total_scenes, total_rooms_in_multi_floors = analyze_all_scenes(
        args.dataset_path, 
        verbose=not args.quiet
    )
    
    if args.quiet:
        print(f"Summary:")
        print(f"- Total multi-pano rooms: {total_rooms}")
        print(f"- Total complete multi-pano floors: {total_floors}")
        print(f"- Total rooms in complete multi-pano floors: {total_rooms_in_multi_floors}")
        print(f"- Total scenes analyzed: {total_scenes}")

if __name__ == "__main__":
    main() 
