import os
import json
import shutil
import re
import argparse
from collections import defaultdict
from count_multi_pano_rooms import analyze_panoramas, find_complete_multi_pano_floors

def get_rooms_by_floor(scene_path):
    """Get all rooms grouped by floor in a scene from the panos directory."""
    panos_dir = os.path.join(scene_path, 'panos')
    if not os.path.exists(panos_dir):
        return {}
    
    rooms_by_floor = defaultdict(set)
    pattern = r'floor_(\d+)_partial_room_(\d+)_pano_(\d+)\.jpg'
    
    for filename in os.listdir(panos_dir):
        if filename.endswith('.jpg'):
            match = re.match(pattern, filename)
            if match:
                floor_id, room_id, _ = match.groups()
                rooms_by_floor[floor_id].add((floor_id, room_id))
    
    return dict(rooms_by_floor)

def organize_multi_pano_rooms(dataset_path, output_dir, quiet=False):
    """
    Create symbolic links for scenes that have at least one floor where every room
    has multiple panoramas. For each scene, only include floors where all rooms
    have multiple panoramas, excluding other floors entirely.
    
    Args:
        dataset_path: Path to the main dataset directory
        output_dir: Directory to store the organized data
        quiet: Whether to suppress detailed output
    """
    # Convert paths to absolute paths
    dataset_path = os.path.abspath(dataset_path)
    output_dir = os.path.abspath(output_dir)
    
    if not quiet:
        print(f"Using dataset path: {dataset_path}")
        print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store scene information
    scene_index = {}
    
    # Get all scene directories
    scene_dirs = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d)) and d.isdigit()]
    scene_dirs.sort()
    
    total_scenes = 0
    total_floors = 0
    
    for scene_dir in scene_dirs:
        scene_path = os.path.join(dataset_path, scene_dir)
        multi_pano_rooms, all_room_panos = analyze_panoramas(scene_path)
        rooms_by_floor = get_rooms_by_floor(scene_path)
        complete_multi_pano_floors = find_complete_multi_pano_floors(all_room_panos)
        
        # Check if at least one floor has all rooms with multiple panoramas
        if complete_multi_pano_floors:
            if not quiet:
                print(f"\nProcessing scene {scene_dir}...")
                print(f"  Found {len(complete_multi_pano_floors)} floor(s) with all rooms having multiple panoramas")
            
            scene_output_dir = os.path.join(output_dir, scene_dir)
            os.makedirs(scene_output_dir, exist_ok=True)
            
            # Create the floor_plans and zind_data.json symlinks
            # These files are common to the scene and not specific to floors
            src_floor_plans = os.path.join(scene_path, 'floor_plans')
            dst_floor_plans = os.path.join(scene_output_dir, 'floor_plans')
            src_metadata = os.path.join(scene_path, 'zind_data.json')
            dst_metadata = os.path.join(scene_output_dir, 'zind_data.json')
            
            # Create floor_plans symlink
            if os.path.exists(src_floor_plans):
                if os.path.exists(dst_floor_plans):
                    if os.path.islink(dst_floor_plans):
                        os.unlink(dst_floor_plans)
                    else:
                        shutil.rmtree(dst_floor_plans)
                os.symlink(src_floor_plans, dst_floor_plans, target_is_directory=True)
                if not quiet:
                    print(f"  Created symlink for floor_plans")
                    
            # Create zind_data.json symlink
            if os.path.exists(src_metadata):
                if os.path.exists(dst_metadata):
                    os.unlink(dst_metadata)
                os.symlink(src_metadata, dst_metadata)
                if not quiet:
                    print(f"  Created symlink for zind_data.json")
            
            # Create a new panos directory that will only contain the qualifying floors
            src_panos_dir = os.path.join(scene_path, 'panos')
            dst_panos_dir = os.path.join(scene_output_dir, 'panos')
            
            if os.path.exists(dst_panos_dir):
                if os.path.islink(dst_panos_dir):
                    os.unlink(dst_panos_dir)
                else:
                    shutil.rmtree(dst_panos_dir)
            
            # Create new panos directory
            os.makedirs(dst_panos_dir, exist_ok=True)
            
            # Only copy the panorama files from floors where all rooms have multiple panoramas
            if os.path.exists(src_panos_dir):
                num_files_copied = 0
                for filename in os.listdir(src_panos_dir):
                    if filename.endswith('.jpg'):
                        # Extract floor ID to check if this file belongs to a complete multi-pano floor
                        pattern = r'floor_(\d+)_partial_room_(\d+)_pano_(\d+)\.jpg'
                        match = re.match(pattern, filename)
                        if match:
                            floor_id = match.group(1)
                            # Only create symlinks for files from complete multi-pano floors
                            if floor_id in complete_multi_pano_floors:
                                src_file = os.path.join(src_panos_dir, filename)
                                dst_file = os.path.join(dst_panos_dir, filename)
                                os.symlink(src_file, dst_file)
                                num_files_copied += 1
                if not quiet:
                    print(f"  Created {num_files_copied} symlinks for panorama files from {len(complete_multi_pano_floors)} floors")
            
            # Store scene information
            scene_info = {
                "total_floors": len(rooms_by_floor),
                "included_floors": len(complete_multi_pano_floors),
                "floors": {},
                "source_path": os.path.abspath(scene_path)
            }
            
            # Store information only for floors where all rooms have multiple panoramas
            for floor_id in complete_multi_pano_floors:
                floor_rooms = {
                    room_key: room_files
                    for room_key, room_files in all_room_panos.items()
                    if room_key[0] == floor_id
                }
                
                floor_info = {
                    "total_rooms": len(floor_rooms),
                    "rooms": {}
                }
                
                for (_, room_id), pano_files in sorted(floor_rooms.items()):
                    room_key = f"room_{room_id}"
                    floor_info["rooms"][room_key] = {
                        "room_id": room_id,
                        "num_panoramas": len(pano_files),
                        "panoramas": sorted(pano_files)
                    }
                
                scene_info["floors"][f"floor_{floor_id}"] = floor_info
            
            scene_index[scene_dir] = scene_info
            total_scenes += 1
            total_floors += len(complete_multi_pano_floors)
    
    # Save the index file
    index_path = os.path.join(output_dir, "scene_index.json")
    with open(index_path, 'w') as f:
        json.dump({
            "total_scenes": total_scenes,
            "total_included_floors": total_floors,
            "scenes": scene_index,
            "dataset_path": dataset_path,
            "output_path": output_dir
        }, f, indent=2)
    
    if not quiet:
        print(f"\nOrganization complete!")
        print(f"- Found {total_scenes} scenes with at least one complete multi-pano floor")
        print(f"- Total floors with all rooms having multiple panoramas: {total_floors}")
        print(f"- Index file created at: {index_path}")
        print(f"- Organized data created in: {output_dir}")
        print(f"- Only included panoramas from floors where all rooms have multiple images")
    else:
        print(f"Organized {total_scenes} scenes with {total_floors} complete multi-pano floors in {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Organize scenes that have at least one floor where all rooms have multiple panoramas, '
                    'excluding other floors entirely.'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='dataset',
        help='Path to the main dataset directory (default: dataset)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='zind_subset',
        help='Directory to store the organized data (default: zind_subset)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    organize_multi_pano_rooms(
        args.dataset_path,
        args.output_dir,
        quiet=args.quiet
    )

if __name__ == "__main__":
    main() 