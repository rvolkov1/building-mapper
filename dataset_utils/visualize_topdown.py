import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
import re
import argparse

def find_pano_data(data, pano_id):
    """Find the panorama data in the nested JSON structure"""
    for floor_id, floor in data['merger'].items():
        for complete_room_id, complete_room in floor.items():
            for partial_room_id, partial_room in complete_room.items():
                for pano, pano_info in partial_room.items():
                    if isinstance(pano_info, dict) and pano == f"pano_{pano_id}":
                        return pano_info, floor_id, complete_room_id, partial_room_id
    return None, None, None, None

def visualize_room_top_down(json_path, pano_id, output_dir=None):
    """
    Visualize a room layout with WDO as a top-down view.
    
    Parameters:
    - json_path: Path to the zind_data.json file
    - pano_id: ID of the panorama to visualize
    - output_dir: Directory to save the output visualization
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Find the specified panorama data
    pano_data, floor_id, complete_room_id, partial_room_id = find_pano_data(data, pano_id)
    
    if not pano_data:
        print(f"Panorama ID {pano_id} not found in the data.")
        return
    
    # Get the original image path to extract info
    image_path = pano_data.get('image_path', '')
    filename = os.path.basename(image_path)
    print(f"Found panorama: {filename}")
    
    # Get the layout data (use complete layout)
    layout = pano_data['layout_complete']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot room outline
    vertices = np.array(layout['vertices'])
    # Close the polygon by appending the first vertex
    vertices_closed = np.vstack([vertices, vertices[0]])
    ax.plot(vertices_closed[:, 0], vertices_closed[:, 1], 'k-', linewidth=2, label='Room outline')
    
    # Fill the room with a light color
    room_poly = Polygon(vertices, alpha=0.2, color='gray')
    ax.add_patch(room_poly)
    
    # Process WDO elements in triplets
    wdo_types = {
        'doors': {'color': 'red', 'marker': 's', 'label': 'Doors', 'prefix': 'D'},
        'windows': {'color': 'green', 'marker': '^', 'label': 'Windows', 'prefix': 'W'},
        'openings': {'color': 'blue', 'marker': 'o', 'label': 'Openings', 'prefix': 'O'}
    }
    
    for wdo_type, style in wdo_types.items():
        if wdo_type in layout and layout[wdo_type]:
            wdo_points = layout[wdo_type]
            
            # Handle case where the data isn't divisible by 3
            if len(wdo_points) % 3 != 0:
                print(f"Warning: {wdo_type} data not in triplet format (length: {len(wdo_points)})")
                continue

            # Process triplets (each WDO element has 3 entries)
            num_elements = len(wdo_points) // 3
            
            # Plot each element
            for i in range(num_elements):
                # Get the left and right boundary points (first two points in triplet)
                left_point = np.array(wdo_points[i * 3])
                right_point = np.array(wdo_points[i * 3 + 1])
                
                # Draw a line connecting left and right boundary
                ax.plot([left_point[0], right_point[0]], 
                        [left_point[1], right_point[1]], 
                        color=style['color'], linewidth=3, 
                        label=style['label'] if i == 0 else "")
                
                # Add markers at endpoints
                ax.scatter([left_point[0], right_point[0]], 
                           [left_point[1], right_point[1]], 
                           color=style['color'], marker=style['marker'], s=80)
                
                # Add label at the center of the line
                center_x = (left_point[0] + right_point[0]) / 2
                center_y = (left_point[1] + right_point[1]) / 2
                ax.text(center_x, center_y, f"{style['prefix']}{i+1}", 
                        fontsize=9, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Extract room info from filename
    match = re.search(r'floor_(\d+)_partial_room_(\d+)_pano_(\d+)', filename)
    if match:
        floor_num, room_num, pano_num = match.groups()
        title = f"Floor {floor_num}, Room {room_num}, Pano {pano_num} - Top-down View"
    else:
        title = f"Floor {floor_id}, Room {complete_room_id}_{partial_room_id}, Pano {pano_id} - Top-down View"
    
    # Add title and legend
    ax.set_title(title, fontsize=14)
    
    # Handle legend with duplicates removed
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    
    # Compute a nice margin
    x_min, y_min = vertices.min(axis=0) - 0.5
    x_max, y_max = vertices.max(axis=0) + 0.5
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add room label
    room_label = pano_data.get('label', 'Unknown')
    ax.text(0.02, 0.98, f"Room type: {room_label}", transform=ax.transAxes, 
           fontsize=12, va='top', ha='left',
           bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))
    
    # Add WDO counts
    wdo_counts = []
    for wdo_type in wdo_types.keys():
        if wdo_type in layout and layout[wdo_type]:
            count = len(layout[wdo_type]) // 3
            wdo_counts.append(f"{count} {wdo_type}")
    
    wdo_text = ", ".join(wdo_counts)
    ax.text(0.02, 0.93, f"WDO elements: {wdo_text}", transform=ax.transAxes, 
           fontsize=10, va='top', ha='left',
           bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))
    
    # Display and save the figure
    if output_dir is None:
        output_dir = os.path.dirname(json_path)
    
    output_path = os.path.join(output_dir, f'top_down_view_pano_{pano_id}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Saved top-down view to {output_path}")
    
    # Print statistics
    print(f"\nRoom Statistics:")
    print(f"Room type: {room_label}")
    print(f"Number of vertices: {len(layout['vertices'])}")
    
    for wdo_type in wdo_types.keys():
        if wdo_type in layout:
            count = len(layout[wdo_type]) // 3
            print(f"Number of {wdo_type}: {count}")
    
    plt.tight_layout()
    
    return fig

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize a room layout with WDO as a top-down view.')
    parser.add_argument('json_path', type=str, help='Path to the zind_data.json file')
    parser.add_argument('pano_id', type=str, help='ID of the panorama to visualize')
    parser.add_argument('--output-dir', type=str, help='Directory to save the output visualization (optional)')
    
    args = parser.parse_args()
    
    fig = visualize_room_top_down(args.json_path, args.pano_id, args.output_dir)
    plt.show() 