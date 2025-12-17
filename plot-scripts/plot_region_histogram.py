import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'tool-data')
VIS_PNG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'visualization-tool-pngs')


def plot_region_histogram(regions_data, yref_shape=(192, 288)):
    """
    Create an interactive histogram showing distance distribution from selected region.
    
    Args:
        regions_data: Dictionary containing region data
        yref_shape: Shape of the reference field (height, width) for calculating region centers
    """
    num_regions = 54
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.15, top=0.85)
    
    # Store current selected region
    selected_region = [0]  # Default to region 0
    
    # Store current influence type
    current_influence_type = ['positive']  # 'positive' or 'negative'
    current_regions_data = [regions_data]
    
    # Calculate region grid positions (row, col) for distance calculation
    # This only needs to be done once since region positions don't change
    region_positions = []
    for region_idx in range(num_regions):
        region_key = f'region_{region_idx:02d}'
        if region_key in regions_data:
            region_info = regions_data[region_key]['region_info']
            region_x = region_info['col']
            region_y = region_info['row']
            region_positions.append([region_y, region_x])  # [row, col]
        else:
            region_positions.append([0, 0])
    
    region_positions = np.array(region_positions)
    
    # Store histogram plot
    hist_plot = [None]
    
    # Add influence type buttons at the top
    ax_positive = plt.axes([0.1, 0.92, 0.2, 0.04])
    ax_negative = plt.axes([0.35, 0.92, 0.2, 0.04])
    
    btn_positive = widgets.Button(ax_positive, 'Positive Influence', hovercolor='lightgreen')
    btn_negative = widgets.Button(ax_negative, 'Negative Influence', hovercolor='lightcoral')
    
    def load_influence_data(influence_type):
        """Load the appropriate regions data based on influence type."""
        if influence_type == "positive":
            return np.load(os.path.join(TOOL_DATA_DIR, 'regions_data_positive.npy'), allow_pickle=True).item()
        elif influence_type == "negative":
            return np.load(os.path.join(TOOL_DATA_DIR, 'regions_data_negative.npy'), allow_pickle=True).item()
        else:
            return current_regions_data[0]
    
    def update_button_colors():
        """Update button colors to highlight the active influence type."""
        # Reset all buttons to default colors
        btn_positive.color = 'lightgray'
        btn_negative.color = 'lightgray'
        
        # Highlight the active button
        if current_influence_type[0] == "positive":
            btn_positive.color = 'lightgreen'
        elif current_influence_type[0] == "negative":
            btn_negative.color = 'lightcoral'
    
    def calculate_distances(region_idx):
        """Calculate grid-based distances from selected region to all other regions.
        
        Distance is measured in region hops:
        - 0 = same region (self)
        - 1 = neighboring regions (adjacent horizontally, vertically, or diagonally)
        - 2 = regions 2 hops away
        - etc.
        
        Uses Chebyshev distance (max of row and column differences).
        
        Returns:
            distances: array of distances
            region_indices: array of corresponding region indices
        """
        if region_idx >= len(region_positions):
            return np.array([]), np.array([])
        
        selected_pos = region_positions[region_idx]
        distances = []
        region_indices = []
        
        for i in range(num_regions):
            if i != region_idx:
                other_pos = region_positions[i]
                # Chebyshev distance: max(|row_diff|, |col_diff|)
                # This treats diagonal neighbors as distance 1
                row_diff = abs(selected_pos[0] - other_pos[0])
                col_diff = abs(selected_pos[1] - other_pos[1])
                dist = max(row_diff, col_diff)
                distances.append(dist)
                region_indices.append(i)
        
        return np.array(distances), np.array(region_indices)
    
    def update_histogram(region_idx):
        """Update the histogram for the selected region."""
        selected_region[0] = region_idx
        
        # Clear previous histogram and axes
        ax.clear()
        hist_plot[0] = None
        
        # Calculate distances from selected region to all other regions
        distances, other_region_indices = calculate_distances(region_idx)
        
        if len(distances) == 0:
            influence_type_str = current_influence_type[0].capitalize()
            ax.set_title(f'Distance Histogram - Selected Region: {region_idx:02d} ({influence_type_str} Influence - No data)', 
                        fontsize=14, pad=5)
            fig.canvas.draw_idle()
            return
        
        # Get selected region key
        selected_region_key = f'region_{region_idx:02d}'
        
        # Sum influence values for each distance
        # For each distance from 0 to 8, sum the influence of selected region on regions at that distance
        distance_influence_sums = np.zeros(9)  # 0 to 8
        
        # For each other region, get its distance and influence value
        for i in range(len(distances)):
            dist = int(distances[i])
            if dist <= 8:  # Only consider distances 0-8
                other_region_idx = other_region_indices[i]
                other_region_key = f'region_{other_region_idx:02d}'
                
                # Get influence of selected region on this other region
                # This is stored in other_region's influences dict under selected_region_key
                if other_region_key in current_regions_data[0]:
                    other_region_influences = current_regions_data[0][other_region_key].get('influences', {})
                    influence_value = other_region_influences.get(selected_region_key, 0.0)
                    distance_influence_sums[dist] += influence_value
        
        # For negative influences, convert to absolute values for display
        if current_influence_type[0] == 'negative':
            distance_influence_sums = np.abs(distance_influence_sums)
        
        # Create bar graph
        distances_x = np.arange(9)  # 0 to 8
        hist_plot[0] = ax.bar(distances_x, distance_influence_sums, 
                             color='steelblue', alpha=0.7, 
                             edgecolor='black', linewidth=1.2)
        
        # Set x-axis ticks to integer values (0 to 8)
        ax.set_xticks(distances_x)
        ax.set_xticklabels([str(i) for i in range(9)])
        ax.set_xlim(-0.5, 8.5)  # Fixed x-axis range
        
        # Update title
        influence_type_str = current_influence_type[0].capitalize()
        ax.set_title(f'Distance Histogram - Selected Region: {region_idx:02d} ({influence_type_str} Influence)', 
                    fontsize=14, pad=5)
        
        # Set labels
        ax.set_xlabel('Distance from Selected Region (in regions)', fontsize=12)
        ax.set_ylabel('Sum of Influence Values', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Update y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Force redraw
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    # Add region selection slider
    axcolor = 'lightgoldenrodyellow'
    ax_region = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
    slider_region = widgets.Slider(ax_region, 'Selected Region', 0, num_regions-1, 
                                   valinit=0, valstep=1)
    
    # Add region control buttons
    btn_width = 0.025
    btn_height = 0.03
    ax_btn_r_minus = plt.axes([0.88, 0.05, btn_width, btn_height])
    ax_btn_r_plus = plt.axes([0.905, 0.05, btn_width, btn_height])
    btn_r_minus = widgets.Button(ax_btn_r_minus, '-', hovercolor='0.85')
    btn_r_plus = widgets.Button(ax_btn_r_plus, '+', hovercolor='0.85')
    
    def slider_update(val):
        """Handle region slider updates."""
        region_idx = int(slider_region.val)
        update_histogram(region_idx)
    
    def r_minus(event):
        """Decrease region by 1."""
        val = int(slider_region.val)
        if val > 0:
            slider_region.set_val(val - 1)
    
    def r_plus(event):
        """Increase region by 1."""
        val = int(slider_region.val)
        if val < num_regions - 1:
            slider_region.set_val(val + 1)
    
    def on_positive_click(event):
        """Handle positive influence button click."""
        current_influence_type[0] = "positive"
        current_regions_data[0] = load_influence_data("positive")
        update_button_colors()
        update_histogram(selected_region[0])
        fig.canvas.draw_idle()
    
    def on_negative_click(event):
        """Handle negative influence button click."""
        current_influence_type[0] = "negative"
        current_regions_data[0] = load_influence_data("negative")
        update_button_colors()
        update_histogram(selected_region[0])
        fig.canvas.draw_idle()
    
    # Connect event handlers
    slider_region.on_changed(slider_update)
    btn_r_minus.on_clicked(r_minus)
    btn_r_plus.on_clicked(r_plus)
    btn_positive.on_clicked(on_positive_click)
    btn_negative.on_clicked(on_negative_click)
    
    # Initialize button colors
    update_button_colors()
    
    # Initialize histogram with region 0
    update_histogram(0)
    
    # Save the plot before showing
    os.makedirs(VIS_PNG_DIR, exist_ok=True)
    output_filename = os.path.join(VIS_PNG_DIR, 'region_histogram.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to '{output_filename}'")
    
    plt.show()


def main():
    """Main function to load data and create the histogram."""
    print("Loading regions data (positive influence, default)...")
    regions_data = np.load(os.path.join(TOOL_DATA_DIR, 'regions_data_positive.npy'), allow_pickle=True).item()
    
    print(f"Loaded {len(regions_data)} regions")
    
    # Default yref shape (can be adjusted if needed)
    # GST: (192, 288), SST: (901, 1001)
    yref_shape = (192, 288)
    
    plot_region_histogram(regions_data, yref_shape)


if __name__ == '__main__':
    main()

