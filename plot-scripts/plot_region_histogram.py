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
    
    # Store current percentage filter
    current_percent = ['100%']  # '10%', '50%', '100%'
    
    # Calculate region grid positions (row, col) for distance calculation
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
    
    # Add percentage filter buttons at the top
    ax_top10 = plt.axes([0.1, 0.92, 0.15, 0.04])
    ax_top50 = plt.axes([0.3, 0.92, 0.15, 0.04])
    ax_100 = plt.axes([0.5, 0.92, 0.15, 0.04])
    
    btn_top10 = widgets.Button(ax_top10, 'Top 10%', hovercolor='lightblue')
    btn_top50 = widgets.Button(ax_top50, 'Top 50%', hovercolor='lightgreen')
    btn_100 = widgets.Button(ax_100, '100%', hovercolor='lightcoral')
    
    def update_button_colors():
        """Update button colors to highlight the active percentage."""
        # Reset all buttons to default colors
        btn_top10.color = 'lightgray'
        btn_top50.color = 'lightgray'
        btn_100.color = 'lightgray'
        
        # Highlight the active button
        if current_percent[0] == '10%':
            btn_top10.color = 'lightblue'
        elif current_percent[0] == '50%':
            btn_top50.color = 'lightgreen'
        elif current_percent[0] == '100%':
            btn_100.color = 'lightcoral'
    
    def calculate_distances(region_idx):
        """Calculate grid-based distances from selected region to all other regions.
        
        Distance is measured in region hops:
        - 0 = same region (self)
        - 1 = neighboring regions (adjacent horizontally, vertically, or diagonally)
        - 2 = regions 2 hops away
        - etc.
        
        Uses Chebyshev distance (max of row and column differences).
        """
        if region_idx >= len(region_positions):
            return np.array([])
        
        selected_pos = region_positions[region_idx]
        distances = []
        
        for i in range(num_regions):
            if i != region_idx:
                other_pos = region_positions[i]
                # Chebyshev distance: max(|row_diff|, |col_diff|)
                # This treats diagonal neighbors as distance 1
                row_diff = abs(selected_pos[0] - other_pos[0])
                col_diff = abs(selected_pos[1] - other_pos[1])
                dist = max(row_diff, col_diff)
                distances.append(dist)
        
        return np.array(distances)
    
    def update_histogram(region_idx):
        """Update the histogram for the selected region."""
        selected_region[0] = region_idx
        
        # Clear previous histogram and axes
        ax.clear()
        hist_plot[0] = None
        
        # Calculate distances from selected region to all other regions
        distances = calculate_distances(region_idx)
        
        if len(distances) == 0:
            ax.set_title(f'Distance Histogram - Selected Region: {region_idx:02d} (No data)', 
                        fontsize=14, pad=5)
            fig.canvas.draw_idle()
            return
        
        # Determine how many regions to include based on percentage
        total_regions = len(distances)
        if current_percent[0] == '10%':
            n_regions = max(1, int(0.1 * total_regions))
        elif current_percent[0] == '50%':
            n_regions = max(1, int(0.5 * total_regions))
        else:  # 100%
            n_regions = total_regions
        
        # Sort distances and take top N (closest regions)
        sorted_indices = np.argsort(distances)
        selected_distances = distances[sorted_indices[:n_regions]]
        
        # Create histogram
        if len(selected_distances) > 0:
            # Use integer bins since distances are discrete (0, 1, 2, 3, ...)
            max_dist = int(np.max(selected_distances))
            # Create bins for each integer distance value
            bins = np.arange(max_dist + 2) - 0.5  # Shift by 0.5 to center bins on integers
            
            hist_plot[0], bins, _ = ax.hist(selected_distances, bins=bins, 
                                            color='steelblue', alpha=0.7, 
                                            edgecolor='black', linewidth=1.2)
            
            # Set x-axis ticks to integer values
            ax.set_xticks(np.arange(max_dist + 1))
            ax.set_xticklabels([str(i) for i in range(max_dist + 1)])
            
            # Update title
            percent_str = current_percent[0]
            ax.set_title(f'Distance Histogram - Selected Region: {region_idx:02d} ({percent_str} regions)', 
                        fontsize=14, pad=5)
        else:
            ax.set_title(f'Distance Histogram - Selected Region: {region_idx:02d} (No data)', 
                        fontsize=14, pad=5)
        
        # Set labels
        ax.set_xlabel('Distance from Selected Region (in regions)', fontsize=12)
        ax.set_ylabel('Number of Regions', fontsize=12)
        
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
    
    def on_top10_click(event):
        """Handle Top 10% button click."""
        current_percent[0] = '10%'
        update_button_colors()
        update_histogram(selected_region[0])
        fig.canvas.draw_idle()
    
    def on_top50_click(event):
        """Handle Top 50% button click."""
        current_percent[0] = '50%'
        update_button_colors()
        update_histogram(selected_region[0])
        fig.canvas.draw_idle()
    
    def on_100_click(event):
        """Handle 100% button click."""
        current_percent[0] = '100%'
        update_button_colors()
        update_histogram(selected_region[0])
        fig.canvas.draw_idle()
    
    # Connect event handlers
    slider_region.on_changed(slider_update)
    btn_r_minus.on_clicked(r_minus)
    btn_r_plus.on_clicked(r_plus)
    btn_top10.on_clicked(on_top10_click)
    btn_top50.on_clicked(on_top50_click)
    btn_100.on_clicked(on_100_click)
    
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
    print("Loading regions data...")
    regions_data = np.load(os.path.join(TOOL_DATA_DIR, 'regions_data.npy'), allow_pickle=True).item()
    
    print(f"Loaded {len(regions_data)} regions")
    
    # Default yref shape (can be adjusted if needed)
    # GST: (192, 288), SST: (901, 1001)
    yref_shape = (192, 288)
    
    plot_region_histogram(regions_data, yref_shape)


if __name__ == '__main__':
    main()

