import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'tool-data')
VIS_PNG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'visualization-tool-pngs')


def create_region_influence_matrix(regions_data_path=None):
    """
    Create a 54x54 matrix showing region-to-region influences.
    
    Matrix cell (row a, column b) shows the influence of region b on region a.
    
    Args:
        regions_data_path: Path to regions_data.npy file
    
    Returns:
        influence_matrix: 54x54 numpy array
    """
    if regions_data_path is None:
        regions_data_path = os.path.join(TOOL_DATA_DIR, 'regions_data.npy')
    """
    Create a 54x54 matrix showing region-to-region influences.
    
    Matrix cell (row a, column b) shows the influence of region b on region a.
    
    Args:
        regions_data_path: Path to regions_data.npy file
    
    Returns:
        influence_matrix: 54x54 numpy array
    """
    print(f"Loading regions data from '{regions_data_path}'...")
    regions_data = np.load(regions_data_path, allow_pickle=True).item()
    
    num_regions = 54
    influence_matrix = np.zeros((num_regions, num_regions))
    
    # Fill the matrix
    # Row a, Column b = influence of region b on region a
    for a in range(num_regions):
        region_key_a = f'region_{a:02d}'
        if region_key_a not in regions_data:
            print(f"Warning: {region_key_a} not found in data")
            continue
        
        region_a = regions_data[region_key_a]
        influences = region_a.get('influences', {})
        
        for b in range(num_regions):
            region_key_b = f'region_{b:02d}'
            if region_key_b in influences:
                influence_matrix[a, b] = influences[region_key_b]
            else:
                influence_matrix[a, b] = 0.0
    
    return influence_matrix


def plot_region_matrix_interactive(influence_type='overall'):
    """
    Interactive plot of the region influence matrix as a heatmap.
    
    Args:
        influence_type: 'overall', 'positive', or 'negative'
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    # Leave space on the right for buttons, keep plot square
    plt.subplots_adjust(left=0.1, right=0.75, top=0.95, bottom=0.1)
    
    # Store current state
    current_influence_type = [influence_type]
    im = [None]
    cbar = [None]
    
    def load_influence_data(influence_type):
        """Load the appropriate regions data based on influence type."""
        if influence_type == "overall":
            return os.path.join(TOOL_DATA_DIR, 'regions_data.npy')
        elif influence_type == "positive":
            return os.path.join(TOOL_DATA_DIR, 'regions_data_positive.npy')
        elif influence_type == "negative":
            return os.path.join(TOOL_DATA_DIR, 'regions_data_negative.npy')
        else:
            return os.path.join(TOOL_DATA_DIR, 'regions_data.npy')
    
    def update_plot():
        """Update the plot with the current influence type."""
        # Remove previous colorbar if it exists
        if cbar[0] is not None:
            try:
                cbar_ax = cbar[0].ax if hasattr(cbar[0], 'ax') else None
                # Only try to remove if the axes still exists in the figure
                if cbar_ax is not None and cbar_ax in list(fig.axes):
                    cbar[0].remove()
            except Exception:
                # If removal fails for any reason, just continue
                pass
            finally:
                cbar[0] = None
        
        # Clear previous plot elements
        ax.clear()
        
        # Create the influence matrix with current data
        data_path = load_influence_data(current_influence_type[0])
        influence_matrix = create_region_influence_matrix(data_path)
        
        # Use percentile-based normalization to exclude outliers
        # Calculate percentiles to exclude extreme values
        vmin_percentile = np.percentile(influence_matrix, 2)  # 2nd percentile
        vmax_percentile = np.percentile(influence_matrix, 98)  # 98th percentile
        
        # Create normalization that clips outliers
        norm = plt.Normalize(vmin=vmin_percentile, vmax=vmax_percentile)
        
        # Choose colormap based on influence type
        if current_influence_type[0] == 'positive':
            cmap = 'Purples'
        elif current_influence_type[0] == 'negative':
            cmap = 'Purples_r'
        else:  # overall
            cmap = 'RdBu_r'
        
        # Create the heatmap with outlier-clipped normalization
        # Use aspect='equal' to maintain square ratio
        im[0] = ax.imshow(influence_matrix, cmap=cmap, aspect='equal', 
                         interpolation='nearest', norm=norm)
        
        # Add colorbar
        cbar[0] = plt.colorbar(im[0], ax=ax)
        cbar[0].set_label('Influence Value', rotation=270, labelpad=20)
        
        # Set labels
        ax.set_xlabel('Source Region (Influencing)', fontsize=12)
        ax.set_ylabel('Target Region (Influenced)', fontsize=12)
        
        # Update title
        influence_type_str = current_influence_type[0].capitalize()
        ax.set_title(f'Region-to-Region Influence Matrix ({influence_type_str})', 
                    fontsize=14, pad=20)
        
        # Set tick labels (show every 5th region to avoid clutter)
        tick_positions = list(range(0, 54, 5)) + [53]
        tick_labels = [f'R{i:02d}' for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)
        
        # Add grid for better readability
        ax.set_xticks(np.arange(-0.5, 54, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 54, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        fig.canvas.draw_idle()
    
    # Add influence type buttons on the right side
    # Position buttons vertically on the right, below the colorbar
    button_width = 0.15
    button_height = 0.06
    button_spacing = 0.08
    right_start = 0.78
    
    ax_overall = plt.axes([right_start, 0.70, button_width, button_height])
    ax_positive = plt.axes([right_start, 0.60, button_width, button_height])
    ax_negative = plt.axes([right_start, 0.50, button_width, button_height])
    
    btn_overall = widgets.Button(ax_overall, 'Overall Influence', hovercolor='lightblue')
    btn_positive = widgets.Button(ax_positive, 'Positive Influence', hovercolor='lightgreen')
    btn_negative = widgets.Button(ax_negative, 'Negative Influence', hovercolor='lightcoral')
    
    def update_button_colors():
        """Update button colors to highlight the active influence type."""
        # Reset all buttons to default colors
        btn_overall.color = 'lightgray'
        btn_positive.color = 'lightgray'
        btn_negative.color = 'lightgray'
        
        # Highlight the active button
        if current_influence_type[0] == "overall":
            btn_overall.color = 'lightblue'
        elif current_influence_type[0] == "positive":
            btn_positive.color = 'lightgreen'
        elif current_influence_type[0] == "negative":
            btn_negative.color = 'lightcoral'
    
    def on_overall_click(event):
        """Handle overall influence button click."""
        current_influence_type[0] = "overall"
        update_button_colors()
        update_plot()
        fig.canvas.draw_idle()
    
    def on_positive_click(event):
        """Handle positive influence button click."""
        current_influence_type[0] = "positive"
        update_button_colors()
        update_plot()
        fig.canvas.draw_idle()
    
    def on_negative_click(event):
        """Handle negative influence button click."""
        current_influence_type[0] = "negative"
        update_button_colors()
        update_plot()
        fig.canvas.draw_idle()
    
    # Connect button handlers
    btn_overall.on_clicked(on_overall_click)
    btn_positive.on_clicked(on_positive_click)
    btn_negative.on_clicked(on_negative_click)
    
    # Initialize button colors
    update_button_colors()
    
    # Initial plot
    update_plot()
    
    # Save the plot before showing
    os.makedirs(VIS_PNG_DIR, exist_ok=True)
    output_filename = os.path.join(VIS_PNG_DIR, 'region_influence_matrix.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to '{output_filename}'")
    
    plt.show()
    
    return fig, ax


def main():
    """Main function to create and visualize the region influence matrix."""
    # Create the influence matrix for statistics (using overall)
    influence_matrix = create_region_influence_matrix()
    
    # Plot the interactive matrix
    print("\nPlotting interactive influence matrix...")
    plot_region_matrix_interactive('overall')
    
    # Optionally save the matrix
    # output_path = 'region_influence_matrix.npy'
    # print(f"\nSaving matrix to '{output_path}'...")
    # np.save(output_path, influence_matrix)
    # print("Matrix saved successfully!")


if __name__ == '__main__':
    main()

