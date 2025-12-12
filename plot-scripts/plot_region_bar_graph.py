import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'tool-data')
VIS_PNG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'visualization-tool-pngs')


def plot_region_bar_graph(regions_data):
    """
    Create an interactive bar graph showing bidirectional region influences.
    
    Args:
        regions_data: Dictionary containing region data with influences
    """
    num_regions = 54
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.subplots_adjust(bottom=0.15, top=0.85)
    
    # Store current selected region
    selected_region = [0]  # Default to region 0
    
    # Store current influence type
    current_influence_type = ['overall']  # 'overall', 'positive', 'negative'
    current_regions_data = [regions_data]
    
    # Store bar collections
    top_bars = [None]
    bottom_bars = [None]
    legend = [None]
    info_text = [None]
    
    # Region indices for x-axis
    region_indices = np.arange(num_regions)
    
    # Add influence type buttons at the top
    ax_overall = plt.axes([0.1, 0.92, 0.2, 0.04])
    ax_positive = plt.axes([0.35, 0.92, 0.2, 0.04])
    ax_negative = plt.axes([0.6, 0.92, 0.2, 0.04])
    
    btn_overall = widgets.Button(ax_overall, 'Overall Influence', hovercolor='lightblue')
    btn_positive = widgets.Button(ax_positive, 'Positive Influence', hovercolor='lightgreen')
    btn_negative = widgets.Button(ax_negative, 'Negative Influence', hovercolor='lightcoral')
    
    def load_influence_data(influence_type):
        """Load the appropriate regions data based on influence type."""
        if influence_type == "overall":
            return np.load(os.path.join(TOOL_DATA_DIR, 'regions_data.npy'), allow_pickle=True).item()
        elif influence_type == "positive":
            return np.load(os.path.join(TOOL_DATA_DIR, 'regions_data_positive.npy'), allow_pickle=True).item()
        elif influence_type == "negative":
            return np.load(os.path.join(TOOL_DATA_DIR, 'regions_data_negative.npy'), allow_pickle=True).item()
        else:
            return current_regions_data[0]
    
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
    
    def update_plot(region_idx):
        """Update the bar graph for the selected region."""
        selected_region[0] = region_idx
        
        # Clear previous bars
        if top_bars[0] is not None:
            for bar in top_bars[0]:
                bar.remove()
        if bottom_bars[0] is not None:
            for bar in bottom_bars[0]:
                bar.remove()
        
        # Remove previous legend
        if legend[0] is not None:
            legend[0].remove()
            legend[0] = None
        
        # Remove previous text annotation
        if info_text[0] is not None:
            info_text[0].remove()
            info_text[0] = None
        
        # Get region key
        selected_region_key = f'region_{region_idx:02d}'
        
        # Prepare data for bars
        # Top bars: Influence OF selected region on all other regions
        # Bottom bars: Influence ON selected region from all other regions
        
        top_influences = []  # Influence of region z on each region i
        bottom_influences = []  # Influence of each region i on region z
        
        for i in range(num_regions):
            region_i_key = f'region_{i:02d}'
            
            # Top: How much does region z influence region i?
            # This is stored in region_i's influences dict under region_z key
            if region_i_key in current_regions_data[0]:
                region_i_influences = current_regions_data[0][region_i_key]['influences']
                top_influences.append(region_i_influences.get(selected_region_key, 0.0))
            else:
                top_influences.append(0.0)
            
            # Bottom: How much does region i influence region z?
            # This is stored in region_z's influences dict under region_i key
            if selected_region_key in current_regions_data[0]:
                region_z_influences = current_regions_data[0][selected_region_key]['influences']
                bottom_influences.append(region_z_influences.get(region_i_key, 0.0))
            else:
                bottom_influences.append(0.0)
        
        top_influences = np.array(top_influences)
        bottom_influences = np.array(bottom_influences)
        
        # Filter based on influence type
        if current_influence_type[0] == 'positive':
            # Only keep positive values
            top_influences = np.maximum(top_influences, 0)
            bottom_influences = np.maximum(bottom_influences, 0)
        elif current_influence_type[0] == 'negative':
            # Only keep negative values (convert to positive for display)
            top_influences = np.minimum(top_influences, 0)
            bottom_influences = np.minimum(bottom_influences, 0)
            # Convert to positive for display (we'll show them going down)
            top_influences = np.abs(top_influences)
            bottom_influences = np.abs(bottom_influences)
        
        # Create bars based on influence type
        if current_influence_type[0] == 'overall':
            # Overall: symmetric around 0
            top_bars[0] = ax.bar(region_indices, top_influences, 
                                color='steelblue', alpha=0.7, 
                                label=f'Influence OF region {region_idx:02d}')
            bottom_bars[0] = ax.bar(region_indices, -bottom_influences, 
                                   color='coral', alpha=0.7,
                                   label=f'Influence ON region {region_idx:02d}')
        elif current_influence_type[0] == 'positive':
            # Positive: top bars go up from 0, bottom bars also go up (but below axis)
            top_bars[0] = ax.bar(region_indices, top_influences, 
                                color='steelblue', alpha=0.7, 
                                label=f'Influence OF region {region_idx:02d}')
            # Bottom bars: positive values but displayed below axis (negative y)
            bottom_bars[0] = ax.bar(region_indices, -bottom_influences, 
                                   color='coral', alpha=0.7,
                                   label=f'Influence ON region {region_idx:02d}')
        else:  # negative
            # Negative: both bars go down from 0
            top_bars[0] = ax.bar(region_indices, top_influences, 
                                color='steelblue', alpha=0.7, 
                                label=f'Influence OF region {region_idx:02d}')
            bottom_bars[0] = ax.bar(region_indices, -bottom_influences, 
                                   color='coral', alpha=0.7,
                                   label=f'Influence ON region {region_idx:02d}')
        
        # Update title
        influence_type_str = current_influence_type[0].capitalize()
        ax.set_title(f'Bidirectional Region Influences ({influence_type_str}) - Selected Region: {region_idx:02d}', 
                    fontsize=14, pad=10)
        
        # Set labels
        ax.set_xlabel('Region Index', fontsize=12)
        ax.set_ylabel('Influence Value', fontsize=12)
        
        # Set x-axis ticks
        ax.set_xticks(region_indices)
        ax.set_xticklabels([f'R{i:02d}' for i in range(num_regions)], 
                          rotation=45, ha='right', fontsize=8)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
        
        # Add legend with only the current bar handles and labels
        handles = [top_bars[0], bottom_bars[0]]
        labels = [f'Influence OF region {region_idx:02d}', 
                 f'Influence ON region {region_idx:02d}']
        legend[0] = ax.legend(handles, labels, loc='upper right', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set y-axis limits based on influence type
        if current_influence_type[0] == 'overall':
            # Symmetric around 0
            max_abs_value = max(np.abs(top_influences).max(), np.abs(bottom_influences).max())
            if max_abs_value > 0:
                ax.set_ylim(-max_abs_value * 1.1, max_abs_value * 1.1)
        elif current_influence_type[0] == 'positive':
            # Y-axis from 0 to max value in both directions (top: 0 to +max, bottom: 0 to -max)
            max_value = max(top_influences.max(), bottom_influences.max())
            if max_value > 0:
                # Range from -max to +max, showing 0 to max in both directions
                ax.set_ylim(-max_value * 1.1, max_value * 1.1)
            else:
                ax.set_ylim(-1, 1)
        else:  # negative
            # Y-axis from 0 to min value in both directions (top: 0 to -|min|, bottom: 0 to +|min|)
            # Since values are already abs() converted, find the max of the absolute values
            max_abs_value = max(top_influences.max(), bottom_influences.max())
            if max_abs_value > 0:
                # Range from -max_abs to +max_abs, showing 0 to |min| in both directions
                ax.set_ylim(-max_abs_value * 1.1, max_abs_value * 1.1)
            else:
                ax.set_ylim(-1, 1)
        
        # Add text annotation for selected region
        info_text[0] = ax.text(0.02, 0.98, f'Selected: Region {region_idx:02d}\n'
                           f'Top bars: Influence OF region {region_idx:02d} on others\n'
                           f'Bottom bars: Influence ON region {region_idx:02d} from others',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        fig.canvas.draw_idle()
    
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
        update_plot(region_idx)
    
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
    
    def on_overall_click(event):
        """Handle overall influence button click."""
        current_influence_type[0] = "overall"
        current_regions_data[0] = load_influence_data("overall")
        update_button_colors()
        update_plot(selected_region[0])
        fig.canvas.draw_idle()
    
    def on_positive_click(event):
        """Handle positive influence button click."""
        current_influence_type[0] = "positive"
        current_regions_data[0] = load_influence_data("positive")
        update_button_colors()
        update_plot(selected_region[0])
        fig.canvas.draw_idle()
    
    def on_negative_click(event):
        """Handle negative influence button click."""
        current_influence_type[0] = "negative"
        current_regions_data[0] = load_influence_data("negative")
        update_button_colors()
        update_plot(selected_region[0])
        fig.canvas.draw_idle()
    
    # Connect event handlers
    slider_region.on_changed(slider_update)
    btn_r_minus.on_clicked(r_minus)
    btn_r_plus.on_clicked(r_plus)
    btn_overall.on_clicked(on_overall_click)
    btn_positive.on_clicked(on_positive_click)
    btn_negative.on_clicked(on_negative_click)
    
    # Initialize button colors
    update_button_colors()
    
    # Initialize plot with region 0
    update_plot(0)
    
    # Save the plot before showing
    os.makedirs(VIS_PNG_DIR, exist_ok=True)
    output_filename = os.path.join(VIS_PNG_DIR, 'region_bar_graph.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to '{output_filename}'")
    
    plt.show()


def main():
    """Main function to load data and create the bar graph."""
    print("Loading regions data...")
    regions_data = np.load(os.path.join(TOOL_DATA_DIR, 'regions_data.npy'), allow_pickle=True).item()
    
    print(f"Loaded {len(regions_data)} regions")
    
    plot_region_bar_graph(regions_data)


if __name__ == '__main__':
    main()

