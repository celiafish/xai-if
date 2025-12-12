import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'tool-data')
VIS_PNG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'visualization-tool-pngs')

# plots influence of all regions on the selected region
def plot_region_influence_interactive(yref, regions_data, influence_type="overall"):
    """
    Interactive plot for region-to-region influences.
    
    Args:
        yref: Reference field for background visualization
        regions_data: Dictionary containing region data with influences
    """
    fig = plt.figure(figsize=(18, 12))
    
    # Create subplots: left map, top right map, bottom left map, bottom right map
    ax0 = plt.subplot(2, 2, 1)  # Left map
    ax1 = plt.subplot(2, 2, 2)  # Top right map
    ax2 = plt.subplot(2, 2, 3)  # Bottom left map (total influence)
    ax3 = plt.subplot(2, 2, 4)  # Bottom right map
    
    # Add influence type buttons at the top
    ax_overall = plt.axes([0.1, 0.95, 0.2, 0.04])
    ax_positive = plt.axes([0.35, 0.95, 0.2, 0.04])
    ax_negative = plt.axes([0.6, 0.95, 0.2, 0.04])
    
    btn_overall = widgets.Button(ax_overall, 'Overall Influence', hovercolor='lightblue')
    btn_positive = widgets.Button(ax_positive, 'Positive Influence', hovercolor='lightgreen')
    btn_negative = widgets.Button(ax_negative, 'Negative Influence', hovercolor='lightcoral')
    
    plt.subplots_adjust(bottom=0.15)  # Make space for percentage slider
    cmap = plt.get_cmap("RdBu_r")

    # Display reference field on all plots
    im0 = ax0.imshow(yref, cmap=cmap)
    im1 = ax1.imshow(yref, cmap=cmap)
    im2 = ax2.imshow(yref, cmap=cmap)
    im3 = ax3.imshow(yref, cmap=cmap)
    fig.colorbar(im0, ax=ax0)
    ax0.set_title("Reference: Click to select region")
    ax1.set_title("Influence ON selected region: Top N% highlighted")
    ax2.set_title("Total Influence: Static view")
    ax3.set_title("Influence OF selected region: Top N% highlighted")
    ax0.set_xticks([]); ax0.set_yticks([])
    ax1.set_xticks([]); ax1.set_yticks([])
    ax2.set_xticks([]); ax2.set_yticks([])
    ax3.set_xticks([]); ax3.set_yticks([])
    ax0.set_xlim([0, yref.shape[1]])
    ax0.set_ylim([yref.shape[0], 0])
    ax1.set_xlim([0, yref.shape[1]])
    ax1.set_ylim([yref.shape[0], 0])
    ax2.set_xlim([0, yref.shape[1]])
    ax2.set_ylim([yref.shape[0], 0])
    ax3.set_xlim([0, yref.shape[1]])
    ax3.set_ylim([yref.shape[0], 0])
    
    # Draw grid lines to show 54 regions (6x9) on all plots
    for i in range(1, 6):  # Horizontal lines
        y_pos = i * yref.shape[0] / 6
        ax0.axhline(y=y_pos, color='white', linewidth=1, alpha=0.7)
        ax1.axhline(y=y_pos, color='white', linewidth=1, alpha=0.7)
        ax2.axhline(y=y_pos, color='white', linewidth=1, alpha=0.7)
        ax3.axhline(y=y_pos, color='white', linewidth=1, alpha=0.7)
    
    for j in range(1, 9):  # Vertical lines
        x_pos = j * yref.shape[1] / 9
        ax0.axvline(x=x_pos, color='white', linewidth=1, alpha=0.7)
        ax1.axvline(x=x_pos, color='white', linewidth=1, alpha=0.7)
        ax2.axvline(x=x_pos, color='white', linewidth=1, alpha=0.7)
        ax3.axvline(x=x_pos, color='white', linewidth=1, alpha=0.7)

    # Store current state
    selected_region = [None]
    region_influence_plot = [None]
    region_outgoing_plot = [None]
    total_influence_plot = [None]
    total_influence_cbar = [None]  # Store the colorbar for total influence
    current_percent = {'value': 100}
    current_influence_type = [influence_type]
    current_regions_data = [regions_data]
    
    # Create static total influence visualization
    def create_total_influence_map():
        """Create the static total influence map."""
        total_influence_vis = np.full((yref.shape[0], yref.shape[1]), np.nan)
        total_influences = []
        
        # Collect all total influences
        for region_key, region_data in current_regions_data[0].items():
            total_influences.append(region_data['total_influence'])
        
        # Fill visualization with total influences
        for region_key, region_data in current_regions_data[0].items():
            region_info = region_data['region_info']
            region_x = region_info['col']
            region_y = region_info['row']
            
            # Calculate region boundaries
            min_x = int(region_x * yref.shape[1] / 9)
            max_x = int((region_x + 1) * yref.shape[1] / 9)
            min_y = int(region_y * yref.shape[0] / 6)
            max_y = int((region_y + 1) * yref.shape[0] / 6)
            
            # Fill the entire region with the total influence value
            total_influence_vis[min_y:max_y, min_x:max_x] = region_data['total_influence']
        
        # Plot total influence map
        if len(total_influences) > 0:
            norm = plt.Normalize(vmin=np.min(total_influences), vmax=np.max(total_influences))
            # Choose colormap based on influence type (flip for negative)
            cmap_total = 'Blues_r' if current_influence_type[0] == 'negative' else 'Blues'
            
            if total_influence_plot[0] is None:
                # Create new plot
                total_influence_plot[0] = ax2.imshow(total_influence_vis, cmap=cmap_total, norm=norm, alpha=0.7)
                # Add colorbar for total influence
                total_influence_cbar[0] = fig.colorbar(total_influence_plot[0], ax=ax2, orientation='vertical')
                total_influence_cbar[0].set_label('Total Influence')
            else:
                # Update existing plot
                total_influence_plot[0].set_data(total_influence_vis)
                total_influence_plot[0].set_norm(norm)
                total_influence_plot[0].set_cmap(cmap_total)
                # Update colorbar
                if total_influence_cbar[0] is not None:
                    total_influence_cbar[0].mappable.set_clim(norm.vmin, norm.vmax)
                    total_influence_cbar[0].update_normal(total_influence_plot[0])
        
        return total_influence_plot[0]
    
    # Initialize the static total influence map
    create_total_influence_map()

    # Add percentage slider
    axcolor = 'lightgoldenrodyellow'
    ax_percent = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
    slider_percent = widgets.Slider(ax_percent, 'Top % Influential Regions', 10, 100, valinit=100, valstep=10)

    # Add percentage control buttons
    btn_width = 0.025
    btn_height = 0.03
    ax_btn_p_minus = plt.axes([0.88, 0.05, btn_width, btn_height])
    ax_btn_p_plus = plt.axes([0.905, 0.05, btn_width, btn_height])
    btn_p_minus = widgets.Button(ax_btn_p_minus, '-', hovercolor='0.85')
    btn_p_plus = widgets.Button(ax_btn_p_plus, '+', hovercolor='0.85')

    def update_plots(region_idx, percent=None):
        """Update the plots based on selected region and percentage."""
        if percent is not None:
            current_percent['value'] = percent
        percent = current_percent['value']
        
        if region_idx is None:
            return
            
        # Get region data
        region_key = f'region_{region_idx:02d}'
        if region_key not in current_regions_data[0]:
            print(f"Region {region_idx} not found in data")
            return
            
        region_data = current_regions_data[0][region_key]
        influences = region_data['influences']
        
        # Update marker on ax0 to show selected region
        if selected_region[0] is not None:
            selected_region[0].remove()
        
        # Highlight the selected region with a rectangle
        region_x = region_idx % 9
        region_y = region_idx // 9
        
        # Calculate region boundaries
        min_x = region_x * yref.shape[1] / 9
        max_x = (region_x + 1) * yref.shape[1] / 9
        min_y = region_y * yref.shape[0] / 6
        max_y = (region_y + 1) * yref.shape[0] / 6
        
        # Draw rectangle outline
        width = max_x - min_x
        height = max_y - min_y
        rect = plt.Rectangle((min_x, min_y), width, height, 
                           fill=False, edgecolor='black', linewidth=3)
        selected_region[0] = ax0.add_patch(rect)
        
        # Remove previous influence plots
        if region_influence_plot[0] is not None:
            region_influence_plot[0].remove()
            region_influence_plot[0] = None
        
        if region_outgoing_plot[0] is not None:
            region_outgoing_plot[0].remove()
            region_outgoing_plot[0] = None
        
        # Get top influential regions
        influence_values = list(influences.values())
        influence_keys = list(influences.keys())
        
        # Sort by influence value
        sorted_indices = np.argsort(influence_values)
        n_top = max(1, int((percent / 100.0) * len(influence_values)))
        top_indices = sorted_indices[-n_top:]
        
        # Create influence visualization
        influence_vis = np.full((yref.shape[0], yref.shape[1]), np.nan)
        
        for i, region_key in enumerate(influence_keys):
            if i in top_indices:
                # Get region info
                region_data_inner = current_regions_data[0][region_key]
                region_info = region_data_inner['region_info']
                region_x_inner = region_info['col']
                region_y_inner = region_info['row']
                
                # Calculate region boundaries
                min_x = int(region_x_inner * yref.shape[1] / 9)
                max_x = int((region_x_inner + 1) * yref.shape[1] / 9)
                min_y = int(region_y_inner * yref.shape[0] / 6)
                max_y = int((region_y_inner + 1) * yref.shape[0] / 6)
                
                # Fill the entire region with the influence value
                influence_vis[min_y:max_y, min_x:max_x] = influences[region_key]
        
        # Plot influential regions (incoming influences)
        if len(top_indices) > 0:
            top_influences = [influence_values[i] for i in top_indices]
            norm = plt.Normalize(vmin=np.min(top_influences), vmax=np.max(top_influences))
            cmap_incoming = 'Purples_r' if current_influence_type[0] == 'negative' else 'Purples'
            region_influence_plot[0] = ax1.imshow(influence_vis, cmap=cmap_incoming, norm=norm, alpha=0.7)
            
            # Add or update colorbar for influence
            if not hasattr(update_plots, 'cbar') or update_plots.cbar is None:
                update_plots.cbar = fig.colorbar(region_influence_plot[0], ax=ax1, orientation='vertical')
                update_plots.cbar.set_label(f'Region Influence (Top {percent:.0f}%)')
            else:
                update_plots.cbar.mappable = region_influence_plot[0]
                update_plots.cbar.mappable.set_clim(np.min(top_influences), np.max(top_influences))
                update_plots.cbar.set_label(f'Region Influence (Top {percent:.0f}%)')
                update_plots.cbar.update_normal(region_influence_plot[0])
        
        # Create outgoing influence visualization (influence of selected region on all other regions)
        outgoing_influence_vis = np.full((yref.shape[0], yref.shape[1]), np.nan)
        
        # Get all regions and their influences on the selected region
        all_region_keys = list(current_regions_data[0].keys())
        outgoing_influences = {}
        
        # For each region, get how much the selected region influences it
        for region_key in all_region_keys:
            region_data_inner = current_regions_data[0][region_key]
            region_influences = region_data_inner['influences']
            selected_region_key = f'region_{region_idx:02d}'
            outgoing_influences[region_key] = region_influences[selected_region_key]
        
        # Sort outgoing influences
        outgoing_values = list(outgoing_influences.values())
        outgoing_keys = list(outgoing_influences.keys())
        outgoing_sorted_indices = np.argsort(outgoing_values)
        outgoing_n_top = max(1, int((percent / 100.0) * len(outgoing_values)))
        outgoing_top_indices = outgoing_sorted_indices[-outgoing_n_top:]
        
        # Fill visualization with outgoing influences
        for i, region_key in enumerate(outgoing_keys):
            if i in outgoing_top_indices:
                region_data_inner = current_regions_data[0][region_key]
                region_info = region_data_inner['region_info']
                region_x_inner = region_info['col']
                region_y_inner = region_info['row']
                
                # Calculate region boundaries
                min_x = int(region_x_inner * yref.shape[1] / 9)
                max_x = int((region_x_inner + 1) * yref.shape[1] / 9)
                min_y = int(region_y_inner * yref.shape[0] / 6)
                max_y = int((region_y_inner + 1) * yref.shape[0] / 6)
                
                # Fill the entire region with the influence value
                outgoing_influence_vis[min_y:max_y, min_x:max_x] = outgoing_influences[region_key]
        
        # Plot outgoing influential regions
        if len(outgoing_top_indices) > 0:
            outgoing_top_influences = [outgoing_values[i] for i in outgoing_top_indices]
            outgoing_norm = plt.Normalize(vmin=np.min(outgoing_top_influences), vmax=np.max(outgoing_top_influences))
            cmap_outgoing = 'Greys_r' if current_influence_type[0] == 'negative' else 'Greys'
            region_outgoing_plot[0] = ax3.imshow(outgoing_influence_vis, cmap=cmap_outgoing, norm=outgoing_norm, alpha=0.7)
            
            # Add or update colorbar for outgoing influence
            if not hasattr(update_plots, 'cbar2') or update_plots.cbar2 is None:
                update_plots.cbar2 = fig.colorbar(region_outgoing_plot[0], ax=ax3, orientation='vertical')
                update_plots.cbar2.set_label(f'Outgoing Influence (Top {percent:.0f}%)')
            else:
                update_plots.cbar2.mappable = region_outgoing_plot[0]
                update_plots.cbar2.mappable.set_clim(np.min(outgoing_top_influences), np.max(outgoing_top_influences))
                update_plots.cbar2.set_label(f'Outgoing Influence (Top {percent:.0f}%)')
                update_plots.cbar2.update_normal(region_outgoing_plot[0])
        
        fig.canvas.draw_idle()

    def onclick(event):
        """Handle mouse clicks on the left plot."""
        if event.inaxes != ax0:
            return
        
        x_img, y_img = event.xdata, event.ydata
        
        # Convert click coordinates to region indices
        region_x = int(x_img / (yref.shape[1] / 9))
        region_y = int(y_img / (yref.shape[0] / 6))
        
        # Ensure coordinates are within bounds
        region_x = min(region_x, 8)
        region_y = min(region_y, 5)
        
        # Convert to 1D region index
        region_idx = region_y * 9 + region_x
        
        #print(f"Selected region {region_idx} (Row {region_y}, Col {region_x})")
        update_plots(region_idx)

    def slider_update(val):
        """Handle percentage slider updates."""
        percent = slider_percent.val
        if selected_region[0] is not None:
            # Get current region from the rectangle
            # This is a bit hacky, but we can extract region from the rectangle position
            rect = selected_region[0]
            x_center = rect.get_x() + rect.get_width() / 2
            y_center = rect.get_y() + rect.get_height() / 2
            
            region_x = int(x_center / (yref.shape[1] / 9))
            region_y = int(y_center / (yref.shape[0] / 6))
            region_idx = region_y * 9 + region_x
            
            update_plots(region_idx, percent)

    def p_minus(event):
        """Decrease percentage by 10."""
        val = int(slider_percent.val)
        if val > 10:
            slider_percent.set_val(val - 10)

    def p_plus(event):
        """Increase percentage by 10."""
        val = int(slider_percent.val)
        if val < 100:
            slider_percent.set_val(val + 10)
    
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
    
    def on_overall_click(event):
        """Handle overall influence button click."""
        current_influence_type[0] = "overall"
        current_regions_data[0] = load_influence_data("overall")
        update_button_colors()
        create_total_influence_map()
        if selected_region[0] is not None:
            # Get current region from the rectangle
            rect = selected_region[0]
            x_center = rect.get_x() + rect.get_width() / 2
            y_center = rect.get_y() + rect.get_height() / 2
            region_x = int(x_center / (yref.shape[1] / 9))
            region_y = int(y_center / (yref.shape[0] / 6))
            region_idx = region_y * 9 + region_x
            update_plots(region_idx, current_percent['value'])
        fig.canvas.draw_idle()
    
    def on_positive_click(event):
        """Handle positive influence button click."""
        current_influence_type[0] = "positive"
        current_regions_data[0] = load_influence_data("positive")
        update_button_colors()
        create_total_influence_map()
        if selected_region[0] is not None:
            # Get current region from the rectangle
            rect = selected_region[0]
            x_center = rect.get_x() + rect.get_width() / 2
            y_center = rect.get_y() + rect.get_height() / 2
            region_x = int(x_center / (yref.shape[1] / 9))
            region_y = int(y_center / (yref.shape[0] / 6))
            region_idx = region_y * 9 + region_x
            update_plots(region_idx, current_percent['value'])
        fig.canvas.draw_idle()
    
    def on_negative_click(event):
        """Handle negative influence button click."""
        current_influence_type[0] = "negative"
        current_regions_data[0] = load_influence_data("negative")
        update_button_colors()
        create_total_influence_map()
        if selected_region[0] is not None:
            # Get current region from the rectangle
            rect = selected_region[0]
            x_center = rect.get_x() + rect.get_width() / 2
            y_center = rect.get_y() + rect.get_height() / 2
            region_x = int(x_center / (yref.shape[1] / 9))
            region_y = int(y_center / (yref.shape[0] / 6))
            region_idx = region_y * 9 + region_x
            update_plots(region_idx, current_percent['value'])
        fig.canvas.draw_idle()

    # Connect event handlers
    slider_percent.on_changed(slider_update)
    btn_p_minus.on_clicked(p_minus)
    btn_p_plus.on_clicked(p_plus)
    btn_overall.on_clicked(on_overall_click)
    btn_positive.on_clicked(on_positive_click)
    btn_negative.on_clicked(on_negative_click)
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    # Initialize button colors
    update_button_colors()
    
    # Initialize with region 0
    update_plots(0, 100)
    
    # Save the plot before showing
    os.makedirs(VIS_PNG_DIR, exist_ok=True)
    output_filename = os.path.join(VIS_PNG_DIR, 'region_influences.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to '{output_filename}'")
    
    plt.show()

def main():
    """Main function to load data and create the interactive plot."""
    print("Loading regions data...")
    regions_data = np.load(os.path.join(TOOL_DATA_DIR, 'regions_data.npy'), allow_pickle=True).item()
    yrefs = np.load(os.path.join(TOOL_DATA_DIR, 'yrefs.npy'))
    
    print(f"Loaded {len(regions_data)} regions")
    print(f"Reference field shape: {yrefs.shape}")
    
    # Use the first reference field for visualization
    yref = yrefs[0] if yrefs.ndim > 2 else yrefs
    
    # Print region information
    # print("\nRegion information:")
    # print("=" * 50)
    # for i, (region_key, region_data) in enumerate(regions_data.items()):
    #     if i < 10:  # Show first 10 regions
    #         region_info = region_data['region_info']
    #         influences = region_data['influences']
    #         print(f"{region_key}: Row {region_info['row']}, Col {region_info['col']}, "
    #               f"Train: {region_info['num_train']}, Test: {region_info['num_test']}, "
    #               f"Max influence: {max(influences.values()):.4f}")
    #     elif i == 10:
    #         print(f"... (showing first 10 of {len(regions_data)} regions)")
    #         break
    
    plot_region_influence_interactive(yref, regions_data, "overall")

if __name__ == '__main__':
    main()
