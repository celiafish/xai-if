import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'tool-data')
VIS_PNG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'visualization-tool-pngs')


def plot_influence_interactive(yref, test_coords, train_coords, influences_all_test):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'height_ratios': [1]})
    plt.subplots_adjust(bottom=0.23, top=0.92)  # Make more space for sliders and buttons
    
    # Add influence type filter buttons at the top
    ax_all = plt.axes([0.1, 0.95, 0.2, 0.04])
    ax_positive = plt.axes([0.35, 0.95, 0.2, 0.04])
    ax_negative = plt.axes([0.6, 0.95, 0.2, 0.04])
    
    btn_all = widgets.Button(ax_all, 'All Influences', hovercolor='lightblue')
    btn_positive = widgets.Button(ax_positive, 'Positive Influences', hovercolor='lightgreen')
    btn_negative = widgets.Button(ax_negative, 'Negative Influences', hovercolor='lightcoral')
    cmap = plt.get_cmap("RdBu_r")

    im0 = ax0.imshow(yref, cmap=cmap)
    im1 = ax1.imshow(yref, cmap=cmap, alpha=0.3)
    fig.colorbar(im0, ax=ax0)
    # Do not add a colorbar for the background image on ax1; only add for the influence scatter
    ax0.set_title("Reference: Click to select location")
    ax1.set_title("Influence: Top N% highlighted")
    ax0.set_xticks([]); ax0.set_yticks([])
    ax1.set_xticks([]); ax1.set_yticks([])
    ax0.set_xlim([0, yref.shape[1]])
    ax0.set_ylim([yref.shape[0], 0])
    ax1.set_xlim([0, yref.shape[1]])
    ax1.set_ylim([yref.shape[0], 0])

    marker = [None]
    influence_scatter = [None]
    # Store the current normalized coordinates
    current_coords = {'x': 0.0, 'y': 0.0}
    # Store the current filter type
    current_filter_type = ['all']  # 'all', 'positive', or 'negative'

    # Add sliders for longitude (x), latitude (y), and top percentage
    axcolor = 'lightgoldenrodyellow'
    ax_longitude = plt.axes([0.15, 0.18, 0.65, 0.03], facecolor=axcolor)
    ax_latitude = plt.axes([0.15, 0.13, 0.65, 0.03], facecolor=axcolor)
    ax_percent = plt.axes([0.15, 0.08, 0.65, 0.03], facecolor=axcolor)
    slider_longitude = widgets.Slider(ax_longitude, 'Longitude (x)', 0.0, 1.0, valinit=0.0)
    slider_latitude = widgets.Slider(ax_latitude, 'Latitude (y)', 0.0, 1.0, valinit=0.0)
    slider_percent = widgets.Slider(ax_percent, 'Top % Influential Points', 10, 100, valinit=100, valstep=10)

    # Small, side-by-side buttons at the right of each slider
    btn_width = 0.025
    btn_height = 0.03
    # Longitude buttons
    ax_btn_x_minus = plt.axes([0.88, 0.18, btn_width, btn_height])
    ax_btn_x_plus = plt.axes([0.905, 0.18, btn_width, btn_height])
    btn_x_minus = widgets.Button(ax_btn_x_minus, '-', hovercolor='0.85')
    btn_x_plus = widgets.Button(ax_btn_x_plus, '+', hovercolor='0.85')
    # Latitude buttons
    ax_btn_y_minus = plt.axes([0.88, 0.13, btn_width, btn_height])
    ax_btn_y_plus = plt.axes([0.905, 0.13, btn_width, btn_height])
    btn_y_minus = widgets.Button(ax_btn_y_minus, '-', hovercolor='0.85')
    btn_y_plus = widgets.Button(ax_btn_y_plus, '+', hovercolor='0.85')
    # Percent buttons
    ax_btn_p_minus = plt.axes([0.88, 0.08, btn_width, btn_height])
    ax_btn_p_plus = plt.axes([0.905, 0.08, btn_width, btn_height])
    btn_p_minus = widgets.Button(ax_btn_p_minus, '-', hovercolor='0.85')
    btn_p_plus = widgets.Button(ax_btn_p_plus, '+', hovercolor='0.85')

    # Store the current percentage
    current_percent = {'value': 100}    

    def update_plots(norm_x, norm_y, percent=None):
        if percent is not None:
            current_percent['value'] = percent
        percent = current_percent['value']
        # Find closest test coordinate
        dists = np.linalg.norm(test_coords - np.array([norm_y, norm_x]), axis=1)
        idx = np.argmin(dists)
        loc = test_coords[idx]
        influence = torch.tensor(influences_all_test[idx])  # shape (N_train,)
        current_coords['x'] = loc[1]
        current_coords['y'] = loc[0]
        # Update marker on ax0
        if marker[0] is not None:
            marker[0].remove()
        marker[0] = ax0.scatter(loc[1] * yref.shape[1], loc[0] * yref.shape[0],
                                s=100, edgecolor='yellow', facecolor='none', linewidth=2)
        # Remove previous influence scatter plot if it exists
        if influence_scatter[0] is not None:
            influence_scatter[0].remove()
            influence_scatter[0] = None
        # Filter influences based on current filter type
        influence_np = influence.numpy()
        
        if current_filter_type[0] == 'positive':
            # Only keep positive influences
            mask = influence_np > 0
            filtered_indices = np.where(mask)[0]
            filtered_influences = influence_np[mask]
            filtered_train_coords = train_coords[filtered_indices]
        elif current_filter_type[0] == 'negative':
            # Only keep negative influences
            mask = influence_np < 0
            filtered_indices = np.where(mask)[0]
            filtered_influences = influence_np[mask]
            filtered_train_coords = train_coords[filtered_indices]
        else:  # 'all'
            # Keep all influences
            filtered_indices = np.arange(len(influence_np))
            filtered_influences = influence_np
            filtered_train_coords = train_coords
        
        # Only show top X% influential training points from filtered set
        if len(filtered_influences) == 0:
            # No points to display
            if influence_scatter[0] is not None:
                influence_scatter[0].remove()
                influence_scatter[0] = None
            fig.canvas.draw_idle()
            return
        
        influence_sorted_indices = np.argsort(filtered_influences)
        n_top = max(1, int((percent / 100.0) * len(filtered_influences)))
        top_idx = influence_sorted_indices[-n_top:]
        
        xs = filtered_train_coords[top_idx, 1] * yref.shape[1]
        ys = filtered_train_coords[top_idx, 0] * yref.shape[0]
        top_influences = filtered_influences[top_idx]
        norm = plt.Normalize(vmin=np.min(top_influences), vmax=np.max(top_influences))
        
        # Use reversed colormap for negative influences (darker at bottom, lighter at top)
        if current_filter_type[0] == 'negative':
            cmap_influence = 'Greens_r'
        else:
            cmap_influence = 'Greens'
        
        influence_scatter[0] = ax1.scatter(xs, ys, s=20, c=top_influences, cmap=cmap_influence, norm=norm)
        # Add or update colorbar for influence
        if not hasattr(update_plots, 'cbar') or update_plots.cbar is None:
            update_plots.cbar = fig.colorbar(influence_scatter[0], ax=ax1, orientation='vertical')
            update_plots.cbar.set_label(f'Influence (Top {percent:.0f}%)')
        else:
            # Update colorbar with new data and colormap
            update_plots.cbar.mappable = influence_scatter[0]
            update_plots.cbar.mappable.set_clim(np.min(top_influences), np.max(top_influences))
            update_plots.cbar.set_label(f'Influence (Top {percent:.0f}%)')
            update_plots.cbar.update_normal(influence_scatter[0])
        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes != ax0:
            return
        x_img, y_img = event.xdata, event.ydata
        norm_x = x_img / yref.shape[1]
        norm_y = y_img / yref.shape[0]
        # Update sliders to match click
        slider_longitude.set_val(norm_x)
        slider_latitude.set_val(norm_y)
        update_plots(norm_x, norm_y)

    def slider_update(val):
        norm_x = slider_longitude.val
        norm_y = slider_latitude.val
        percent = slider_percent.val
        update_plots(norm_x, norm_y, percent)

    def x_minus(event):
        val = slider_longitude.val
        if val > 0.0:
            slider_longitude.set_val(max(0.0, val - 0.01))

    def x_plus(event):
        val = slider_longitude.val
        if val < 1.0:
            slider_longitude.set_val(min(1.0, val + 0.01))

    def y_minus(event):
        val = slider_latitude.val
        if val > 0.0:
            slider_latitude.set_val(max(0.0, val - 0.01))

    def y_plus(event):
        val = slider_latitude.val
        if val < 1.0:
            slider_latitude.set_val(min(1.0, val + 0.01))

    def p_minus(event):
        val = int(slider_percent.val)
        if val > 10:
            slider_percent.set_val(val - 10)

    def p_plus(event):
        val = int(slider_percent.val)
        if val < 100:
            slider_percent.set_val(val + 10)
    
    def update_button_colors():
        """Update button colors to highlight the active filter type."""
        # Reset all buttons to default colors
        btn_all.color = 'lightgray'
        btn_positive.color = 'lightgray'
        btn_negative.color = 'lightgray'
        
        # Highlight the active button
        if current_filter_type[0] == 'all':
            btn_all.color = 'lightblue'
        elif current_filter_type[0] == 'positive':
            btn_positive.color = 'lightgreen'
        elif current_filter_type[0] == 'negative':
            btn_negative.color = 'lightcoral'
    
    def on_all_click(event):
        """Handle all influences button click."""
        current_filter_type[0] = 'all'
        update_button_colors()
        update_plots(slider_longitude.val, slider_latitude.val, slider_percent.val)
        fig.canvas.draw_idle()
    
    def on_positive_click(event):
        """Handle positive influences button click."""
        current_filter_type[0] = 'positive'
        update_button_colors()
        update_plots(slider_longitude.val, slider_latitude.val, slider_percent.val)
        fig.canvas.draw_idle()
    
    def on_negative_click(event):
        """Handle negative influences button click."""
        current_filter_type[0] = 'negative'
        update_button_colors()
        update_plots(slider_longitude.val, slider_latitude.val, slider_percent.val)
        fig.canvas.draw_idle()

    slider_longitude.on_changed(slider_update)
    slider_latitude.on_changed(slider_update)
    slider_percent.on_changed(slider_update)
    btn_x_minus.on_clicked(x_minus)
    btn_x_plus.on_clicked(x_plus)
    btn_y_minus.on_clicked(y_minus)
    btn_y_plus.on_clicked(y_plus)
    btn_p_minus.on_clicked(p_minus)
    btn_p_plus.on_clicked(p_plus)
    btn_all.on_clicked(on_all_click)
    btn_positive.on_clicked(on_positive_click)
    btn_negative.on_clicked(on_negative_click)

    # Initialize button colors
    update_button_colors()

    fig.canvas.mpl_connect('button_press_event', onclick)
    # Initialize with the first point and percent
    update_plots(slider_longitude.val, slider_latitude.val, slider_percent.val)
    
    # Save the plot before showing
    os.makedirs(VIS_PNG_DIR, exist_ok=True)
    output_filename = os.path.join(VIS_PNG_DIR, 'point_influences.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to '{output_filename}'")
    
    plt.show()

# Usage 
def main():
    # Load your data from .npy files
    test_coords = np.load(os.path.join(TOOL_DATA_DIR, 'test_coords.npy'))
    train_coords = np.load(os.path.join(TOOL_DATA_DIR, 'train_coords.npy'))
    yrefs = np.load(os.path.join(TOOL_DATA_DIR, 'yrefs.npy'))
    influences_all_test = np.load(os.path.join(TOOL_DATA_DIR, 'influences_all_test.npy'), allow_pickle=True)

    # print("yrefs.shape:", yrefs.shape)
    # print("yrefs[0].shape:", yrefs[0].shape if yrefs.ndim > 1 else "N/A")
    # print("yrefs[0][0].shape:", yrefs[0][0].shape if yrefs.ndim > 2 else "N/A")

    plot_influence_interactive(
        yrefs,
        test_coords,
        train_coords,
        influences_all_test
    )

if __name__ == '__main__':
    main()