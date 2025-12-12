import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'tool-data')

# Load data
test_coords = np.load(os.path.join(TOOL_DATA_DIR, 'test_coords.npy'))
train_coords = np.load(os.path.join(TOOL_DATA_DIR, 'train_coords.npy'))
yrefs = np.load(os.path.join(TOOL_DATA_DIR, 'yrefs.npy'))  # shape (192, 288)
influences_all = np.load(os.path.join(TOOL_DATA_DIR, 'index_15533_influences.npy'), allow_pickle=True)  # shape (1024, 2765)
global_influence_all = np.load(os.path.join(TOOL_DATA_DIR, 'index_15533_global_influence.npy'), allow_pickle=True)  # shape (2765,)

# --- Interactive plotting function ---
def plot_influence_over_time(yrefs, test_coords, train_coords, influences_all, global_influence_all):
    num_timestamps = influences_all.shape[0]
    n_train = train_coords.shape[0]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'height_ratios': [1]})
    # Leave extra space at the very top for the buttons + metrics
    plt.subplots_adjust(bottom=0.18, top=0.86)
    cmap = plt.get_cmap("RdBu_r")

    im0 = ax0.imshow(yrefs, cmap=cmap)
    im1 = ax1.imshow(yrefs, cmap=cmap, alpha=0.3)
    fig.colorbar(im0, ax=ax0)
    ax0.set_title("Reference: Test index 15533 (yellow)")
    ax1.set_title("Influence: Top N% highlighted")
    ax0.set_xticks([]); ax0.set_yticks([])
    ax1.set_xticks([]); ax1.set_yticks([])
    ax0.set_xlim([0, yrefs.shape[1]])
    ax0.set_ylim([yrefs.shape[0], 0])
    ax1.set_xlim([0, yrefs.shape[1]])
    ax1.set_ylim([yrefs.shape[0], 0])

    marker = [None]
    influence_scatter = [None]
    metrics_text = [None]
    
    # Calculate overall dataset metrics (once, not per timestamp)
    all_influences = influences_all.flatten()  # Flatten all influences across all timestamps
    overall_influence = np.sum(all_influences)
    positive_influence = np.sum(all_influences[all_influences > 0])
    negative_influence = np.sum(all_influences[all_influences < 0])
    
    # Create compact metrics attached to each button's axes so placement scales with screen size
    # We will add the text just below each button using its axes transform

    # --- Influence type filter buttons (All / Positive / Negative / Global) ---
    current_filter_type = ['all']  # 'all' | 'positive' | 'negative' | 'global'
    # Reserve a thin row at the very top for buttons
    ax_all    = plt.axes([0.05, 0.89, 0.2, 0.04])
    ax_pos    = plt.axes([0.30, 0.89, 0.2, 0.04])
    ax_neg    = plt.axes([0.55, 0.89, 0.2, 0.04])
    ax_global = plt.axes([0.80, 0.89, 0.15, 0.04])
    btn_all = widgets.Button(ax_all, 'All Influences', hovercolor='lightblue')
    btn_pos = widgets.Button(ax_pos, 'Positive Influences', hovercolor='lightgreen')
    btn_neg = widgets.Button(ax_neg, 'Negative Influences', hovercolor='lightcoral')
    btn_global = widgets.Button(ax_global, 'Global Influence', hovercolor='0.85')

    def update_button_colors():
        btn_all.color = 'lightgray'
        btn_pos.color = 'lightgray'
        btn_neg.color = 'lightgray'
        btn_global.color = 'lightgray'
        if current_filter_type[0] == 'all':
            btn_all.color = 'lightblue'
        elif current_filter_type[0] == 'positive':
            btn_pos.color = 'lightgreen'
        elif current_filter_type[0] == 'negative':
            btn_neg.color = 'lightcoral'
        elif current_filter_type[0] == 'global':
            btn_global.color = '0.75'
    update_button_colors()

    # Metrics placed relative to each button (above the buttons now)
    # y > 1 positions the text above the button; clip_on=False avoids clipping
    ax_all.text(0.5, 1.25, f"Overall: {overall_influence:.4f}", transform=ax_all.transAxes,
                ha='center', va='bottom', fontsize=10, clip_on=False)
    ax_pos.text(0.5, 1.25, f"Positive: {positive_influence:.4f}", transform=ax_pos.transAxes,
                ha='center', va='bottom', fontsize=10, clip_on=False)
    ax_neg.text(0.5, 1.25, f"Negative: {negative_influence:.4f}", transform=ax_neg.transAxes,
                ha='center', va='bottom', fontsize=10, clip_on=False)
    # Global influence is loaded from index_15533_global_influence.npy

    # Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_timestamp = plt.axes([0.15, 0.13, 0.65, 0.03], facecolor=axcolor)
    ax_percent = plt.axes([0.15, 0.08, 0.65, 0.03], facecolor=axcolor)
    slider_timestamp = widgets.Slider(ax_timestamp, 'Timestamp', 0, num_timestamps-1, valinit=0, valstep=1)
    slider_percent = widgets.Slider(ax_percent, 'Top % Influential Points', 10, 100, valinit=100, valstep=10)

    # Small, side-by-side buttons at the right of each slider
    btn_width = 0.025
    btn_height = 0.03
    # Timestamp buttons
    ax_btn_t_minus = plt.axes([0.88, 0.13, btn_width, btn_height])
    ax_btn_t_plus = plt.axes([0.905, 0.13, btn_width, btn_height])
    btn_t_minus = widgets.Button(ax_btn_t_minus, '-', hovercolor='0.85')
    btn_t_plus = widgets.Button(ax_btn_t_plus, '+', hovercolor='0.85')
    # Percent buttons
    ax_btn_p_minus = plt.axes([0.88, 0.08, btn_width, btn_height])
    ax_btn_p_plus = plt.axes([0.905, 0.08, btn_width, btn_height])
    btn_p_minus = widgets.Button(ax_btn_p_minus, '-', hovercolor='0.85')
    btn_p_plus = widgets.Button(ax_btn_p_plus, '+', hovercolor='0.85')

    # Button callbacks
    def on_all_click(event):
        current_filter_type[0] = 'all'
        update_button_colors()
        update_plots()

    def on_pos_click(event):
        current_filter_type[0] = 'positive'
        update_button_colors()
        update_plots()

    def on_neg_click(event):
        current_filter_type[0] = 'negative'
        update_button_colors()
        update_plots()

    def on_global_click(event):
        current_filter_type[0] = 'global'
        update_button_colors()
        update_plots()

    def update_plots(timestamp=None, percent=None):
        timestamp = int(slider_timestamp.val) if timestamp is None else int(timestamp)
        percent = slider_percent.val if percent is None else percent

        # Update left plot (reference field)
        im0.set_data(yrefs)
        # Find the test point location for index 15533
        loc = test_coords[15533]
        if marker[0] is not None:
            marker[0].remove()
        marker[0] = ax0.scatter(loc[1] * yrefs.shape[1], loc[0] * yrefs.shape[0],
                                s=100, edgecolor='yellow', facecolor='none', linewidth=2)

        # Update right plot (influence)
        im1.set_data(yrefs)
        if influence_scatter[0] is not None:
            influence_scatter[0].remove()
            influence_scatter[0] = None
        influence_np = influences_all[timestamp]
        if influence_np is None:
            print(f"No influence data for timestamp {timestamp}")
            fig.canvas.draw_idle()
            return
        
        # Metrics are now static and calculated once at the top
        
        # Apply influence-type filtering based on selected button
        if current_filter_type[0] == 'global':
            filtered_influences = global_influence_all
            filtered_coords = train_coords
        elif current_filter_type[0] == 'positive':
            mask = influence_np > 0
            filtered_influences = influence_np[mask]
            filtered_coords = train_coords[mask]
        elif current_filter_type[0] == 'negative':
            mask = influence_np < 0
            filtered_influences = influence_np[mask]
            filtered_coords = train_coords[mask]
        else:
            filtered_influences = influence_np
            filtered_coords = train_coords

        if len(filtered_influences) == 0:
            if influence_scatter[0] is not None:
                influence_scatter[0].remove()
                influence_scatter[0] = None
            fig.canvas.draw_idle()
            return

        influence_sorted_indices = np.argsort(filtered_influences)
        n_top = max(1, int((percent / 100.0) * len(filtered_influences)))
        top_idx = influence_sorted_indices[-n_top:]
        xs = filtered_coords[top_idx, 1] * yrefs.shape[1]
        ys = filtered_coords[top_idx, 0] * yrefs.shape[0]
        top_influences = filtered_influences[top_idx]
        norm = plt.Normalize(vmin=np.min(top_influences), vmax=np.max(top_influences))
        cmap_influence = plt.get_cmap('PRGn')
        # Use reversed color map for negative filter to show darker at bottom, lighter at top
        cmap_points = 'Greens_r' if current_filter_type[0] == 'negative' else 'Greens'
        influence_scatter[0] = ax1.scatter(xs, ys, s=20, c=top_influences, cmap=cmap_points, norm=norm)
        # Add or update colorbar for influence
        if not hasattr(update_plots, 'cbar') or update_plots.cbar is None:
            update_plots.cbar = fig.colorbar(influence_scatter[0], ax=ax1, orientation='vertical')
            update_plots.cbar.set_label(f'Influence (Top {percent:.0f}%)')
        else:
            update_plots.cbar.mappable = influence_scatter[0]
            update_plots.cbar.mappable.set_clim(np.min(top_influences), np.max(top_influences))
            update_plots.cbar.set_label(f'Influence (Top {percent:.0f}%)')
            update_plots.cbar.update_normal(influence_scatter[0])
        fig.canvas.draw_idle()

    def slider_update(val):
        update_plots()

    def t_minus(event):
        val = int(slider_timestamp.val)
        if val > 0:
            slider_timestamp.set_val(val - 1)

    def t_plus(event):
        val = int(slider_timestamp.val)
        if val < num_timestamps - 1:
            slider_timestamp.set_val(val + 1)

    def p_minus(event):
        val = int(slider_percent.val)
        if val > 10:
            slider_percent.set_val(val - 10)

    def p_plus(event):
        val = int(slider_percent.val)
        if val < 100:
            slider_percent.set_val(val + 10)

    slider_timestamp.on_changed(slider_update)
    slider_percent.on_changed(slider_update)
    btn_t_minus.on_clicked(t_minus)
    btn_t_plus.on_clicked(t_plus)
    btn_p_minus.on_clicked(p_minus)
    btn_p_plus.on_clicked(p_plus)

    # Connect influence-type buttons
    btn_all.on_clicked(on_all_click)
    btn_pos.on_clicked(on_pos_click)
    btn_neg.on_clicked(on_neg_click)
    btn_global.on_clicked(on_global_click)

    # Initialize
    update_plots()
    
    # Save the plot before showing
    VIS_PNG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'visualization-tool-pngs')
    os.makedirs(VIS_PNG_DIR, exist_ok=True)
    output_filename = os.path.join(VIS_PNG_DIR, 'influence_over_time.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to '{output_filename}'")
    
    plt.show()

# Usage 
if __name__ == '__main__':
    plot_influence_over_time(yrefs, test_coords, train_coords, influences_all, global_influence_all) 