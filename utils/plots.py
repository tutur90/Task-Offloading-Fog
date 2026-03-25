import numpy as np
import matplotlib.pyplot as plt




def plot_ternary(grid, values=None, title='Ternary Plot', output_path=None, figsize=(8, 7), 
                 labels=None, cmap='viridis_r', s=30, max_value=None, dpi=400):
    """
    Plot probability grid in ternary diagram
    
    Args:
        grid: Nx3 array where each row sums to 1
        values: Optional Nx1 array for color mapping
        title: Plot title
        figsize: Figure size
        labels: List of 3 labels for [l0, l1, l2], default ['l0', 'l1', 'l2']
        cmap: Colormap name
        s: Point size
        dpi: Resolution of the saved figure
    """
    if values is None:
        values = np.arange(len(grid))
    
    if labels is None:
        labels = ['l0', 'l1', 'l2']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to ternary coordinates
    x = 0.5 * (2 * grid[:, 1] + grid[:, 2])
    y = (np.sqrt(3) / 2) * grid[:, 2]
    
    # Set up colormap with black for values above max_value
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_over('black')

    scatter = ax.scatter(x, y, c=values, cmap=cmap_obj, s=s, vmax=max_value)
    
    # Triangle boundary
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', linewidth=2)
    
    # Labels closer to triangle
    ax.text(0, -0.02, labels[0], fontsize=14, fontweight='bold', ha='center', va='top')
    ax.text(1, -0.02, labels[1], fontsize=14, fontweight='bold', ha='center', va='top')
    ax.text(0.5, np.sqrt(3)/2 + 0.02, labels[2], fontsize=14, fontweight='bold', ha='center', va='bottom')
    
    ax.set_aspect('equal')
    # ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    plt.colorbar(scatter, ax=ax, shrink=0.8, extend='max')
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=dpi)
    
    return fig, ax


def plot_grid_search_heatmap(param_specs, progress, metric_idx=3, metric_name="Score", output_path=None, max_value=None, dpi=400):
    """
    Plot a heatmap for grid search results with exactly two variables.

    Args:
        param_specs: dict mapping param names to lists of values
        progress: dict with "val_metrics" and "test_metrics"
        metric_idx: which metric to plot (default 3 = score)
        metric_name: label for the metric
        output_path: path to save the figure
        max_value: maximum value for colorbar (values above shown in black)
    """
    from utils.grid_search import params_to_key

    if len(param_specs) != 2:
        print(f"Heatmap requires exactly 2 parameters, got {len(param_specs)}")
        return

    param_names = list(param_specs.keys())
    param1_values = param_specs[param_names[0]]
    param2_values = param_specs[param_names[1]]

    # Create matrices for val and test metrics
    val_matrix = np.zeros((len(param1_values), len(param2_values)))
    test_matrix = np.zeros((len(param1_values), len(param2_values)))

    for i, v1 in enumerate(param1_values):
        for j, v2 in enumerate(param2_values):
            key = params_to_key({param_names[0]: v1, param_names[1]: v2})
            if key in progress["val_metrics"]:
                val_matrix[i, j] = progress["val_metrics"][key][metric_idx]
            if key in progress["test_metrics"]:
                test_matrix[i, j] = progress["test_metrics"][key][metric_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Set up colormap with black for values above max_value
    cmap_obj = plt.get_cmap('viridis_r').copy()
    cmap_obj.set_over('black')

    for ax, matrix, title in zip(axes, [val_matrix, test_matrix], ["Validation", "Test"]):
        vmin = matrix.min()
        vmax = max_value if max_value is not None else matrix.max()
        im = ax.imshow(matrix, cmap=cmap_obj, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(param2_values)))
        ax.set_xticklabels(param2_values)
        ax.set_yticks(range(len(param1_values)))
        ax.set_yticklabels(param1_values)
        ax.set_xlabel(param_names[1].split('.')[-1])
        ax.set_ylabel(param_names[0].split('.')[-1])
        ax.set_title(f"{title} {metric_name}")

        # Add value annotations with contrasting colors
        mid_value = (vmin + vmax) / 2
        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                text_color = 'white' 
                ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center',
                       color=text_color, fontsize=8, fontweight='bold')

        plt.colorbar(im, ax=ax, extend='max')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=dpi)
        print(f"Heatmap saved to {output_path}")
    plt.close(fig)