import numpy as np
import matplotlib.pyplot as plt




def plot_ternary(grid, values=None, title='Ternary Plot', output_path=None, figsize=(8, 7), 
                 labels=None, cmap='viridis_r', s=30):
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
    """
    if values is None:
        values = np.arange(len(grid))
    
    if labels is None:
        labels = ['l0', 'l1', 'l2']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to ternary coordinates
    x = 0.5 * (2 * grid[:, 1] + grid[:, 2])
    y = (np.sqrt(3) / 2) * grid[:, 2]
    
    scatter = ax.scatter(x, y, c=values, cmap=cmap, s=s)
    
    # Triangle boundary
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', linewidth=2)
    
    # Labels closer to triangle
    ax.text(0, -0.02, labels[0], fontsize=14, fontweight='bold', ha='center', va='top')
    ax.text(1, -0.02, labels[1], fontsize=14, fontweight='bold', ha='center', va='top')
    ax.text(0.5, np.sqrt(3)/2 + 0.02, labels[2], fontsize=14, fontweight='bold', ha='center', va='bottom')
    
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    plt.colorbar(scatter, ax=ax, shrink=0.8)
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path)
    
    return fig, ax