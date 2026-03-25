import argparse
import numpy as np
import matplotlib.pyplot as plt


def remove_outliers_iqr(arr, factor=1.5):
    mask = np.ones(arr.shape[0], dtype=bool)
    for col in range(arr.shape[1]):
        q1, q3 = np.percentile(arr[:, col], [25, 75])
        iqr = q3 - q1
        mask &= (arr[:, col] >= q1 - factor * iqr) & (arr[:, col] <= q3 + factor * iqr)
    return mask


def plot_pareto_3views(npz_path, dpi=300, output_path=None, obj_names=("TDR", "L", "E"),
                       cmap="viridis_r", remove_outliers=True, iqr_factor=1.5):
    data = np.load(npz_path, allow_pickle=True)
    fitness = data["fitness"][:, :3]

    if remove_outliers:
        mask = remove_outliers_iqr(fitness, factor=iqr_factor)
        clean = fitness[mask]
        n_removed = (~mask).sum()
    else:
        clean = fitness
        n_removed = 0

    # Each view looks along one axis → that axis is colored
    views = [
        (25, 0,   0),  # along TDR
        (25, 90,  1),  # along L
        (75, 45,  2),  # top-down along E
    ]

    fig = plt.figure(figsize=(22, 7), dpi=dpi)

    for idx, (elev, azim, color_col) in enumerate(views):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        order = np.argsort(clean[:, color_col])
        c = clean[order]

        sc = ax.scatter(c[:, 0], c[:, 1], c[:, 2],
                        c=c[:, color_col], cmap=cmap, s=55,
                        edgecolors="k", linewidths=0.3, alpha=0.9, depthshade=True)

        ax.set_xlabel(obj_names[0], fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel(obj_names[1], fontsize=11, fontweight="bold", labelpad=8)
        ax.set_zlabel(obj_names[2], fontsize=11, fontweight="bold", labelpad=8)
        ax.set_title(f"Color = {obj_names[color_col]}  (elev={elev}°, azim={azim}°)",
                     fontsize=12, fontweight="bold")
        ax.view_init(elev=elev, azim=azim)

        for setter, col in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            margin = 0.05 * (c[:, col].max() - c[:, col].min())
            setter(c[:, col].min() - margin, c[:, col].max() + margin)

        ax.tick_params(labelsize=7)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
        cbar.set_label(obj_names[color_col], fontsize=11, fontweight="bold")

    title = f"3D Pareto Front (n={len(clean)}"
    if n_removed > 0:
        title += f", {n_removed} outliers removed"
    title += ")"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path is None:
        output_path = npz_path.replace(".npz", f"_pareto_3views.png")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path} ({dpi} DPI)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 3-view Pareto front from MOEGA checkpoint")
    parser.add_argument("npz_path", help="Path to .npz checkpoint file")
    parser.add_argument("--dpi", type=int, default=400, help="Output DPI (default: 300)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image path")
    parser.add_argument("--obj-names", nargs=3, default=["TDR", "L", "E"], help="Objective names (default: TDR L E)")
    parser.add_argument("--cmap", type=str, default="viridis_r", help="Colormap (default: viridis_r)")
    parser.add_argument("--no-outlier-removal", action="store_true", help="Keep all points")
    parser.add_argument("--iqr-factor", type=float, default=1.5, help="IQR factor for outlier removal (default: 1.5)")
    args = parser.parse_args()

    plot_pareto_3views(
        args.npz_path,
        dpi=args.dpi,
        output_path=args.output,
        obj_names=tuple(args.obj_names),
        cmap=args.cmap,
        remove_outliers=not args.no_outlier_removal,
        iqr_factor=args.iqr_factor,
    )