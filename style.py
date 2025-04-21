from cycler import cycler

# see http://matplotlib.org/users/customizing.html for all options

style1 = {
    # Line styles
    "lines.linewidth": 1.5,
    "lines.antialiased": True,
    # Font
    "font.size": 10.0,
    "font.family": "sans-serif",
    # Axes
    "axes.linewidth": 1.2,
    "axes.titlesize": "x-large",
    "axes.labelsize": "medium",
    "axes.prop_cycle": cycler(
        "color",
        [
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # magenta
            "#7f7f7f",  # gray
            "#bcbd22",  # yellow
            "#17becf",  # cyan
        ],
    ),
    # Ticks
    "xtick.major.size": 4,
    "xtick.minor.size": 2,
    "xtick.major.width": 1.2,
    "xtick.minor.width": 1.2,
    "xtick.major.pad": 4,
    "xtick.minor.pad": 2,
    "xtick.labelsize": "small",
    "xtick.direction": "in",
    "xtick.top": True,
    "xtick.bottom": True,
    "xtick.minor.visible": True,
    "ytick.major.size": 4,
    "ytick.minor.size": 2,
    "ytick.major.width": 1.2,
    "ytick.minor.width": 1.2,
    "ytick.major.pad": 4,
    "ytick.minor.pad": 2,
    "ytick.labelsize": "small",
    "ytick.direction": "in",
    "ytick.right": True,
    "ytick.left": True,
    "ytick.minor.visible": True,
    # Legend
    "legend.fancybox": False,
    "legend.fontsize": "small",
    "legend.scatterpoints": 1,
    "legend.numpoints": 1,
    "legend.loc": "best",
    #'legend.loc': 'center left',
    #'legend.loc': 'upper right',
    #'legend.loc': 'lower left',
    #'legend.loc': 'lower right',
    # Figure
    "figure.figsize": [3.321, 7/9*3.321],
    "figure.titlesize": "large",
    # Images
    "image.cmap": "magma",
    "image.origin": "lower",
    # Saving
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
}
