import numpy as np
from ase import Atoms
from ase.gui.gui import GUI
from ase.gui.images import Images


def view_with_colors(atoms: Atoms, colors: np.ndarray) -> None:
    """Interactive ASE GUI viewer with custom per-atom RGB colors.

    Launches the standard ASE GUI but bypasses element-based coloring,
    allowing arbitrary per-atom colors defined by an RGB array. Useful
    for visualizing SOAP descriptors, coordination environments, or any
    other per-atom scalar/vector mapped to color.

    :param atoms: ASE Atoms object
    :param colors: Array of shape (n_atoms, 3) with RGB values in [0, 1].
                   Must have the same length as atoms.
    """
    colors = np.asarray(colors, dtype=float)
    hex_colors = [
        '#{:02X}{:02X}{:02X}'.format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b in colors
    ]

    images = Images([atoms])
    gui = GUI(images)

    def patched_get_colors(rgb: bool = False):
        if rgb:
            return [tuple(c) for c in colors]
        return list(hex_colors)

    gui.get_colors = patched_get_colors
    gui.draw()
    gui.run()