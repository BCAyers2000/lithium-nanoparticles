import os
import re
from typing import Dict, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb, Normalize
from matplotlib.colorbar import ColorbarBase
from pymatgen.core import Structure
from wulffpack import SingleCrystal


def extract_surface_energies_pbs(parent_dir: str) -> Dict[Tuple[int, int, int], float]:
    """Extract surface energies from a directory structure containing PBS output files.
    
    Args:
        parent_dir: Directory path containing subdirectories with PBS output files.
        
    Returns:
        Dictionary mapping Miller indices tuples to surface energy values in J/m^2.
    """
def extract_surface_energies_slurm(parent_dir: str) -> Dict[Tuple[int, int, int], float]:
    """Extract surface energies from a directory structure containing SLURM output files.
    
    Args:
        parent_dir: Path to the parent directory containing Miller index subdirectories
        
    Returns:
        Dictionary mapping Miller indices tuples to their corresponding surface energies in J/m^2
        
    Example:
        >>> energies = extract_surface_energies_slurm("/path/to/calculations")
        >>> print(energies[(1, 1, 1)])
        2.34
    """
    results = {}
    
    for dirname in (d for d in os.listdir(parent_dir) if d.startswith('[')):
        full_path = os.path.join(parent_dir, dirname)
        if not os.path.isdir(full_path):
            continue
            
        miller_match = re.search(r'\[(\d{3})\]', dirname)
        if not miller_match:
            continue
            
        miller_indices = miller_match.group(1)
        miller_tuple = tuple(int(miller_indices[i]) for i in range(3))
        
        for file in os.listdir(full_path):
            if file.startswith('slurm-') and file.endswith('.out'):
                slurm_path = os.path.join(full_path, file)
                try:
                    with open(slurm_path, 'r') as f:
                        content = f.read()
                        energy_match = re.search(r'Surface energy:\s*(\d+\.\d+)\s*J/m\^2', content)
                        if energy_match:
                            results[miller_tuple] = float(energy_match.group(1))
                            break
                except Exception as e:
                    print(f"Error reading {slurm_path}: {e}")
    
    return results


def extract_surface_energies(parent_dir: str, system: str = 'pbs') -> Dict[Tuple[int, int, int], float]:
    """Extract surface energies from a directory structure containing job output files.
    
    Args:
        parent_dir: Directory path containing subdirectories with output files.
        system: Job system type ('pbs' or 'slurm'). Defaults to 'pbs'.
        
    Returns:
        Dictionary mapping Miller indices tuples to surface energy values in J/m^2.
        
    Raises:
        ValueError: If system type is not recognized.
    """
    if system.lower() == 'pbs':
        return extract_surface_energies_pbs(parent_dir)
    elif system.lower() == 'slurm':
        return extract_surface_energies_slurm(parent_dir)
    else:
        raise ValueError(f"Unrecognized system type: {system}. Use 'pbs' or 'slurm'.")


def extract_surface_energies_pbs(parent_dir: str) -> Dict[Tuple[int, int, int], float]:
    results = {}
    
    for dirname in os.listdir(parent_dir):
        if not dirname.startswith('[') or not dirname.endswith(']'):
            continue
            
        full_path = os.path.join(parent_dir, dirname)
        if not os.path.isdir(full_path):
            continue
        
        try:
            miller_str = dirname.strip('[]')
            if len(miller_str) != 3:
                continue
            miller_tuple = tuple(int(miller_str[i]) for i in range(3))
            
            output_file = f"miller-{miller_str}.o"
            matching_files = [f for f in os.listdir(full_path) if f.startswith(output_file)]
            
            if matching_files:
                pbs_path = os.path.join(full_path, matching_files[0])
                try:
                    with open(pbs_path, 'r') as f:
                        content = f.read()
                        energy_match = re.search(r'Surface energy:\s*(\d+\.\d+)\s*J/m\^2', content)
                        if energy_match:
                            results[miller_tuple] = float(energy_match.group(1))
                except Exception as e:
                    print(f"Error reading {pbs_path}: {e}")
                        
        except Exception as e:
            print(f"Error processing directory {dirname}: {e}")
            continue
    
    return results


def simple_wulff(
    surface_energies: Dict[Tuple[int, int, int], float],
    bulk_path: str,
    output_path: Optional[str] = None,
    view_angles: Tuple[float, float, float] = (45, 45, 0),
    colorbar_x: float = 0.38,
    colorbar_y: float = 0.15,
    colorbar_width: float = 0.45
) -> SingleCrystal:
    """Create a Wulff shape visualization with surface energy-based coloring and colorbar.
    
    Args:
        surface_energies: Dictionary mapping Miller indices to surface energies.
        bulk_path: Path to bulk structure file.
        output_path: Optional path to save the visualization.
        view_angles: Tuple of angles (elevation, azimuth, roll) for viewing the shape.
        colorbar_x: X-position of the colorbar.
        colorbar_y: Y-position of the colorbar.
        colorbar_width: Width of the colorbar.
        
    Returns:
        SingleCrystal object representing the Wulff construction.
    """
    bulk_structure = Structure.from_file(bulk_path)
    bulk_structure_ase = bulk_structure.to_ase_atoms()
    particle = SingleCrystal(surface_energies, primitive_structure=bulk_structure_ase)
    
    facet_fractions = particle.facet_fractions
    active_surfaces = {miller: surface_energies[miller] 
                      for miller in facet_fractions.keys() if facet_fractions[miller] > 0}
    
    min_energy = min(active_surfaces.values())
    max_energy = max(active_surfaces.values())
    
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax = fig.add_subplot(gs[0], projection='3d')
    
    color_order = sorted(active_surfaces.keys(), 
                        key=lambda x: (sum(x), x[0], x[1], x[2]))
    
    n_surfaces = len(color_order)
    colors = {}
    saturation_factor = 0.85
    
    for i, miller in enumerate(color_order):
        color_val = i / (n_surfaces - 1) if n_surfaces > 1 else 0.5
        rgb_color = plt.cm.viridis(color_val)[:3]
        hsv_color = rgb_to_hsv(rgb_color)
        hsv_color[1] *= saturation_factor
        rgb_color = hsv_to_rgb(hsv_color)
        colors[miller] = (*rgb_color, 1.0)
    
    particle.make_plot(ax, colors=colors, alpha=0.95)
    ax.view_init(*view_angles)
    ax.set_axis_off()
    
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')
    
    sorted_by_percentage = sorted(facet_fractions.items(),
                                key=lambda x: x[1],
                                reverse=True)
    
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[m], alpha=0.9)
                      for m, _ in sorted_by_percentage if m in colors]
    
    legend_labels = [f"({m[0]}{m[1]}{m[2]}) - {p*100:.1f}%" 
                    for m, p in sorted_by_percentage if m in colors]
    
    legend = legend_ax.legend(
        legend_elements,
        legend_labels,
        title="Miller Indices",
        loc='center left',
        bbox_to_anchor=(-0.2, 0.45),
        bbox_transform=legend_ax.transAxes,
        fontsize=15,
        title_fontsize=17,
        labelspacing=1.32,
        handletextpad=1.1,
        handlelength=1.65,
        frameon=True,
        edgecolor='lightgray'
    )
    
    cbar_ax = fig.add_axes([
        colorbar_x - (colorbar_width/2),
        colorbar_y,
        colorbar_width,
        0.04
    ])
    
    norm = Normalize(vmin=min_energy, vmax=max_energy)
    cbar = ColorbarBase(cbar_ax, cmap=plt.cm.viridis, norm=norm, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)
    
    fig.text(
        colorbar_x,
        colorbar_y - 0.06,
        "Surface Energy (J/m²)",
        fontsize=14,
        ha='center'
    )
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return particle


def main() -> None:
    """Main execution function for processing surface energies and generating Wulff construction visualisation."""
    SURFACES_DIR = "/mnt/lustre/a2fs-nvme/work/e89/e89/ba3g18/bondi-production/[-0.5V]"
    BULK_PATH = "/mnt/lustre/a2fs-nvme/work/e89/e89/ba3g18/Li.cif"
    OUTPUT_PATH = "/mnt/lustre/a2fs-nvme/work/e89/e89/ba3g18/Colour_bar/-0.5V_wulff.png"
    SYSTEM_TYPE = "pbs"  # or "slurm"
    
    surface_energies = extract_surface_energies(SURFACES_DIR, system=SYSTEM_TYPE)
    
    if surface_energies:
        particle = simple_wulff(
            surface_energies=surface_energies,
            bulk_path=BULK_PATH,
            output_path=OUTPUT_PATH
        )
    else:
        print("No surface energies found in the directories")


if __name__ == "__main__":
    main()
