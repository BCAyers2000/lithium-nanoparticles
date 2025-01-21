import os
import re
from typing import Dict, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb, Normalize
from pymatgen.core import Structure
from wulffpack import SingleCrystal

def extract_surface_energies(parent_dir: str) -> Dict[Tuple[int, int, int], float]:
    """
    Extract surface energies from a directory structure containing SLURM output files.
    
    Args:
        parent_dir: Path to the parent directory containing Miller index subdirectories
        
    Returns:
        Dictionary mapping Miller indices tuples to their corresponding surface energies in J/m^2
        
    Example:
        >>> energies = extract_surface_energies("/path/to/calculations")
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

def simple_wulff(
    surface_energies: Dict[Tuple[int, int, int], float],
    bulk_path: str,
    output_path: Optional[str] = None,
    view_angles: Tuple[float, float, float] = (45, 45, 0)
) -> SingleCrystal:
    """
    Create a Wulff shape visualization with surface energy-based coloring.
    
    Args:
        surface_energies: Dictionary mapping Miller indices to surface energies
        bulk_path: Path to the bulk structure CIF file
        output_path: Optional path to save the generated figure
        view_angles: Tuple of (elevation, azimuth, roll) angles for visualization
        
    Returns:
        SingleCrystal object representing the constructed Wulff shape
        
    Example:
        >>> particle = simple_wulff(energies, "bulk.cif", "wulff.png")
    """
    bulk_structure = Structure.from_file(bulk_path)
    bulk_structure_ase = bulk_structure.to_ase_atoms()
    particle = SingleCrystal(surface_energies, primitive_structure=bulk_structure_ase)
    
    facet_fractions = particle.facet_fractions
    active_surfaces = {miller: surface_energies[miller] 
                      for miller in facet_fractions.keys()}
    
    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
    ax = fig.add_subplot(gs[1], projection='3d')
    
    sorted_surfaces = sorted(active_surfaces.keys(), 
                           key=lambda x: (sum(x), x[0], x[1], x[2]))
    
    n_surfaces = len(sorted_surfaces)
    colors = {}
    saturation_factor = 0.85
    
    for i, miller in enumerate(sorted_surfaces):
        color_val = i / (n_surfaces - 1) if n_surfaces > 1 else 0.5
        rgb_color = plt.cm.viridis(color_val)[:3]
        hsv_color = rgb_to_hsv(rgb_color)
        hsv_color[1] *= saturation_factor
        rgb_color = hsv_to_rgb(hsv_color)
        colors[miller] = (*rgb_color, 1.0)
    
    particle.make_plot(ax, colors=colors, alpha=0.95)
    ax.view_init(*view_angles)
    ax.set_axis_off()
    
    legend_ax = fig.add_subplot(gs[0])
    legend_ax.axis('off')
    
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=colors[m], alpha=0.9)
                      for m in sorted_surfaces]
    legend_labels = [f"({m[0]}{m[1]}{m[2]})" 
                    for m in sorted_surfaces]
    
    legend = legend_ax.legend(
        legend_elements,
        legend_labels,
        title="Miller Indices",
        loc='center',
        ncol=len(active_surfaces) // 2 + len(active_surfaces) % 2,
        bbox_to_anchor=(0.5, 0.5),
        bbox_transform=legend_ax.transAxes
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return particle

def main() -> None:
    """
    Main execution function that processes surface energies and generates
    a Wulff construction visualization.
    """
    SURFACES_DIR = "/mnt/lustre/a2fs-nvme/work/e89/e89/ba3g18/bondi-production/[0.0V]"
    BULK_PATH = "/mnt/lustre/a2fs-nvme/work/e89/e89/ba3g18/Li.cif"
    OUTPUT_PATH = "simple-wulff_shape.png"
    
    surface_energies = extract_surface_energies(SURFACES_DIR)
    
    if surface_energies:
        print("Found surface energies:", surface_energies)
        particle = simple_wulff(
            surface_energies=surface_energies,
            bulk_path=BULK_PATH,
            output_path=OUTPUT_PATH
        )
    else:
        print("No surface energies found in the directories")

if __name__ == "__main__":
    main()