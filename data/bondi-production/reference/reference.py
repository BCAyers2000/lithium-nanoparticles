import numpy as np
from ase.io import read
from ase.units import Ry
from pathlib import Path
from ase.build import bulk
from ase.atoms import Atoms
from tabulate import tabulate
from typing import Dict, Any, Tuple
from ase.io.espresso import write_fortran_namelist
from ase.calculators.espresso import Espresso, EspressoProfile

USE_ENVIRON = True
BULK_KPTS = [10, 10, 10]
PSEUDOPOTENTIALS = {"Li": "Li.pbe-sl-kjpaw_psl.1.0.0.UPF"}
APPLIED_POTENTIALS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]


def extract_electrolyte_energy(output_file: Path) -> float:
    """Extract electrolyte free energy from QE output and convert to eV"""
    with output_file.open() as f:
        lines = [line for line in f if "electrolyte free energy" in line]
        return float(lines[-1].split()[4]) * Ry if lines else 0.0


def create_espresso_profile(use_environ: bool = False) -> EspressoProfile:
    """Create an EspressoProfile with the correct paths"""
    base_command = "srun --mpi=pmix /iridisfs/home/ba3g18/Repos/q-e/bin/pw.x"
    environ_flag = " --environ" if use_environ else ""
    return EspressoProfile(
        command=f"{base_command}{environ_flag}",
        pseudo_dir="/home/ba3g18/Repos/Pseudopotentials/pslibrary.1.0.0/pbe/PSEUDOPOTENTIALS",
    )


def get_input_data(calculation_type: str = "vc-relax") -> Dict[str, Any]:
    """Get the input data for Espresso calculations"""
    return {
        "control": {
            "calculation": calculation_type,
            "verbosity": "high",
            "restart_mode": "from_scratch",
            "nstep": 999,
            "tstress": False,
            "tprnfor": True,
            "outdir": "./",
            "prefix": "pw.dir",
            "etot_conv_thr": 1.0e-6,
            "forc_conv_thr": 1.0e-6,
            "disk_io": "minimal",
        },
        "system": {
            "tot_charge": 0.0,
            "ecutwfc": 80.0,
            "ecutrho": 800,
            "occupations": "smearing",
            "degauss": 0.01,
            "smearing": "cold",
            "input_dft": "pbe",
            "nspin": 1,
        },
        "electrons": {
            "electron_maxstep": 999,
            "scf_must_converge": True,
            "conv_thr": 1.0e-14,
            "mixing_mode": "plain",
            "mixing_beta": 0.80,
            "startingwfc": "random",
            "diagonalization": "david",
        },
        "ions": {"ion_dynamics": "bfgs", "upscale": 1e4, "bfgs_ndim": 1},
        "cell": {"press_conv_thr": 0.1, "cell_dofree": "all"},
    }


def get_environ_input_data() -> Dict[str, Any]:
    """
    Get the input data for environ calculations
    """
    return {
            "environ": {
                "verbose": 0,
                "cion(1)": 1.0,
                "cion(2)": 1.0,
                "zion(1)": 1.0,
                "zion(2)": -1.0,
                "cionmax": 5.0,
                "system_dim": 2,
                "system_axis": 3,
                "environ_thr": 0.1,
                "env_pressure": 0.0,
                "temperature": 298.15,
                "environ_type": "input",
                "env_electrolyte_ntyp": 2,
                "env_surface_tension": 37.3,
                "electrolyte_entropy": "full",
                "env_static_permittivity": 89.9,
                "electrolyte_linearized": False,
            },
            "boundary": {
                "alpha": 1.32,
                "radius_mode": "bondi",
                "solvent_mode": "ionic",
                "electrolyte_mode": "ionic",
            },
            "electrostatic": {
                "pbc_axis": 3,
                "pbc_dim": 2,
                "tol": 1.0e-15,
                "inner_tol": 1.0e-18,
                "pbc_correction": "parabolic",
            },
        }

def run_calculation(
    atoms: Atoms,
    input_data: Dict[str, Any],
    profile: EspressoProfile,
    directory: Path,
    kpts: list,
) -> Tuple[float, Atoms]:
    """Run an Espresso calculation"""
    calc = Espresso(
        input_data=input_data,
        pseudopotentials=PSEUDOPOTENTIALS,
        profile=profile,
        directory=directory,
        kpts=kpts,
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    relaxed_atoms = read(f"{directory}/espresso.pwo")
    return energy, relaxed_atoms


def prepare_li_plus(bulk_atoms: Atoms) -> Atoms:
    """Prepare a Li+ ion in the relaxed bulk cell parameters, centred"""
    cell = bulk_atoms.get_cell()
    li = Atoms("Li", positions=[[0.0, 0.0, 0.0]])
    li.set_cell(cell)
    li.set_pbc(True)
    li.center()
    return li


def generate_potential_table(reference_electrode: float) -> str:
    """Generate a table of chemical potentials for different applied potentials"""
    table_data = []
    for potential in APPLIED_POTENTIALS:
        table_data.append(
            [
                f"{potential:.2f}",
                f"{reference_electrode - potential:.3f}",
                f"{-reference_electrode - potential:.3f}",
            ]
        )
    headers = [
        "Applied Potential V (abs)",
        "Applied Potential V (ref)",
        "Applied Potential / μGC eV (ref)",
    ]
    return tabulate(table_data, headers=headers, tablefmt="grid")


def main() -> None:
    atoms = read("/scratch/ba3g18/QE/Bondi-ref/bulk.pwo")
    bulk_profile = create_espresso_profile()
    bulk_dir = Path("Bulk")
    bulk_dir.mkdir(exist_ok=True)

    bulk_input_data = get_input_data("vc-relax")
    bulk_energy, relaxed_bulk = run_calculation(
        atoms, bulk_input_data, bulk_profile, bulk_dir, kpts=BULK_KPTS
    )

    li_dir = Path("Li_plus")
    li_dir.mkdir(exist_ok=True)
    li = prepare_li_plus(relaxed_bulk)
    li_profile = create_espresso_profile(use_environ=USE_ENVIRON)

    if USE_ENVIRON:
        with (li_dir / "environ.in").open("w") as f:
            write_fortran_namelist(f, get_environ_input_data())

    li_input_data = get_input_data("relax")
    li_input_data["control"].update({"etot_conv_thr": 1.0e-5, "forc_conv_thr": 1.0e-5})
    li_input_data["electrons"].update(
        {
            "mixing_beta": 0.1,
            "mixing_mode": "local-TF",
            "conv_thr": 1.0e-14,
            "diagonalization": "david",
        }
    )
    li_input_data["system"].update(
        {
            "tot_charge": 1.0,
            "nbnd": int((3 * len(li) / 2) * 1.5),
        }
    )

    li_energy, _ = run_calculation(
        li, li_input_data, li_profile, li_dir, kpts=None
    )

    li_per_atom = bulk_energy / len(atoms)
    electrolyte_energy = extract_electrolyte_energy(li_dir / "espresso.pwo")
    reference_electrode = li_energy - electrolyte_energy - li_per_atom
    table = generate_potential_table(reference_electrode)

    with open("applied-bias.txt", "w") as f:
        f.write(f"Reference electrode energy: {reference_electrode:.3f} eV\n\n")
        f.write(table)

if __name__ == "__main__":
    main()
