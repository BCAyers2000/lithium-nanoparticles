import numpy as np
from ase.io import read
from pathlib import Path
from ase.build import bulk
from ase.atoms import Atoms
from typing import Dict, Any, Tuple
from ase.optimize import BFGSLineSearch
from ase.io.espresso import write_fortran_namelist
from ase.calculators.espresso import Espresso, EspressoProfile

# Variables
IBRAV = 12
MU_GC =   -1.904
KSPACING = 0.10
EXTERNAL = True
USE_ENVIRON = True
PSEUDOPOTENTIALS = {"Li": "Li.pbe-sl-kjpaw_psl.1.0.0.UPF"}
PSEUDO_DIR = "/work/e89/e89/ba3g18/Repos/Pseudopotentials/Pslibrary"
COMMAND = ("srun -v --distribution=block:block --hint=nomultithread --cpu-freq=2250000 "
           "/work/e89/e89/ba3g18/Repos/q-e/bin/pw.x")

def generate_kpts(atoms: Atoms, periodic_3d: bool = False) -> list:
    """
    Generate k-points based on cell size and periodic boundary conditions
    """
    cell_lengths = atoms.cell.lengths()
    kpts = np.ceil(2 * np.pi / (cell_lengths * KSPACING)).astype(int)
    if not periodic_3d:
        kpts[2] = 1
    return kpts.tolist()


def setup_atoms() -> Atoms:
    """
    Set up the initial atomic structure
    """
    return bulk("Li", "bcc", cubic=True, a=3.44)


def create_espresso_profile(use_environ: bool = False) -> EspressoProfile:
    """
    Create an EspressoProfile with the correct paths
    """
    command = COMMAND
    if use_environ:
        command += " --environ"
    return EspressoProfile(
        command=command,
        pseudo_dir=PSEUDO_DIR,
    )


def get_input_data(calculation_type: str = "vc-relax") -> Dict[str, Any]:
    """
    Get the input data for Espresso calculations
    """
    return {
        "control": {
            "calculation": calculation_type,
            "verbosity": "high",
            "restart_mode": "from_scratch",
            "nstep": 999,
            "tstress": False,
            "tprnfor": True,
            "outdir": "./Li/",
            "prefix": "Lithium",
            "etot_conv_thr": 1.0e-10,
            "forc_conv_thr": 1.0e-10,
            "disk_io": "minimal",
            "pseudo_dir": PSEUDO_DIR,
        },
        "system": {
            "ibrav": 0,
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
            "diago_thr_init": 1.0e-12,
            "diago_full_acc": True,
            "diago_david_ndim": 4,
            "startingwfc": "random",
            "diagonalization": "david",
        },
        "ions": {"ion_dynamics": "bfgs", "upscale": 1e6, "bfgs_ndim": 1},
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
    periodic_3d: bool = True,
    use_external: bool = False,
) -> Tuple[float, Atoms]:
    """
    Run an Espresso calculation
    """
    calc = Espresso(
        input_data=input_data,
        pseudopotentials=PSEUDOPOTENTIALS,
        profile=profile,
        directory=directory,
        kpts=generate_kpts(atoms, periodic_3d=periodic_3d),
    )
    atoms.calc = calc

    if use_external:
        opt = BFGSLineSearch(
            atoms, trajectory=f"{directory}/QE.traj", logfile=f"{directory}/QE.log"
        )
        opt.run(fmax=0.025, steps=10000)
        energy = atoms.get_potential_energy()
        relaxed_atoms = atoms.copy()
    else:
        energy = atoms.get_potential_energy()
        relaxed_atoms = read(f"{directory}/espresso.pwo")

    return energy, relaxed_atoms


def prepare_slab(relaxed_bulk: Atoms) -> Atoms:
    """
    Prepare a slab from the relaxed bulk structure
    """
    slab = read("/mnt/lustre/a2fs-nvme/work/e89/e89/ba3g18/bondi-production/[-1.0V]/[331]/Surface/espresso.pwo", index=-1)
    return slab


def main() -> None:
    atoms = setup_atoms()
    bulk_profile = create_espresso_profile()

    bulk_dir = Path("Bulk")
    bulk_dir.mkdir(exist_ok=True)
    bulk_input_data = get_input_data("scf")
    bulk_energy, relaxed_bulk = run_calculation(
        atoms,
        bulk_input_data,
        bulk_profile,
        bulk_dir,
        periodic_3d=True,
        use_external=False,
    )

    slab_dir = Path("Surface")
    slab_dir.mkdir(exist_ok=True)
    slab = prepare_slab(relaxed_bulk)
    slab_profile = create_espresso_profile(use_environ=USE_ENVIRON)
   
    if USE_ENVIRON:
        with (slab_dir / "environ.in").open("w") as f:
            write_fortran_namelist(f, get_environ_input_data())

    slab_input_data = get_input_data("scf" if EXTERNAL else "relax")
    slab_input_data["control"].update(
        {"etot_conv_thr": 1.0e-5, "forc_conv_thr": 3.88e-4}
    )
    slab_input_data["electrons"].update(
        {
            "mixing_beta": 0.2,
            "mixing_mode": "local-TF",
            "conv_thr": 1.0e-14 if EXTERNAL else 1.0e-6,  
            "diagonalization": "david",
        }
    )
    slab_input_data["system"].update(
        {
            "ibrav": IBRAV,
            "nbnd": int((3 * len(slab) / 2) * 1.5),
            "lgcscf": True,
            "gcscf_conv_thr": 1e-03,
            "gcscf_mu": MU_GC,
        }
    )
    

    slab_energy, _ = run_calculation(
        slab,
        slab_input_data,
        slab_profile,
        slab_dir,
        periodic_3d=False,
        use_external=EXTERNAL,
    )

    area = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
    surface_energy = (
        1 / 2 / area * (slab_energy - ((len(slab) / len(atoms)) * bulk_energy))
    )
    print(f"Surface energy: {surface_energy * 16.02:.4f} J/m^2")


if __name__ == "__main__":
    main()
