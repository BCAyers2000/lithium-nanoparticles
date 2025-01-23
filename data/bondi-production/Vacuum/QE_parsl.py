import parsl
from tqdm import tqdm
from pathlib import Path
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.launchers import SimpleLauncher
from parsl.executors import HighThroughputExecutor

@parsl.python_app
def run_quantum_espresso(
    structure_path=None, is_bulk=False, bulk_energy=None, bulk_atoms_per_cell=None
):
    import os
    import numpy as np
    from ase.io import read
    from pathlib import Path
    from ase.calculators.espresso import Espresso, EspressoProfile

    NUM_OMP = 1
    NUM_MPI = 192
    NODES_PER_CALC = 2

    def generate_kpts(atoms, periodic_3d=False):
        KSPACING = 0.10
        cell_lengths = atoms.cell.lengths()
        kpts = np.ceil(2 * np.pi / (cell_lengths * KSPACING)).astype(int)
        if not periodic_3d:
            kpts[2] = 1
        return kpts.tolist()

    def get_base_input_data(calculation_type, atoms_length=None):
        input_data = {
            "control": {
                "calculation": calculation_type,
                "restart_mode": "from_scratch",
                "outdir": "./",
                "tprnfor": True,
                "tstress": True,
                "disk_io": "none",
                "etot_conv_thr": 1e-10,
                "forc_conv_thr": 1e-10,
            },
            "system": {
                "ecutwfc": 80,
                "ecutrho": 800,
                "occupations": "smearing",
                "smearing": "cold",
                "degauss": 0.01,
                "input_dft": "PBE",
                "ibrav": 0,
            },
            "electrons": {
                "electron_maxstep": 200,
                "conv_thr": 1.0e-10,
                "mixing_mode": "plain",
                "mixing_beta": 0.8,
                "diagonalization": "david",
            },
            "ions": {"ion_dynamics": "bfgs", "upscale": 1e+6, "bfgs_ndim": 1},
            "cell": {"press_conv_thr": 0.1, "cell_dofree": "all"},
        }

        if calculation_type == "relax":
            input_data["control"].update({
                "etot_conv_thr": 1.0e-5,
                "forc_conv_thr": 3.88e-4,
                "tprnfor": True,
                "tstress": False,
            })
            input_data["electrons"].update({
                "mixing_beta": 0.2,
                "conv_thr": 1.0e-6,
                "mixing_mode": "local-TF",
                "diagonalization": "david",
            })
            input_data["system"].update({
                "ibrav": 0,
                "nbnd": int((3 * atoms_length / 2) * 1.5),
            })

        return input_data

    if is_bulk:
        atoms = read("Structures/bulk.xyz")
        directory = Path(f"Bulk_{os.getpid()}")
        calculation_type = "vc-relax"
        kpts = [30, 30, 30]
    else:
        atoms = read(structure_path)
        slab_name = Path(structure_path).stem
        directory = Path(f"Surface_{slab_name}_{os.getpid()}")
        calculation_type = "relax"
        kpts = generate_kpts(atoms, periodic_3d=False)

    directory.mkdir(exist_ok=True)

    srun_command = (
        f"srun --mpi=pmix "
        f"-N {NODES_PER_CALC} "
        f"--ntasks={NUM_MPI * NODES_PER_CALC} "
        f"--ntasks-per-node={NUM_MPI} "
        f"--cpus-per-task={NUM_OMP} "
        "/iridisfs/home/ba3g18/Repos/q-e/bin/pw.x"
    )
    
    input_data = get_base_input_data(
        calculation_type, len(atoms) if not is_bulk else None
    )

    profile = EspressoProfile(
        command=srun_command,
        pseudo_dir="/iridisfs/home/ba3g18/Repos/Pseudopotentials/pslibrary.1.0.0/pbe/PSEUDOPOTENTIALS",
    )
    
    calc = Espresso(
        input_data=input_data,
        pseudopotentials={"Li": "Li.pbe-sl-kjpaw_psl.1.0.0.UPF"},
        profile=profile,
        directory=str(directory),
        kpts=kpts,
    )

    atoms.calc = calc

    try:
        energy = atoms.get_potential_energy()

        if is_bulk:
            relaxed_atoms = read(f"{directory}/espresso.pwo")
            return {
                "success": True,
                "type": "bulk",
                "energy": energy,
                "atoms": relaxed_atoms,
                "error": None,
            }
        else:
            area = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
            surface_energy = (
                1
                / 2
                / area
                * (energy - ((len(atoms) / bulk_atoms_per_cell) * bulk_energy))
            )
            return {
                "success": True,
                "type": "slab",
                "slab": Path(structure_path).stem,
                "energy": energy,
                "surface_energy": surface_energy * 16.02,
                "area": area,
                "error": None,
            }
    except Exception as e:
        return {
            "success": False,
            "type": "bulk" if is_bulk else "slab",
            "slab": Path(structure_path).stem if not is_bulk else None,
            "error": str(e),
        }

def run_calculations():
    try:
        SLABS_DIR = Path("Structures")
        NUM_OMP = 1
        NODES_PER_CALC = 2
        MAX_SLURM_JOBS = 1
        NODES_PER_BLOCK = 28
        NUM_PARALLEL = NODES_PER_BLOCK // NODES_PER_CALC

        def setup_parsl_config():
            worker_init = """
module purge
module purge
module load pmix  
module load binutils/2.42  
module load intel-mpi/2021.12
module load mkl/2024.1.0
module load intel-compilers/2024.1.0  

source /iridisfs/home/ba3g18/.bashrc
conda activate Quacc_restarts

export OMP_PLACES=cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export I_MPI_FABRICS=shm:ofi
export I_MPI_COLL_INTRANODE=shm
export I_MPI_ADJUST_ALLGATHERV=1 
export MKL_THREADING_LAYER=sequential
export I_MPI_PMI_LIBRARY=/iridisfs/i6software/slurm/24.05.1/lib/libpmi.so 
            """
            return Config(
                executors=[
                    HighThroughputExecutor(
                        label="qe_worker",
                        max_workers_per_node=NUM_PARALLEL,
                        cores_per_worker=1e-6,
                        provider=SlurmProvider(
                            partition="batch",
                            account="special",
                            cores_per_node=NUM_OMP,
                            worker_init=worker_init,
                            walltime="60:00:00",
                            nodes_per_block=NODES_PER_BLOCK,
                            init_blocks=1,
                            min_blocks=1,
                            max_blocks=MAX_SLURM_JOBS,
                            launcher=SimpleLauncher(),
                        ),
                    )
                ],
            )

        parsl.load(setup_parsl_config())
        
        bulk_future = run_quantum_espresso(is_bulk=True)
        bulk_result = bulk_future.result()

        if not bulk_result["success"]:
            print(f"Bulk calculation failed: {bulk_result['error']}")
            return

        bulk_energy = bulk_result["energy"]
        bulk_atoms = bulk_result["atoms"]
        bulk_atoms_per_cell = len(bulk_atoms)

        slab_files = [f for f in SLABS_DIR.glob("*.xyz") if f.stem != "bulk"]
        slab_futures = []

        for slab_file in slab_files:
            future = run_quantum_espresso(
                structure_path=str(slab_file),
                is_bulk=False,
                bulk_energy=bulk_energy,
                bulk_atoms_per_cell=bulk_atoms_per_cell,
            )
            slab_futures.append((future, slab_file.stem))

        results = []
        failed_slabs = []

        with tqdm(total=len(slab_futures), desc="Completed") as pbar:
            for future, slab_name in slab_futures:
                try:
                    result = future.result()
                    if result["success"]:
                        results.append(result)
                        print(
                            f"\nCompleted {result['slab']}: {result['surface_energy']:.4f} J/m^2"
                        )
                    else:
                        failed_slabs.append((slab_name, result["error"]))
                except Exception as e:
                    failed_slabs.append((slab_name, str(e)))
                pbar.update(1)

        if results:
            results.sort(key=lambda x: x["surface_energy"])
            with open("surface_energies.txt", "w") as f:
                f.write("Slab\tSurface Energy (J/m^2)\tArea (Å²)\n")
                for result in results:
                    f.write(
                        f"{result['slab']}\t{result['surface_energy']:.4f}\t{result['area']:.4f}\n"
                    )

        if failed_slabs:
            print("\nFailed calculations:")
            for slab_name, error in failed_slabs:
                print(f"- {slab_name}: {error}")

    finally:
        parsl.dfk().cleanup()
        parsl.clear()

if __name__ == "__main__":
    run_calculations()