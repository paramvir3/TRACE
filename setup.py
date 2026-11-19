from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent

setup(
    name="transformers-ace",
    version="2.3.0",
    description="Local equivariant transformer potentials built on ACE descriptors",
    long_description=(ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/paramvir3/TRACE",
    license="MIT",
    packages=find_packages(),
    # Runtime data that trace_redox loads by a path relative to its own
    # modules (penetration.py, polarizability.py, frame_pair.py, wall.py,
    # two_site.py): the first-principles ion densities, polarizabilities and
    # pair tables.  two_site.json records the sha256 of four of them, so they
    # must be installed byte-identical.
    package_data={"trace_redox": ["data/*.npz", "data/*.json"]},
    # 3.8 is the floor this is actually exercised on (the reference conda
    # environment is 3.8.16); the upper bound is untested rather than known bad.
    python_requires=">=3.8,<3.14",
    install_requires=[
        # e3nn 0.4.x AND 0.6.x are both supported, and this range is measured,
        # not assumed.  An earlier pin read ">=0.6.0,<0.7.0" on the grounds that
        # "tensor-product normalisation changed between 0.4.x and 0.6.x, so
        # checkpoints are not interchangeable".  That is false for this
        # codebase: the two published CsPbI3 checkpoints, trained under 0.4.4,
        # evaluate to BIT-IDENTICAL energies on 0.6.0 (dE = 0.0e+00 eV on a
        # 10-atom perovskite cell), with forces differing by 1.9e-07 in norm --
        # float32 round-off on a float32 model.  The conventions the package
        # depends on are identical on both: the 1o basis is cartesian xyz, the
        # degree-1 spherical harmonics are the identity, and the wigner_3j
        # tensors agree entry for entry, signs included.
        #
        # What DOES differ is the precision of o3.D_from_matrix -- 6.9e-18 on
        # 0.4.4, 3.0e-15 on 0.6.0, because 0.6.0 rebuilds the Wigner matrices
        # through so3_generators at the default dtype instead of reading a
        # float64 cache.  It is used only by tests, never in the forward path.
        # See tests/electrostatics/test_e3nn_conventions.py, which asserts the
        # conventions on whichever series is installed.
        "torch>=2.2.0",
        "e3nn>=0.4.4,<0.7.0",
        "ase>=3.28",
        "numpy>=1.23",
        "scipy>=1.10",
        "pyyaml>=6.0",
        "matplotlib>=3.6",
        "packaging>=23",
    ],
    extras_require={
        "accelerated": ["torch-scatter"],
        "test": ["pytest>=8"],
        # The CUDA/AOT deployment path (transformers_ace.aot ->
        # torch._inductor.package.package_aoti -> the .pt2 the LAMMPS pair style
        # loads) needs a newer torch than the eager path does.  torch 2.2.0
        # raises ModuleNotFoundError for torch._inductor.package, and
        # tests/test_aot_export.py SKIPS with that message rather than failing,
        # because eager training and inference are unaffected by its absence.
        "aot": ["torch>=2.5"],
    },
)
