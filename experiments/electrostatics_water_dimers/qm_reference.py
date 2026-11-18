"""Reference dipoles and static polarizabilities for the water clusters (PySCF).

Level: PBE0/aug-cc-pVDZ, one level for BOTH quantities, so that a model trained
on the dipoles can be tested on the (untrained) polarizabilities without a
level-of-theory mismatch between the two.  The energies and forces stay the
MB-pol (CCSD(T)-fitted) ones of the source data.

    mu    : SCF dipole at zero field, e*A (the clusters are neutral, so it does
            not depend on the origin).
    alpha : alpha_ab = d mu_a / d F_b by central differences in a uniform field
            F = +-0.002 au along each axis: H' = H + F . r for the electrons, so
            the induced dipole is +alpha F.  Truncation O(F^2) ~ 1e-6 relative.
            Symmetrised; the antisymmetric part is reported as a check.

Usage: python qm_reference.py SPLIT [N_WORKERS] [THREADS_PER_WORKER]
Writes data_qm/SPLIT.xyz (the input frames with info['dipole'] and
info['polarizability'] added) and data_qm/SPLIT.log.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

BOHR = 0.529177210903          # A
FIELD = 0.002                  # au
BASIS = "aug-cc-pvdz"
XC = "pbe0"
HERE = os.path.dirname(os.path.abspath(__file__))


def scf_dipole(mol, field, dm0=None):
    from pyscf import dft

    mf = dft.RKS(mol)
    mf.xc = XC
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-7
    mf.max_cycle = 200
    if field is not None:
        with mol.with_common_orig((0.0, 0.0, 0.0)):
            r = mol.intor_symmetric("int1e_r", comp=3)
        h1 = mf.get_hcore() + np.einsum("x,xij->ij", field, r)
        mf.get_hcore = lambda *args, **kwargs: h1
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    dm = mf.make_rdm1()
    # Dipole of nuclei + electrons about the origin, au.  Computed directly so
    # the field-modified hcore cannot leak into it.
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        r = mol.intor_symmetric("int1e_r", comp=3)
    electronic = -np.einsum("xij,ji->x", r, dm).real
    nuclear = np.einsum("i,ix->x", mol.atom_charges(), mol.atom_coords())
    return electronic + nuclear, dm, mf.e_tot


def frame_reference(args):
    index, symbols, positions, threads = args
    os.environ["OMP_NUM_THREADS"] = str(threads)
    from pyscf import gto, lib

    lib.num_threads(threads)
    t0 = time.perf_counter()
    mol = gto.M(atom=list(zip(symbols, positions.tolist())), basis=BASIS, unit="Angstrom",
                verbose=0, charge=0, spin=0)
    mu0, dm0, e0 = scf_dipole(mol, None)
    alpha = np.zeros((3, 3))
    for b in range(3):
        field = np.zeros(3)
        field[b] = FIELD
        mu_plus, _, _ = scf_dipole(mol, field, dm0)
        mu_minus, _, _ = scf_dipole(mol, -field, dm0)
        alpha[:, b] = (mu_plus - mu_minus) / (2.0 * FIELD)
    asym = float(np.abs(alpha - alpha.T).max())
    alpha = 0.5 * (alpha + alpha.T)
    return {
        "index": index,
        "dipole_eA": mu0 * BOHR,
        "polarizability_A3": alpha * BOHR ** 3,
        "asym_au": asym,
        "e_scf_hartree": e0,
        "seconds": time.perf_counter() - t0,
        "nbas": mol.nao_nr(),
    }


def main():
    from ase.io import read, write

    split = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    src = os.path.join(HERE, "data", f"{split}.xyz") if split in ("train", "valid", "test") else split
    name = os.path.splitext(os.path.basename(src))[0]
    frames = read(src, ":")
    out_dir = os.path.join(HERE, "data_qm")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(k, a.get_chemical_symbols(), a.positions.copy(), threads) for k, a in enumerate(frames)]
    results = [None] * len(frames)
    t0 = time.perf_counter()
    with open(os.path.join(out_dir, f"{name}.log"), "w") as log, Pool(workers) as pool:
        for res in pool.imap_unordered(frame_reference, jobs):
            results[res["index"]] = res
            log.write(f"{res['index']:4d} n={len(frames[res['index']]):2d} nbas={res['nbas']:4d} "
                      f"{res['seconds']:7.1f}s |mu|={np.linalg.norm(res['dipole_eA']):.5f} eA "
                      f"alpha_iso={np.trace(res['polarizability_A3']) / 3:.5f} A3 asym={res['asym_au']:.1e}\n")
            log.flush()
    for a, res in zip(frames, results):
        a.info["dipole"] = res["dipole_eA"]
        a.info["polarizability"] = res["polarizability_A3"].reshape(-1)
        a.info["qm_level"] = f"{XC}/{BASIS}"
    write(os.path.join(out_dir, f"{name}.xyz"), frames, format="extxyz")
    print(f"{name}: {len(frames)} frames in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
