# TRACE GPU Production Backend

## Current status

`pair_style transformers_ace` is a correctness-oriented LibTorch interface. It
uses LAMMPS full neighbor lists and evaluates local owned-atom energies with
ghost atoms. It correctly requires `newton on` for MPI runs, differentiates one
scalar energy to obtain conservative forces and virial, and maps one MPI rank
to one visible GPU. It is **not** a million-atom H100 implementation: every
force evaluation presently allocates CPU tensors, copies positions/types/edges
to CUDA, builds a dynamic autograd graph, and copies forces and strain gradients
back to CPU.

This distinction is deliberate. Strict physical locality is necessary for
linear weak scaling, but it is insufficient without a device-resident execution
path.

## Required production design

A production `pair_style transformers_ace/kk` backend must satisfy all of the
following conditions.

1. Use LAMMPS Kokkos CUDA device views for positions, types, neighbor lists,
   forces, and virial. No CPU tensor construction or host-device transfer is
   permitted in the per-timestep force path.
2. Build a directed CSR edge list on the GPU from the local-plus-ghost full
   neighbor list. Reuse capacity-managed device buffers rather than allocating
   edge tensors on every step.
3. Execute radial functions, real spherical harmonics, ACE-density reductions,
   Clebsch-Gordan products, segment softmax, tensorial attention, atomic-energy
   reduction, and the position/strain backward pass on the GPU. These kernels
   must preserve the same O(3) conventions as the Python reference.
4. Keep force and virial tensors device resident. Use GPU-aware MPI for halo
   exchange and reverse communication with one MPI rank per GPU. A rank may
   communicate only its cutoff halo; attention depth must not enlarge that halo.
5. Export a static inference-and-derivative program ahead of time. The runtime
   must not invoke Python, TorchScript tracing, or dynamic autograd graph
   construction on each molecular-dynamics step.

The most practical implementation route is a Kokkos pair style coupled to an
ahead-of-time exported TRACE program, with fused or specialized GPU kernels for
the sparse tensor products and segment reductions. A direct port of the current
generic LibTorch operations would be correct but is unlikely to be competitive
on H100 hardware.

## Verification gates

The backend is not allowed to claim production H100 scaling until it passes:

- energy, force, and six-component virial agreement against the Python
  reference for periodic cells, including triclinic cells and cutoff-boundary
  configurations;
- agreement between one GPU and multi-GPU domain decomposition at `run 0` and
  over a short NVE trajectory;
- no host-device copies or per-step allocations in an Nsight Systems trace;
- strong-scaling and weak-scaling measurements with fixed density/cutoff,
  reported with atoms per GPU, neighbor count, precision, and throughput;
- a long NVE energy-drift test and NPT equation-of-state/pressure test.

Until these gates pass, the native pair style should be used for correctness,
small-to-medium simulations, and LAMMPS/PLUMED workflow validation rather than
as evidence of million-atom performance.
