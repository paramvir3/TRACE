# TRACE v4: Connected Fixed-Environment Tensorial Cross-Attention

TRACE v4 is a research architecture, versioned separately from v3. It remains
strictly local: no key, value, or update is computed from an evolved neighbor
state. Its added capacity is entirely a function of the raw species and
geometry in one cutoff environment.

## Connected pair-density token

For an equivariant edge density `a_ij` and its neighbor sum
`A_i = sum_j a_ij`, let `B` be a learned O(3)-equivariant bilinear tensor
product. The usual quadratic density product contains same-neighbor terms:

```math
B(A_i, A_i) = sum_{j,k} B(a_ij, a_ik).
```

TRACE v4 additionally constructs the exact distinct-neighbor part

```math
D_i^(3) = B(A_i, A_i) - sum_j B(a_ij, a_ij)
          = sum_{j != k} B(a_ij, a_ik).
```

The implementation uses one per-center density product and one scatter-reduced
edge-self product. This is an exact pair correction for the chosen learned
bilinear map. It is not described as a full higher-order cumulant ACE basis;
that would require all set partitions and has a different computational cost.
Higher v4 correlation tokens recurse from `D_i^(3)` with `A_i`.

## Shared-value multi-query attention

For fixed token `t_ia`, each head still computes its own invariant score
`s_iah` and cutoff-preserving attention `alpha_iah`. A single equivariant value
is shared across heads:

```math
v_ia = TP(h_i, t_ia),
u_i = sum_a [H^-1 sum_h alpha_iah sigmoid(g_ih)] v_ia.
```

`g_ih` is a scalar function of the receiving center. Thus the bracket is an
invariant scalar and `u_i` transforms exactly as `v_ia`. This replaces `H`
head-wise tensor products by one tensor product per token. It is a complexity
optimization, not a claim of equal accuracy to v3; it must be evaluated at
matched force-error and throughput.

## Smooth shell factor and tensor SwiGLU

The score includes smooth Gaussian shell features,

```math
g_s(r) = exp[-(r-c_s)^2/(2 sigma_s^2)] f_c(r),
s_iah <- s_iah + sum_s w_hs g_s(r_ia).
```

They are continuous radial logit terms, so they do not create a hard token
selection discontinuity. Non-scalar channels are modulated only by invariant
gates. For every irrep copy `x_c^(l,p)`,

```math
x_c^(l,p) <- [1 + 0.1 tanh(SiLU(a_c) b_c)] x_c^(l,p),
```

where `a_c,b_c` are functions of scalar channels and squared irrep norms.
The multiplier is invariant, so parity and O(3) transformation laws remain
unchanged. The final gate is identity initialized.

## Correlation-rank curriculum

The fixed correlation space can be activated from a nested prefix of irrep
multiplicities to the full rank. Entire copies of an irrep are masked together,
never individual `m` components; the curriculum therefore preserves O(3)
equivariance. This is a training-only schedule and does not reduce the full-rank
inference cost unless followed by structured pruning and retraining.

## Required ablations

v4 is a hypothesis, not an established accuracy result. Report, at matched
parameter count and measured throughput: v3, v4 without pair correction, v4
without shell factor, v4 without tensor SwiGLU, and v4 without the rank
curriculum. Evaluate energy, force, stress, EOS, NVE drift, and the intended
phase/dynamics observables.
