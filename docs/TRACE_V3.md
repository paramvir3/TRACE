# TRACE v3: Fixed-Environment Tensorial Cross-Attention

TRACE v3 is a strictly local energy model.  It computes a frozen set of
equivariant tokens for every center atom and applies cross-attention only from
that center's evolving state to those tokens.  It does not use an updated
neighbor state as a key, value, or message.

For a directed neighbor edge `j -> i`, let the species embedding of the
neighbor be \(\mathbf e(Z_j)\), and let \(R_n\) and \(Y_{\ell m}\) denote the
radial and real spherical-harmonic bases.  In the implementation, a radial
network produces the weights of an equivariant tensor product.  Thus the edge
density is, explicitly,

\[
 \mathbf a_{ij} =
 \operatorname{TP}_{\mathbf w(r_{ij})}\!\left[
 \mathbf e(Z_j),\,f_c(r_{ij})R(r_{ij})Y(\widehat{\mathbf r}_{ij})\right].
\]

The ACE density is \(\mathbf A_i=\sum_j\mathbf a_{ij}\).  Recursive learned
Clebsch-Gordan contractions of this density generate fixed *correlation-degree*
features.  The first density has maximum neighbor body order two; multiplying
by \(\mathbf A_i\) raises that maximum by one, while repeated-neighbor terms
also retain lower-body contributions.  TRACE v3 provides both the directed
edge tokens and one projected token for every retained correlation degree to
the attention block.  These tokens are produced before attention and stay fixed
for every attention layer.  This is a learned, truncated ACE-density
correlation basis, not an enumeration of a complete sparse ACE U-basis.

Let `t_ia` be any such fixed token and `h_i^(t)` the center feature.  Matching
irreps are contracted to an invariant attention score,

\[
s_{iah} = \sum_{L,p} w_{hLp}
\left[Q_h^{Lp}\mathbf h_i^{(t)} \otimes_{CG}
K_h^{Lp}\mathbf t_{ia}\right]_{0e} + b_h(r_{ia}).
\]

The value is a center-conditioned equivariant tensor product,

\[
\mathbf v_{iah} = \operatorname{TP}_h(
\mathbf h_i^{(t)}, \mathbf t_{ia}),
\qquad
\mathbf h_i^{(t+1)} = \Phi\!\left(
\mathbf h_i^{(t)} + \sum_{a,h}\alpha_{iah}\mathbf v_{iah}\right).
\]

Because no term contains `h_j^(t)`, induction gives

\[
\mathbf h_i^{(t)} = F_t\!\left(Z_i,
\{(Z_j,\mathbf r_{ij}) : j\in\mathcal N(i)\}\right),
\]

for every depth `t`.  Attention depth therefore does not change the physical
receptive field or require an additional inter-domain halo exchange.

The final energy readout explicitly concatenates the scalar irreps and squared
norms of all higher-order irreps.  Forces and stress are derivatives of the
same scalar energy.  Architecture-version-3 checkpoints are not compatible
with v1 or v2 weights.

## Running

Use one of the supplied configurations from the repository root:

```bash
python train.py --config configs/trace_v3_cspbi3.yaml
python train.py --config configs/trace_v3_water_ccsd.yaml
```

Both configurations use only the local datasets supplied in this repository.
