# JAX port — change summary

This document summarises all modifications made to comet-emu to make the
emulator JAX-traceable (compatible with `jax.jit`, `jax.grad`, `jax.vmap`).

---

## `comet/cosmology.py` — new `JAXCosmology` class (~250 lines added)

JAX-traceable analogue of `Cosmology`.  All cosmological parameters are stored
as JAX arrays; all background quantities are computed with JAX operations and
are therefore fully differentiable:

- Growth factor and growth rate (`growth_factor`, `growth_factor_lambda`,
  `growth_factor_w0`, `growth_factor_w0wa`)
- Alcock-Paczynski distortion parameters (`compute_ap_params`)
- Comoving transverse distance and Hubble rate (`comoving_transverse_distance`,
  `Hz`)
- Growth amplitude relative to a fiducial cosmology (`compute_growth_amplitude`)
- Supports `de_model` in `{'lambda', 'w0', 'w0wa'}`

---

## `comet/splines.py` — JAX evaluation path in `Splines` (~60 lines added)

- `_is_jax_value` helper
- `build()`: stores `x_raw` as native-endian `float64` (FITS big-endian arrays
  are converted here); early-exits without building a scipy spline when inputs
  are JAX arrays
- `_eval_varx_jax()`: new method for JAX-native spline evaluation using
  `jnp.interp` per column, returning a JAX array
- `_linear_bucket_table()`: segment lookup with buckets uniform in k (no log
  per node), cached on the breakpoints, for the JAX bispectrum kernels

---

## `comet/bispectrum.py` — JAX fused kernels

Upstream's file, except the JAX kernels `_fused_moments_jax` /
`_fused_moments_diagrams_jax`, rewritten for XLA:CPU. Same values as the
numba kernels to rounding (≤ 2e-15, checked on Bell/BX, VDG_infty and EFT
with both counterterm types, `extra_damping`, `'logk3'`, a fixed `Pdw_mu`,
and Scoccimarro); the numba kernels are unchanged:

- spline table of the JAX kernels from `_jax_spline_table` (linear buckets,
  segment index and breakpoint tabulated per bucket); `_ppoly_eval_jax` only
  takes a log for the power-law tails
- leg 1, which only depends on (pair, mu1), evaluated once on that grid
  (`_legs_jax`)
- model terms selected statically (`_jax_kernel_flags`): no W_infty
  exponentials for EFT, no `cB1`/`cB2` terms unless IvaPhiNis, no `cnloB`
  terms unless EggLeeSco
- reciprocals and `rsqrt` instead of most divisions, `den**1.5` written with
  `rsqrt`
- diagram kernel: each diagram column is cheap arithmetic on per-node pieces
  shared by all columns (`_pair_pieces_jax`, `_stoch_part_jax`), with the Pdw
  of each leg kept in memory by `_keep_jax`. On XLA:CPU
  `lax.optimization_barrier` does **not** stop fusion, so the previous
  barriers were no-ops and every diagram column re-evaluated the spline from
  16 stored coefficient arrays per leg; dividing by a traced 1.0 does, since
  XLA never duplicates a division
- diagram kernel on the (pair, mu12, mu1, phi) node grid (`_legs_jax(...,
  mu12_major=True)`), so that the (mu1, phi) nodes contracted with `ang` are
  contiguous: each diagram column is projected by a plain matmul with `ang`
  and a small contraction with `rad`. On the (pair, mu1, mu12, phi) grid the
  einsum made XLA stack the diagrams and transpose the whole stack (~70 MB)
  first
- optional `damp` argument of both kernels (`extra_damping`)

Speed-up of the forward pass of the full model (emulator included, jit),
for 2628 pairs, 5 multipoles and quad (7, 16, 5) `'k3'` (old and new code
timed alternately on the same machine; the factors depend on the hardware
and on the number of pairs):

| | VDG_infty | EFT |
|---|---|---|
| `Bell_Sugi` | x1.4 | x1.5 |
| `BX_ell_Sugi`, desilike's 14 diagrams | x5.2 | x6.1 |
| `BX_ell_Sugi`, all diagrams | x5.7 | x7.7 |
| `BX_ell_Sugi`, desilike's diagrams, (4, 28, 3) `'logk3'` | | x4.4 |

The emulator itself (mostly the GP triangular solves of the stable
prediction) is unchanged and takes about a third of `Bell_Sugi` now, so
`Bell_Sugi` gains less with fewer pairs.

---

## `comet/PTEmu.py` — main changes (~1 000 lines added, ~350 modified)

### New module-level utilities

| Function | Purpose |
|---|---|
| `_is_jax(x)` | True if `x` is a JAX array or tracer |
| `_xp_zeros_like`, `_xp_ones_like` | numpy/JAX dispatch |
| `_xp_stack(arrays)` | stack 1-D arrays; works for numpy and JAX |
| `_xp_safe_param(x, ref)` | coerce scalar to 1-D array in the right array module |
| `_jax_gp_predict(X_test, gp_dict)` | JAX GP prediction (RBF + Matérn-5/2 kernel); replaces sklearn at trace time |
| `_jax_legendre_outer(ells, x)` | vectorised Legendre polynomials P₀, P₂, P₄, P₆ |

### New methods on `PTEmu`

| Method | Purpose |
|---|---|
| `_init_jax_gp_arrays()` | Extracts GP hyperparameters from fitted sklearn models into plain numpy arrays so they can be captured as static constants during JAX tracing |
| `_jax_Pdw_ppoly_table(ell_for_recon)` | JAX counterpart of `_PdwEvaluator.ppoly_table`: the power-basis table of the exact Pdw spline (as a polynomial in mu^2 for `Pdw_mu='leg'`), built from the traced emulator outputs by `splines.ppoly_table_jax` (the fixed linear map of `splines._cubic_ppoly_basis`) |
| `_jax_sugi(...)` | Shared JAX Sugiyama path: static geometry and factorised-projection weights, traced Pdw table / AP / biases, fed to `BispectrumNum._fused_moments_jax` (or `_fused_moments_diagrams_jax`), called once on all the pairs |
| `_jax_bell_sugi(...)` | JAX-native `Bell_Sugi`, on `_jax_sugi` |
| `_jax_bx_ell_sugi(...)` | JAX-native `BX_ell_Sugi`, on `_jax_sugi`; when `X_list` is provided returns `{ll: Array(npair, nx)}` matching the numpy path's format |

### Dual-path modifications to existing methods

| Method | Change |
|---|---|
| `_load_emulator_data` | `k_table` (big-endian `>f8`) and `P6` (big-endian `>f4`) are converted to native `float64` at FITS load time so `jnp.asarray` accepts them |
| `_eval_emulator` | JAX path: GP prediction via `_jax_gp_predict`; amplitude scaling and growth rate computed via `JAXCosmology`; supports all three `de_model` values; handles `nonu` models (shape GP, `s12` derivation) |
| `_update_params` | Clears stale JAX tracers from `self.params` before re-running; JAX-aware parameter assembly |
| `_update_AP_params` | JAX path: computes AP parameters via `JAXCosmology.compute_ap_params`; accepts pre-computed `q_tr_lo` tuple |
| `_update_bias_params` | JAX-aware update of bias / RSD / observational-systematic parameters |
| `_W_kurt` | JAX branch (`jnp.sqrt`, `jnp.exp`) for kurtosis FoG damping |
| `_W_obs_syst` | Uses `_xp_ones_like` so the `f_out` damping factor is JAX-safe |
| `_PX_ell6_novir_noAP` | JAX path: P6 octopole contribution assembled via `jnp.einsum` on the natively-typed `self.P6` array |
| `PX_ell` | JAX build path: `PXNL_ell` initialised as `jnp.zeros`; per-diagram emulator outputs accumulated via `jnp.asarray`; `_eval_varx_jax` used for spline evaluation; LOS averaging assembled in JAX |

---

## `comet/__init__.py`

Minor additions (6 lines) — exposes `JAXCosmology` and utility helpers at the
package level.

## `comet/tables.py`

One-line fix for JAX compatibility.

---

## Scope / limitations

- Supported `model` values: `VDG_infty`, `EFT` (and `_nonu` variants).  `RS`
  (real-space) is not wired through desilike.
- Supported `de_model` values: `'lambda'`, `'w0'`, `'w0wa'`.
- Bispectrum: `Bell_Sugi` / `BX_ell_Sugi` with `model='VDG_infty'` or
  `'EFT'` (both `cnloB` types), any `mu12_transform`, and Pdw either at each
  leg's mu (`change_Pdw_mu('leg')`, default) or at a fixed mu
  (`change_Pdw_mu(0.6)` reproduces the previous results). `Bell_Scocc` is
  numpy only.
- `use_Mpc=False` (desilike always uses `h⁻¹ Mpc` units).
- JIT vs eager numerical agreement: power-spectrum poles agree at
  `rtol ~ 2×10⁻³` (XLA floating-point fusion in the GP path is amplified
  by the near-cancellation in the bias decomposition); bispectrum poles at
  similar tolerance.
