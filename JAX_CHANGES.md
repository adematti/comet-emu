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

---

## `comet/bispectrum.py` — JAX Pdw evaluation in the fused kernels

Upstream's file, except the JAX spline evaluation used by
`_fused_moments_jax` / `_fused_moments_diagrams_jax` (same values to
rounding; the numba kernels are unchanged):

- `_ppoly_eval_jax`: coefficients gathered from the flattened table at a
  computed index, and one log + one exp per leg (shared by the bucket lookup
  and both power-law tails) instead of three logs and two exps
- `_ppoly_eval_legs_jax`: the three legs behind `lax.optimization_barrier`,
  so XLA evaluates each once instead of inside every diagram's fusion
- optional `damp` argument of both kernels (`extra_damping`)

Kernel time, 2628 pairs, `Pdw_mu='leg'`: `Bell_Sugi` 46 → 27 ms, its
gradient 344 → 283 ms, `BX_ell_Sugi` 229 → 110 ms (VDG_infty).

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
