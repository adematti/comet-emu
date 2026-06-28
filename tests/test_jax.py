"""
Tests and benchmarks for JAX-traceable PTEmu methods (Pell, Bell_Sugi).

Covers:
  - Correctness: JAX path vs numpy path (tolerance: GP noise floor ~1-2 %)
  - jax.jit: result identical to eager; timing vs numpy baseline
  - jax.grad: finite gradient wrt cosmological and bias parameters
  - jax.vmap: batch over N cosmologies, result matches N sequential calls

Run as a script:
    python tests/test_jax.py

Or with pytest:
    pytest tests/test_jax.py -v
"""

import warnings
import time
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from comet.PTEmu import PTEmu

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

MODEL    = 'VDG_infty'
DE_MODEL = 'lambda'

P_FID = dict(
    wb=0.02237, wc=0.1200, ns=0.9649, Mnu=0.06,
    As=2.083e-9, h=0.6736, Ok=0.0, w0=-1.0, wa=0.0, z=0.8,
)

P_BASE = dict(
    wb=0.02237, wc=0.1200, ns=0.9649, Mnu=0.06, As=2.083e-9,
    b1=2.0, b2=0.1, g2=-0.1, g21=0.0,
    c0=0.5, c2=0.3, c4=0.1, cnlo=0.0,
    NP0=300.0, NP20=200.0, NP22=100.0,
    cnloB=0.0, NB0=0.0, MB0=0.0, cB1=0.0, cB2=0.0,
    avir=0.0, avirB=0.0,
    sigma_z=0.0, gamma_z=1.0, f_out=0.0,
    h=0.6736, Ok=0.0, q_tr=1.0, q_lo=1.0,
    z=0.8,   # kept concrete (numpy) — z must not be a JAX tracer
)

ELL       = [0, 2, 4]
K         = np.logspace(-2, 0, 60)
BELL_PAIR = np.array([[0.05, 0.07], [0.10, 0.12]])
BELL_ELL  = ((0, 0, 0), (2, 0, 2))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np_params():
    return {k: np.atleast_1d(np.float64(v)) for k, v in P_BASE.items()}


def _jax_params(**overrides):
    p = {k: jnp.atleast_1d(jnp.array(np.float64(v)))
         for k, v in P_BASE.items() if k != 'z'}
    p['z'] = np.atleast_1d(np.float64(P_BASE['z']))
    p.update(overrides)
    return p


@pytest.fixture(scope='module')
def emu():
    e = PTEmu(model=MODEL, use_Mpc=True)
    p_fid = {k: np.atleast_1d(np.float64(v)) for k, v in P_FID.items()}
    e.define_fiducial_cosmology(params_fid=p_fid, de_model=DE_MODEL)
    return e


def _pell_sum(emu, params, ell=None):
    ell = ell or ELL
    res = emu.Pell(K, params, ell, de_model=DE_MODEL)
    return sum(jnp.sum(jnp.asarray(res[f'ell{l}'])) for l in ell)


# ---------------------------------------------------------------------------
# 1. Pell — correctness
# ---------------------------------------------------------------------------

def test_jax_matches_numpy(emu):
    """JAX and numpy Pell must agree to within ~2 % (GP noise floor)."""
    ref    = emu.Pell(K, _np_params(),  ELL, de_model=DE_MODEL)
    result = emu.Pell(K, _jax_params(), ELL, de_model=DE_MODEL)
    for l in ELL:
        r_np  = np.asarray(ref[f'ell{l}']).ravel()
        r_jx  = np.asarray(result[f'ell{l}']).ravel()
        scale = np.max(np.abs(r_np))
        err   = np.max(np.abs(r_np - r_jx)) / scale
        assert err < 0.02, f'ell{l}: JAX vs numpy error {err:.4%} of peak > 2 %'


def test_jit_correctness(emu):
    """jax.jit result must match eager JAX to machine precision."""
    wc0 = jnp.float64(P_BASE['wc'])

    def fn(wc):
        p = _jax_params(wc=jnp.atleast_1d(wc))
        return jnp.concatenate(
            [jnp.ravel(jnp.asarray(emu.Pell(K, p, [l], de_model=DE_MODEL)[f'ell{l}']))
             for l in ELL])

    eager  = fn(wc0)
    jitted = jax.jit(fn)(wc0)
    np.testing.assert_allclose(np.array(eager), np.array(jitted), rtol=1e-5,
                               err_msg='jit result differs from eager result')


# ---------------------------------------------------------------------------
# 2. Pell — gradients
# ---------------------------------------------------------------------------

COSMO_KEYS = ('wc', 'wb', 'ns', 'As', 'h', 'Ok')
BIAS_KEYS  = ('b1', 'NP0')


def _fn_for_key(emu, key, ell=None):
    _ell = ell or [0]
    def fn(val):
        p = _jax_params(**{key: jnp.atleast_1d(val)})
        return _pell_sum(emu, p, _ell)
    return fn


@pytest.mark.parametrize('key', COSMO_KEYS)
def test_grad_cosmo(emu, key):
    val = jnp.float64(P_BASE[key])
    g   = jax.jit(jax.grad(_fn_for_key(emu, key)))(val)
    assert not bool(jnp.isnan(g)), f'd/d({key}) is NaN'
    assert not bool(jnp.isinf(g)), f'd/d({key}) is Inf'


@pytest.mark.parametrize('key', BIAS_KEYS)
def test_grad_bias(emu, key):
    val = jnp.float64(P_BASE[key])
    g   = jax.jit(jax.grad(_fn_for_key(emu, key)))(val)
    assert not bool(jnp.isnan(g)), f'd/d({key}) is NaN'
    assert not bool(jnp.isinf(g)), f'd/d({key}) is Inf'


def test_grad_all_ell(emu):
    val = jnp.float64(P_BASE['wc'])
    g   = jax.jit(jax.grad(_fn_for_key(emu, 'wc', ell=[0, 2, 4])))(val)
    assert not bool(jnp.isnan(g)), 'd/d(wc) over ell=[0,2,4] is NaN'
    assert not bool(jnp.isinf(g)), 'd/d(wc) over ell=[0,2,4] is Inf'


def test_grad_jit_matches_eager_grad(emu):
    fn      = _fn_for_key(emu, 'wc')
    val     = jnp.float64(P_BASE['wc'])
    g_eager = jax.grad(fn)(val)
    g_jit   = jax.jit(jax.grad(fn))(val)
    np.testing.assert_allclose(float(g_eager), float(g_jit), rtol=1e-5,
                               err_msg='jit grad differs from eager grad')


# ---------------------------------------------------------------------------
# 3. Pell — vmap
# ---------------------------------------------------------------------------

def _fn_single_pell(emu):
    def fn(wc):
        p = _jax_params(wc=jnp.atleast_1d(wc))
        return jnp.ravel(jnp.asarray(emu.Pell(K, p, [0], de_model=DE_MODEL)['ell0']))
    return fn


def test_vmap_correctness(emu):
    wcs = jnp.array([P_BASE['wc'] + 0.005 * i for i in range(5)], dtype=jnp.float64)
    fn_single     = _fn_single_pell(emu)
    fn_batch      = jax.jit(jax.vmap(fn_single))
    fn_single_jit = jax.jit(fn_single)
    batch_out = fn_batch(wcs)
    for i, wc in enumerate(wcs):
        single_out = fn_single_jit(wc)
        np.testing.assert_allclose(
            np.array(batch_out[i]), np.array(single_out), rtol=1e-5,
            err_msg=f'vmap[{i}] differs from single call at wc={float(wc):.4f}')


# ---------------------------------------------------------------------------
# 4. Bell_Sugi — correctness and gradients
# ---------------------------------------------------------------------------

def test_bell_sugi_jax_matches_numpy(emu):
    """JAX Bell_Sugi must match numpy to within ~2 % (GP noise floor)."""
    res_np  = emu.Bell_Sugi(BELL_PAIR, _np_params(),  ell=BELL_ELL, de_model=DE_MODEL)
    res_jax = emu.Bell_Sugi(BELL_PAIR, _jax_params(), ell=BELL_ELL, de_model=DE_MODEL)
    for ll in BELL_ELL:
        r_np  = np.asarray(res_np[ll]).ravel()
        r_jax = np.asarray(res_jax[ll]).ravel()
        scale = np.max(np.abs(r_np)) + 1e-30
        err   = np.max(np.abs(r_np - r_jax)) / scale
        assert err < 0.02, f'Bell_Sugi {ll}: JAX vs numpy error {err:.4%} > 2%'


def test_bell_sugi_grad_b1(emu):
    def fn(b1):
        p   = _jax_params(b1=jnp.atleast_1d(b1))
        res = emu.Bell_Sugi(BELL_PAIR, p, ell=BELL_ELL, de_model=DE_MODEL)
        return sum(jnp.sum(jnp.asarray(res[ll])) for ll in BELL_ELL)
    g = jax.jit(jax.grad(fn))(jnp.float64(P_BASE['b1']))
    assert not bool(jnp.isnan(g)), 'Bell_Sugi d/d(b1) is NaN'
    assert not bool(jnp.isinf(g)), 'Bell_Sugi d/d(b1) is Inf'


def test_bell_sugi_grad_wc(emu):
    def fn(wc):
        p   = _jax_params(wc=jnp.atleast_1d(wc))
        res = emu.Bell_Sugi(BELL_PAIR, p, ell=BELL_ELL, de_model=DE_MODEL)
        return sum(jnp.sum(jnp.asarray(res[ll])) for ll in BELL_ELL)
    g = jax.jit(jax.grad(fn))(jnp.float64(P_BASE['wc']))
    assert not bool(jnp.isnan(g)), 'Bell_Sugi d/d(wc) is NaN'
    assert not bool(jnp.isinf(g)), 'Bell_Sugi d/d(wc) is Inf'


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _time_fn(fn, args_list, n_reps=None, block=False):
    """Run fn(*args) for each args in args_list and return mean wall time."""
    if n_reps is not None:
        args_list = list(args_list) * n_reps
    t0 = time.perf_counter()
    for args in args_list:
        result = fn(*args) if isinstance(args, (list, tuple)) else fn(args)
        if block:
            result.block_until_ready()
    return (time.perf_counter() - t0) / len(args_list)


def _print_bench(label, t_np, t_jit, t_compile=None, extra_rows=()):
    w = 38
    print(f'\n[{label}]')
    if t_compile is not None:
        print(f'  {"Compilation":<{w}} {t_compile*1e3:8.1f} ms')
    print(f'  {"numpy":<{w}} {t_np*1e3:8.2f} ms / call')
    print(f'  {"jit(JAX)":<{w}} {t_jit*1e3:8.2f} ms / call')
    print(f'  {"Speedup (numpy / jit)":<{w}} {t_np/t_jit:8.1f}x')
    for name, val in extra_rows:
        print(f'  {name:<{w}} {val}')


# ---------------------------------------------------------------------------
# Pell benchmarks
# ---------------------------------------------------------------------------

def bench_pell(emu, n_warm=30):
    """numpy vs jit(JAX) for Pell; also grad and vmap."""
    rng  = np.random.default_rng(0)
    wc0  = jnp.float64(P_BASE['wc'])
    p_np = _np_params()

    # --- numpy baseline ---
    wc_vals_np = [P_BASE['wc'] + 1e-5 * v for v in rng.standard_normal(n_warm)]
    t_np = _time_fn(
        lambda wc: emu.Pell(K, {**p_np, 'wc': np.atleast_1d(np.float64(wc))},
                            ELL, de_model=DE_MODEL),
        wc_vals_np)

    # --- JAX JIT ---
    def fn_jax(wc):
        p = _jax_params(wc=jnp.atleast_1d(wc))
        return _pell_sum(emu, p, ELL)

    fn_jit = jax.jit(fn_jax)

    t0 = time.perf_counter()
    fn_jit(wc0).block_until_ready()
    t_compile = time.perf_counter() - t0

    wc_vals_jax = jnp.array(wc0 + 1e-5 * rng.standard_normal(n_warm))
    t_jit = _time_fn(fn_jit, wc_vals_jax, block=True)

    _print_bench(f'Pell — {len(K)} k-points, ell={ELL}', t_np, t_jit, t_compile)

    # --- grad ---
    fn_grad_jit = jax.jit(jax.grad(fn_jax))
    t0 = time.perf_counter()
    fn_grad_jit(wc0).block_until_ready()
    t_grad_compile = time.perf_counter() - t0
    t_grad = _time_fn(fn_grad_jit, wc_vals_jax, block=True)
    _print_bench('Pell grad wrt wc — jit(grad)', t_np, t_grad,
                 t_compile=t_grad_compile,
                 extra_rows=[('(numpy is Pell only, no grad)', '')])

    # --- vmap ---
    n_cosmo = 32
    wcs = jnp.array([P_BASE['wc'] + 0.002 * i for i in range(n_cosmo)],
                    dtype=jnp.float64)
    fn_single = _fn_single_pell(emu)
    fn_vmap   = jax.jit(jax.vmap(fn_single))
    fn_jit1   = jax.jit(fn_single)
    fn_vmap(wcs).block_until_ready()
    fn_jit1(wcs[0]).block_until_ready()

    n_reps = 10
    t_vmap = _time_fn(fn_vmap, [wcs] * n_reps, block=True) / n_reps
    t_seq  = sum(_time_fn(fn_jit1, [wc] * 1, block=True)
                 for wc in wcs) / n_cosmo
    t_np_1 = _time_fn(
        lambda wc: emu.Pell(K, {**p_np, 'wc': np.atleast_1d(float(wc))},
                            [0], de_model=DE_MODEL),
        [float(wc) for wc in wcs[:10]])

    print(f'\n[Pell vmap — N={n_cosmo} cosmologies, {len(K)} k-points (ell=0)]')
    w = 38
    print(f'  {"numpy (per cosmo)":<{w}} {t_np_1*1e3:8.2f} ms / call')
    print(f'  {"jit sequential (per cosmo)":<{w}} {t_seq*1e3:8.2f} ms / call')
    print(f'  {"vmap (per cosmo)":<{w}} {t_vmap/n_cosmo*1e3:8.2f} ms / call')
    print(f'  {"Speedup numpy / vmap":<{w}} {t_np_1*n_cosmo/t_vmap:8.1f}x')
    print(f'  {"Speedup jit-seq / vmap":<{w}} {t_seq*n_cosmo/t_vmap:8.1f}x')


# ---------------------------------------------------------------------------
# Bell_Sugi benchmarks
# ---------------------------------------------------------------------------

def bench_bell_sugi(emu, n_warm=10):
    """numpy vs jit(JAX) for Bell_Sugi."""
    rng  = np.random.default_rng(1)
    b1_0 = jnp.float64(P_BASE['b1'])
    p_np = _np_params()

    # --- numpy baseline ---
    b1_vals_np = [P_BASE['b1'] + 0.01 * v for v in rng.standard_normal(n_warm)]
    t_np = _time_fn(
        lambda b1: emu.Bell_Sugi(
            BELL_PAIR,
            {**p_np, 'b1': np.atleast_1d(np.float64(b1))},
            ell=BELL_ELL, de_model=DE_MODEL),
        b1_vals_np)

    # --- JAX JIT ---
    def fn_jax(b1):
        p   = _jax_params(b1=jnp.atleast_1d(b1))
        res = emu.Bell_Sugi(BELL_PAIR, p, ell=BELL_ELL, de_model=DE_MODEL)
        return jnp.stack([jnp.asarray(res[ll]) for ll in BELL_ELL])

    fn_jit = jax.jit(fn_jax)

    t0 = time.perf_counter()
    fn_jit(b1_0).block_until_ready()
    t_compile = time.perf_counter() - t0

    b1_vals_jax = jnp.array(b1_0 + 0.01 * rng.standard_normal(n_warm))
    t_jit = _time_fn(fn_jit, b1_vals_jax, block=True)

    _print_bench(f'Bell_Sugi — {BELL_PAIR.shape[0]} pairs, ell={list(BELL_ELL)}',
                 t_np, t_jit, t_compile)

    # --- grad wrt b1 ---
    def fn_scalar(b1):
        p   = _jax_params(b1=jnp.atleast_1d(b1))
        res = emu.Bell_Sugi(BELL_PAIR, p, ell=BELL_ELL, de_model=DE_MODEL)
        return sum(jnp.sum(jnp.asarray(res[ll])) for ll in BELL_ELL)

    fn_grad_jit = jax.jit(jax.grad(fn_scalar))
    t0 = time.perf_counter()
    fn_grad_jit(b1_0).block_until_ready()
    t_grad_compile = time.perf_counter() - t0
    t_grad = _time_fn(fn_grad_jit, b1_vals_jax, block=True)
    _print_bench('Bell_Sugi grad wrt b1 — jit(grad)', t_np, t_grad,
                 t_compile=t_grad_compile,
                 extra_rows=[('(numpy is Bell_Sugi only, no grad)', '')])


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    e = PTEmu(model=MODEL, use_Mpc=True)
    p_fid = {k: np.atleast_1d(np.float64(v)) for k, v in P_FID.items()}
    e.define_fiducial_cosmology(params_fid=p_fid, de_model=DE_MODEL)

    pad = 38

    print('=' * 60)
    print('Correctness — Pell')
    print('=' * 60)
    test_jax_matches_numpy(e)
    print(f'  {"JAX vs numpy (all ell)":<{pad}} PASS')
    test_jit_correctness(e)
    print(f'  {"JIT correctness":<{pad}} PASS')

    print()
    print('=' * 60)
    print('Gradients — Pell')
    print('=' * 60)
    for key in COSMO_KEYS + BIAS_KEYS:
        val = jnp.float64(P_BASE[key])
        g   = jax.jit(jax.grad(_fn_for_key(e, key)))(val)
        print(f'  {"d/d("+key+")":<{pad}} {float(g):.4g}')
    test_grad_jit_matches_eager_grad(e)
    print(f'  {"jit(grad) == eager grad":<{pad}} PASS')
    test_grad_all_ell(e)
    print(f'  {"d/d(wc) over ell=[0,2,4]":<{pad}} PASS')

    print()
    print('=' * 60)
    print('Correctness — Bell_Sugi')
    print('=' * 60)
    test_bell_sugi_jax_matches_numpy(e)
    print(f'  {"Bell_Sugi JAX vs numpy":<{pad}} PASS')
    test_bell_sugi_grad_b1(e)
    print(f'  {"Bell_Sugi d/d(b1)":<{pad}} PASS')
    test_bell_sugi_grad_wc(e)
    print(f'  {"Bell_Sugi d/d(wc)":<{pad}} PASS')

    print()
    print('=' * 60)
    print('vmap — Pell')
    print('=' * 60)
    test_vmap_correctness(e)
    print(f'  {"vmap vs sequential":<{pad}} PASS')

    print()
    print('=' * 60)
    print('Benchmarks')
    print('=' * 60)
    bench_pell(e)
    bench_bell_sugi(e)
