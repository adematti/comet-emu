"""Spline module."""

import numpy as np
from scipy.interpolate import PPoly, make_interp_spline


def _build_bucket_table(G, nbucket=2048):
    r"""Precompute an O(1) segment-lookup table for the breakpoints ``G``.

    Evaluating a piecewise cubic at ``k`` first needs the segment index ``i``
    with ``G[i] <= k < G[i+1]``. Searching for it per point is what makes the
    JAX spline path slow: ``jnp.searchsorted`` (used internally by
    ``interpax``) is emitted by XLA as an unvectorized loop, benchmarking
    ~23x slower than a gather. Since the knot grid is fixed between calls,
    the search can be replaced by a direct address computation done once.

    Overlay ``nbucket`` buckets uniform in ``log k``. The bucket holding ``k``
    is pure arithmetic, ``b = floor((log k - log G[0]) * inv_dlog)``, and
    ``seg[b]`` gives the segment at that bucket's left edge. Sizing every
    bucket narrower than the smallest segment means a bucket straddles at
    most one breakpoint, so one comparison finishes the job::

        i = seg[b];   i += (k >= G[i+1])

    Log spacing suits a quasi-logarithmic k-table: linear spacing there
    varies by ~380x across the range but log spacing by only ~16x, so log
    buckets need ~1k entries where linear would need ~6.5k.

    ``PPoly.from_spline`` repeats the boundary knots (a cubic B-spline emits
    k+1 copies at each end), so ``G`` is not strictly increasing and carries
    zero-width segments. Bucketing therefore runs over the distinct values
    ``V``, each mapped back to the largest original index holding it --
    exactly what ``searchsorted(..., side='right') - 1`` selects.

    Returns ``(seg, last_index, V, log_lo, inv_dlog)``.
    """
    V = np.unique(G)
    last_index = (np.searchsorted(G, V, side='right') - 1).astype(np.int32)
    logV = np.log(V)
    lo, hi = logV[0], logV[-1]
    need = int(np.ceil((hi - lo)/np.diff(logV).min()))
    if nbucket < need:
        nbucket = 1 << int(np.ceil(np.log2(need)))
    inv_dlog = nbucket/(hi - lo)
    seg = np.clip(np.searchsorted(
        V, np.exp(lo + np.arange(nbucket)/inv_dlog), side='right') - 1,
        0, V.size - 1).astype(np.int32)
    # Verify against searchsorted on the knots, the segment midpoints and both
    # float neighbours of every knot -- where an off-by-one would hide.
    probe = np.clip(np.concatenate(
        [V, np.sqrt(V[:-1]*V[1:]), np.nextafter(V, hi), np.nextafter(V, 0.0)]),
        V[0], V[-1])
    b = np.clip(((np.log(probe) - lo)*inv_dlog).astype(int), 0, nbucket - 1)
    j = seg[b]
    j = np.minimum(j + (probe >= V[np.minimum(j + 1, V.size - 1)]), V.size - 1)
    if not np.array_equal(
            np.clip(last_index[j], 0, G.size - 2),
            np.clip(np.searchsorted(G, probe, side='right') - 1, 0, G.size - 2)):
        raise RuntimeError("bucket lookup disagrees with searchsorted")
    return seg, last_index, V, float(lo), float(inv_dlog)



def _is_jax_value(x):
    """Return True if x is a JAX array or tracer (not a plain numpy array)."""
    return not isinstance(x, np.ndarray)


# Static part of the power-basis export of a not-a-knot cubic spline, keyed on
# the knots (see _cubic_ppoly_basis)
_PPOLY_BASIS_CACHE = {}


def _cubic_ppoly_basis(x):
    r"""Static part of the power-basis export of the interpolating spline
    through ``(x, y)`` (``make_interp_spline``, i.e. cubic not-a-knot, as in
    :meth:`Splines.build`), for any ``y``.

    :meth:`Splines.as_ppoly` converts the built B-spline with
    ``PPoly.from_spline``; for fixed knots the coefficients are linear in
    ``y``, ``coeffs = M @ y``. Returns a dict with the breakpoints ``G``, the
    map ``M`` (4, nseg, nx), and the bucket table of `_build_bucket_table`
    (``seg``, ``last_index``, ``V``, ``log_lo``, ``inv_dlog``), which only
    depend on ``x``. Cached on ``x``. Used by `ppoly_table_jax`.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    key = (x.size, x.tobytes())
    basis = _PPOLY_BASIS_CACHE.get(key)
    if basis is None:
        pps = [PPoly.from_spline(make_interp_spline(x, e))
               for e in np.eye(x.size)]
        G = np.ascontiguousarray(pps[0].x)
        M = np.ascontiguousarray(np.stack([pp.c for pp in pps], axis=-1))
        seg, last_index, V, log_lo, inv_dlog = _build_bucket_table(G)
        basis = dict(G=G, M=M, seg=seg, last_index=last_index, V=V,
                     log_lo=log_lo, inv_dlog=inv_dlog)
        _PPOLY_BASIS_CACHE[key] = basis
    return basis


def ppoly_table_jax(x, y, h=None, id_min=0, id_max=-1):
    r"""JAX-traceable counterpart of :meth:`Splines.ppoly_table`, for values
    ``y`` that may be JAX tracers.

    :meth:`Splines.ppoly_table` exports a spline fitted by scipy, which needs
    concrete values. Since for fixed knots the coefficients are a linear map
    of ``y`` (`_cubic_ppoly_basis`), the same table is obtained here as
    ``M @ y``, with the power-law tails of :meth:`Splines.build`. With
    ``h`` (:math:`h^{-1}\,\mathrm{Mpc}` units, ``use_Mpc=False``) the
    spline goes through :math:`(x/h, y h^3)`, i.e. it is the spline on
    ``x`` with :math:`x \to hx`: the knots are scaled by ``scale = 1/h``
    and the coefficient of :math:`dx^{3-p}` by :math:`h^{6-p}`.

    Parameters
    ----------
    x: numpy.ndarray, shape (nx,)
        Knots (static).
    y: array, shape (nx, ncoef, nparams)
        Values of ``ncoef`` functions per parameter set (e.g. the
        coefficients of a polynomial in :math:`\mu^2`, see
        `PTEmu._jax_Pdw_ppoly_table`). The tails are those of ``y[:, 0]``.
    h: array or float, optional
        Per-parameter-set :math:`h`, or **None** for :math:`\mathrm{Mpc}`
        units.
    id_min, id_max: int
        Indices of the edge points used for the tails, as in
        :class:`Splines`.

    Returns
    -------
    table: dict
        As :meth:`Splines.ppoly_table`, with the same reference knots ``G``
        and bucket arrays for every parameter set, ``scale`` (nparams,),
        ``coeffs`` (4, nseg, ncoef, nparams) and ``tails`` (4, nparams), the
        last three as JAX arrays.
    """
    import jax.numpy as jnp
    basis = _cubic_ppoly_basis(x)
    npar = y.shape[-1]
    coeffs = jnp.einsum('psk,knm->psnm', jnp.asarray(basis['M']), y)

    # power-law tails of y[:, 0], with safe denominators so that no 0/0
    # enters the gradient
    y0 = y[:, 0, :]
    lo, lo2, hi, hi2 = id_min, id_min + 2, id_max, id_max - 2
    nz_min = y0[lo] != 0
    ratio_min = jnp.abs(y0[lo2] / jnp.where(nz_min, y0[lo], 1.0))
    dly_min = jnp.where(nz_min, jnp.log10(jnp.where(nz_min, ratio_min, 1.0)),
                        0.0)
    neff_min = dly_min / np.log10(x[lo2] / x[lo])
    nz_max = y0[hi2] != 0
    ratio_max = jnp.where(nz_max,
                          jnp.abs(y0[hi] / jnp.where(nz_max, y0[hi2], 1.0)),
                          1.0)
    neff_max = jnp.log10(ratio_max) / np.log10(x[hi] / x[hi2])
    neff_max = jnp.where(jnp.isnan(neff_max), 0.0, neff_max)
    y_min = y0[lo]
    y_max = y0[hi]

    if h is None:
        scale = jnp.ones(npar)
    else:
        h = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(h)), (npar,))
        scale = 1.0 / h
        hpow = h[None, :]**(6 - np.arange(4))[:, None]        # (4, npar)
        coeffs = coeffs * hpow[:, None, None, :]
        y_min = y_min * h**3
        y_max = y_max * h**3
    tails = jnp.stack([y_min, neff_min, y_max, neff_max], axis=0)
    return dict(G=basis['G'], scale=scale, seg=basis['seg'],
                last_index=basis['last_index'], V=basis['V'],
                log_lo=basis['log_lo'], inv_dlog=basis['inv_dlog'],
                coeffs=coeffs, tails=tails)


class Splines:
    r"""Class for handling splined objects within Comet.
    """

    def __init__(self, use_Mpc, ncol=0, id_min=0, crossover_check=False):
        self.use_Mpc = use_Mpc
        self.id_min = id_min
        self.id_max = -1
        self.ncol = tuple([ncol]) if not isinstance(ncol, tuple) else ncol
        self.crossover_check = crossover_check
        self.x_raw = None
        self.y_raw = None
        self._ppoly_table = None

    def build(self, x, y, h=None, axis=0):
        self._ppoly_table = None    # invalidate the cached power-basis export
        self.x_raw = np.asarray(x, dtype=np.float64)  # force native endian (FITS uses big-endian)
        self.y_raw = y
        self.size_last = y.shape[-1]
        self.ids = np.arange(self.size_last, dtype=np.int32)

        if _is_jax_value(y):
            self.x_min = np.atleast_1d(self.x_raw[self.id_min])
            self.x_max = np.atleast_1d(self.x_raw[self.id_max])
            if h is not None:
                import jax.numpy as jnp
                self.h = jnp.atleast_1d(h) if _is_jax_value(h) else np.atleast_1d(h)
                self.h3 = self.h**3
            return

        if self.use_Mpc:
            self.spline = [make_interp_spline(x, y[...,n], axis=axis) \
                           for n in range(self.size_last)]
        else:
            self.h = np.atleast_1d(h)
            self.h3 = self.h**3
            xh = np.divide.outer(x, self.h)
            yh3 = y*self.h3
            self.spline = [make_interp_spline(xh[...,n], yh3[...,n], axis=axis)\
                           for n in range(self.size_last)]

        # low-k extrapolation
        self.y_min = y[self.id_min]
        self.x_min = np.atleast_1d(x[self.id_min])
        ids_nonzero = self.y_min != 0
        dly_min = np.empty_like(self.y_min)
        dly_min[ids_nonzero] = np.log10(
            np.abs(y[self.id_min+2][ids_nonzero]/self.y_min[ids_nonzero]))
        dly_min[~ids_nonzero] = 0.0
        dlx_min = np.log10(np.abs(x[self.id_min+2]/x[self.id_min]))
        self.neff_min = dly_min/dlx_min
        if self.use_Mpc:
            if len(self.ncol) > 1:
                self.x_min = self.x_min[:,np.newaxis]
            self.extrapolation_min = lambda x,n: self.y_min[...,n] \
                * np.divide.outer(x,self.x_min)**self.neff_min[...,n]
            self.extrapolation_min_derivative = lambda x,n: self.y_min[...,n] \
                * np.divide.outer(x,self.x_min)**(self.neff_min[...,n]-1) \
                * self.neff_min[...,n] / self.x_min
        else:
            self.y_min = self.y_min*self.h3
            self.x_min = self.x_min/self.h
            self.extrapolation_min = lambda x,n: self.y_min[...,n] \
                * np.power.outer(
                    np.divide.outer(x,self.x_min[...,n]),self.neff_min[...,n])
            self.extrapolation_min_derivative = lambda x,n: self.y_min[...,n] \
                * np.power.outer(
                    np.divide.outer(x,self.x_min[...,n]),
                    self.neff_min[...,n]-1) \
                * self.neff_min[...,n] / self.x_min[...,n]

        # high-k extrapolation
        self.y_max = y[self.id_max]
        self.x_max = np.atleast_1d(x[self.id_max])
        ids_nonzero = y[self.id_max-2] != 0
        dy_max = np.empty_like(self.y_max)
        dy_max[ids_nonzero] = np.abs(
            self.y_max[ids_nonzero]/y[self.id_max-2][ids_nonzero])
        dy_max[~ids_nonzero] = 1.0
        dly_max = np.log10(dy_max)
        dlx_max = np.log10(np.abs(x[self.id_max]/x[self.id_max-2]))
        self.neff_max = dly_max/dlx_max
        self.neff_max[np.isnan(self.neff_max)] = 0.0
        if self.use_Mpc:
            if len(self.ncol) > 1:
                self.x_max = self.x_max[:,np.newaxis]
            self.extrapolation_max_plaw = lambda x,n: self.y_max[...,n] \
                * np.divide.outer(x,self.x_max)**self.neff_max[...,n]
            self.extrapolation_max_plaw_derivative = lambda x,n: \
                self.y_max[...,n] \
                * np.divide.outer(x,self.x_max)**(self.neff_max[...,n]-1) \
                * self.neff_max[...,n] / self.x_max
        else:
            self.y_max = self.y_max*self.h3
            self.x_max = self.x_max/self.h
            self.extrapolation_max_plaw = lambda x,n: self.y_max[...,n] \
                * np.power.outer(
                    np.divide.outer(x,self.x_max[...,n]), self.neff_max[...,n])
            self.extrapolation_max_plaw_derivative = lambda x,n: \
                self.y_max[...,n] * np.power.outer(
                    np.divide.outer(x,self.x_max[...,n]),
                    self.neff_max[...,n]-1) \
                * self.neff_max[...,n] / self.x_max[...,n]

        if self.crossover_check:
            self.mask = np.where((dy_max > 2) | (dy_max < 0.5))
            self.slope = (y[self.id_max] - y[self.id_max-2]) \
                         / (x[self.id_max] - x[self.id_max-2])
            self.intrcpt = y[self.id_max-2] - self.slope * x[self.id_max-2]
            if not self.use_Mpc:
                self.slope *= self.h3*self.h
                self.intrcpt *= self.h3
            self.extrapolation_max_lin = lambda x,n: np.multiply.outer(x,
                self.slope[...,n]) + self.intrcpt[...,n]
            self.extrapolation_max_lin_derivative = lambda x,n: \
                np.multiply.outer(np.ones_like(x),self.slope[...,n])

    def extrapolation_max(self, x, n):
        y = self.extrapolation_max_plaw(x,n) # nx x nell
        if self.crossover_check and sum(self.mask[-1] == n) > 0:
            ids = (Ellipsis,) + tuple(self.mask[i][self.mask[-1] == n] \
                                      for i in range(len(self.mask)-1))
            y[ids] = self.extrapolation_max_lin(x,n)[ids]
        return y

    def extrapolation_max_derivative(self, x, n):
        y = self.extrapolation_max_plaw_derivative(x,n) # nx x nell
        if self.crossover_check and sum(self.mask[-1] == n) > 0:
            ids = (Ellipsis,) + tuple(self.mask[i][self.mask[-1] == n] \
                                      for i in range(len(self.mask)-1))
            y[ids] = self.extrapolation_max_lin_derivative(x,n)[ids]
        return y

    def _eval_extrapolation_min(self, x):
        y = np.stack([np.squeeze(self.extrapolation_min(x,n)) \
                      for n in range(self.size_last)],
                     axis=-1)
        return np.atleast_2d(y)

    def _eval_extrapolation_min_derivative(self, x):
        y = np.stack([np.squeeze(self.extrapolation_min_derivative(x,n)) \
                      for n in range(self.size_last)],
                     axis=-1)
        return np.atleast_2d(y)

    def as_ppoly(self):
        r"""Export the interpolating spline as power-basis piecewise-cubic
        coefficients plus its power-law extrapolation tails, for consumption
        by external kernels (e.g. the fused numba/jax bispectrum kernels)
        that cannot call scipy at runtime.

        The interior is an *exact* basis change of the already-built B-spline
        (via ``scipy.interpolate.PPoly.from_spline``) — no re-interpolation,
        reproducing :meth:`eval` to machine precision. The returned ``tails``
        reproduce the power-law extrapolation :meth:`eval` applies outside
        ``[breakpoints[0], breakpoints[-1]]`` (which coincide with the edge
        knots ``x_min``/``x_max``):
        :math:`P(k) = y_\mathrm{edge}\,(k / k_\mathrm{edge})^{n_\mathrm{eff}}`.
        The optional high-k *linear* crossover (``crossover_check=True``) is
        **not** represented; ``Pdw_spline`` does not use it.

        Intended for scalar-valued splines (``ncol == 0``), such as
        ``Pdw_spline``.

        Returns
        -------
        breakpoints : numpy.ndarray, shape ``(nbp, size_last)``
            Per-z knot vectors. These are identical across z when the spline
            domain is shared (``use_Mpc=True``, or scalar ``h``), and differ
            (each ``k_table / h_n``) for a per-z ``h`` array.
        coeffs : numpy.ndarray, shape ``(4, nbp - 1, size_last)``
            Power-basis coefficients about each breakpoint:
            :math:`p_i(k) = c_{0i}\,dk^3 + c_{1i}\,dk^2 + c_{2i}\,dk + c_{3i}`
            with :math:`dk = k - \mathrm{breakpoints}_i`.
        tails : numpy.ndarray, shape ``(4, size_last)``
            Per-z power-law tail parameters
            ``[y_min, neff_min, y_max, neff_max]``. Below ``breakpoints[0]``
            use ``y_min * (k / breakpoints[0])**neff_min``; above
            ``breakpoints[-1]`` use ``y_max * (k / breakpoints[-1])**neff_max``.
        """
        n = self.size_last
        pps = [PPoly.from_spline(self.spline[i]) for i in range(n)]
        breakpoints = np.stack([pp.x for pp in pps], axis=-1)
        coeffs = np.stack([pp.c for pp in pps], axis=-1)
        # Tail parameters mirror extrapolation_min / extrapolation_max_plaw.
        # The edge amplitudes/slopes are already h-rescaled by build() when
        # use_Mpc is False, matching the units the breakpoints live in.
        to_row = lambda a: np.broadcast_to(
            np.atleast_1d(np.asarray(a, dtype=float)).ravel(), (n,))
        tails = np.ascontiguousarray(np.stack(
            [to_row(self.y_min), to_row(self.neff_min),
             to_row(self.y_max), to_row(self.neff_max)], axis=0))
        return breakpoints, coeffs, tails

    def ppoly_table(self):
        r"""Power-basis export of the spline (see :meth:`as_ppoly`) together
        with an O(1) segment-lookup table (see `_build_bucket_table`), for
        compiled kernels that evaluate the spline inline.

        The per-z knot vectors are either identical or each ``x / h_n``, i.e.
        proportional to a single reference grid ``G``, so one lookup table
        serves every z: segment of ``k`` for z ``m`` is that of
        ``k / scale[m]`` in ``G``. Cached until the next :meth:`build`.

        Returns
        -------
        table: dict
            ``G`` (nbp,), ``scale`` (nparams,), the bucket arrays ``seg``,
            ``last_index`` and ``V``, the floats ``log_lo`` and ``inv_dlog``,
            ``coeffs`` (4, nbp - 1, nparams) and ``tails`` (4, nparams) as in
            :meth:`as_ppoly`.
        """
        if self._ppoly_table is None:
            breakpoints, coeffs, tails = self.as_ppoly()
            G = np.ascontiguousarray(breakpoints[:, 0])
            scale = np.ascontiguousarray(breakpoints[0, :]/breakpoints[0, 0])
            if not np.allclose(breakpoints, G[:, None]*scale[None, :],
                               rtol=1e-12):
                raise ValueError("Knot vectors are not proportional across "
                                 "the parameter batch.")
            seg, last_index, V, log_lo, inv_dlog = _build_bucket_table(G)
            self._ppoly_table = dict(
                G=G, scale=scale, seg=seg, last_index=last_index, V=V,
                log_lo=log_lo, inv_dlog=inv_dlog,
                coeffs=np.ascontiguousarray(coeffs),
                tails=np.ascontiguousarray(tails))
        return self._ppoly_table

    def _eval_spline(self, x):
        y = np.stack([np.squeeze(self.spline[n](x)) \
                      for n in range(self.size_last)],
                     axis=-1)
        return np.atleast_2d(y)

    def _eval_spline_derivative(self, x):
        y = np.stack([np.squeeze(self.spline[n].derivative(1)(x)) \
                      for n in range(self.size_last)],
                     axis=-1)
        return np.atleast_2d(y)

    def _eval_extrapolation_max(self, x):
        y = np.stack([np.squeeze(self.extrapolation_max(x,n)) \
                      for n in range(self.size_last)],
                     axis=-1)
        return np.atleast_2d(y)

    def _eval_extrapolation_max_derivative(self, x):
        y = np.stack([np.squeeze(self.extrapolation_max_derivative(x,n)) \
                      for n in range(self.size_last)],
                     axis=-1)
        return np.atleast_2d(y)

    def eval(self, x):
        mask_less = x < self.x_min if self.use_Mpc \
                    else x[:,None] < self.x_min # -> nx x N
        mask_greater = x > self.x_max if self.use_Mpc \
                       else x[:,None] > self.x_max
        mask = ~mask_less & ~mask_greater

        y = np.empty(x.shape + (*self.ncol,self.size_last,)) \
            if sum(self.ncol) > 0 else np.empty(x.shape+(self.size_last,))

        y[mask_less] = self._eval_extrapolation_min(x)[mask_less]
        y[mask] = self._eval_spline(x)[mask]
        y[mask_greater] = self._eval_extrapolation_max(x)[mask_greater]
        return y

    def eval_varx(self, x): # last dimension of x matches self.size_last
        if _is_jax_value(x):
            return self._eval_varx_jax(x)
        n = len(x.shape) - 1
        mask_less = x < self.x_min # -> nx x N
        mask_greater = x > self.x_max
        mask = ~mask_less & ~mask_greater

        y = np.empty(x.shape[:n] + (*self.ncol,self.size_last,)) \
            if sum(self.ncol) > 0 else np.empty(x.shape[:n]+(self.size_last,))

        for i in range(self.size_last):
            y[mask_less[...,i],...,i] = \
                self.extrapolation_min(x[mask_less[...,i],i],i)
            y[mask[...,i],...,i] = self.spline[i](x[mask[...,i],i])
            y[mask_greater[...,i],...,i] = \
                self.extrapolation_max(x[mask_greater[...,i],i],i)
        return y

    def _eval_varx_jax(self, x):
        """JAX-compatible eval_varx with power-law extrapolation outside the grid.

        Mirrors the numpy eval_varx behaviour: power-law (or linear, when
        crossover_check triggers) outside [x_min, x_max], cubic interp inside.
        """
        import jax.numpy as jnp
        import interpax
        n_col_flat = int(np.prod(self.ncol)) if sum(self.ncol) > 0 else 1
        out_cols = []
        for col_idx in range(self.size_last):
            xi = x[..., col_idx]
            if self.use_Mpc:
                xg = jnp.asarray(self.x_raw)
            else:
                xg = jnp.asarray(self.x_raw) / self.h[col_idx]

            if sum(self.ncol) > 0:
                yi_col = self.y_raw[..., col_idx]
                if not self.use_Mpc:
                    yi_col = yi_col * self.h3[col_idx]
            else:
                yi_col = self.y_raw[..., col_idx]
                if not self.use_Mpc:
                    yi_col = yi_col * self.h3[col_idx]

            x_lo  = xg[self.id_min]
            x_hi  = xg[-1]
            x_lo2 = xg[self.id_min + 2]
            x_hi3 = xg[-3]
            log_dx_lo = jnp.log10(x_lo2 / x_lo)
            log_dx_hi = jnp.log10(x_hi / x_hi3)

            ncol_parts = []
            for j in range(n_col_flat):
                if sum(self.ncol) > 0:
                    j_idx = np.unravel_index(j, self.ncol)
                    y_1d = jnp.asarray(yi_col[(slice(None),) + j_idx])
                else:
                    y_1d = jnp.asarray(yi_col)

                y_lo  = y_1d[self.id_min]
                y_lo2 = y_1d[self.id_min + 2]
                y_hi  = y_1d[-1]
                y_hi3 = y_1d[-3]

                # Low-k power-law: y = y_lo * (xi/x_lo)^neff_lo
                # Safe denominator: replace 0 with 1 before dividing so JAX reverse-mode
                # never differentiates through 1/0 (which gives -inf, and 0*-inf = NaN).
                y_lo_safe = jnp.where(y_lo != 0, y_lo, jnp.ones_like(y_lo))
                ratio_lo = jnp.where(y_lo != 0, jnp.abs(y_lo2 / y_lo_safe), 1.0)
                neff_lo = jnp.log10(jnp.maximum(ratio_lo, 1e-30)) / log_dx_lo
                y_extrap_lo = y_lo * jnp.power(xi / x_lo, neff_lo)

                # High-k power-law: y = y_hi * (xi/x_hi)^neff_hi
                y_hi3_safe = jnp.where(y_hi3 != 0, y_hi3, jnp.ones_like(y_hi3))
                ratio_hi = jnp.where(y_hi3 != 0, jnp.abs(y_hi / y_hi3_safe), 1.0)
                neff_hi = jnp.log10(jnp.maximum(ratio_hi, 1e-30)) / log_dx_hi
                y_extrap_hi = y_hi * jnp.power(xi / x_hi, neff_hi)

                if self.crossover_check:
                    # Linear fallback when ratio is extreme (same threshold as numpy path)
                    slope = (y_hi - y_hi3) / (x_hi - x_hi3)
                    intrcpt = y_hi3 - slope * x_hi3
                    y_extrap_hi = jnp.where(
                        (ratio_hi > 2.0) | (ratio_hi < 0.5),
                        slope * xi + intrcpt,
                        y_extrap_hi)

                # Cubic, to match the numpy path's scipy spline: jnp.interp is piecewise
                # linear, which on the emulator's ~6%-spaced k table leaves a b1-independent,
                # sign-oscillating ~0.2% error in P_ell peaking between knots.
                y_interp = interpax.interp1d(xi.ravel(), xg, y_1d, method='cubic2',
                                             extrap=True).reshape(xi.shape)
                y_out = jnp.where(xi < x_lo, y_extrap_lo,
                        jnp.where(xi > x_hi, y_extrap_hi, y_interp))
                ncol_parts.append(y_out)

            if sum(self.ncol) > 0:
                out_col = jnp.stack(ncol_parts, axis=-1).reshape(xi.shape + self.ncol)
            else:
                out_col = ncol_parts[0]
            out_cols.append(out_col)
        return jnp.stack(out_cols, axis=-1)

    # this is just a quick fix for compatibility with the bispectrum module
    # derivative should also be applied to the extrapolations
    def derivative(self, n):
        return self.spline[0].derivative(n)

    def eval_derivative(self, x):
        mask_less = x < self.x_min if self.use_Mpc \
                    else x[:,None] < self.x_min # -> nx x N
        mask_greater = x > self.x_max if self.use_Mpc \
                       else x[:,None] > self.x_max
        mask = ~mask_less & ~mask_greater

        y = np.empty(x.shape + (*self.ncol,self.size_last,)) \
            if sum(self.ncol) > 0 else np.empty(x.shape+(self.size_last,))

        y[mask_less] = self._eval_extrapolation_min_derivative(x)[mask_less]
        y[mask] = self._eval_spline_derivative(x)[mask]
        y[mask_greater] = self._eval_extrapolation_max_derivative(x) \
                          [mask_greater]
        return y
