"""Cosmology module."""

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import beta, betainc


class Cosmology:
    r"""Class containing cosmological parameters and background quantities.

    This object is used to compute background quantities (expansion, growth)
    of the specific cosmology passed as input.

    In all cases including an evolving dark energy density, its equation of
    state is parametrized as :math:`w(a)=w_0+w_a(1-a)`
    (`Chevalier & Polansky 2001
    <https://www.worldscientific.com/doi/abs/10.1142/S0218271801000822>`_,
    `Linder 2003
    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.90.091301>`_).
    """

    def __init__(self, Om0, H0, Ok0=0.0, Or0=0.0, de_model='lambda',
                 w0=-1.0, wa=0.0):
        r"""Class constructor.

        Parameters
        ----------
        Om0: float
            Fractional total matter density at present time,
            :math:`\Omega_\mathrm{m,0}`.
        H0: float
            Hubble constant at present time, :math:`H_0`.
        Ok0: float, optional
            Fractional curvature density at present time,
            :math:`\Omega_\mathrm{k,0}`. Defaults to 0.0.
        Or0: float, optional
            Fractional radiation density at present time,
            :math:`\Omega_\mathrm{r,0}`, Defaults to 0.0.
        de_model: str, optional
            Selected dark energy model. It can be chosen from the list
            [`"lambda"`, `"w0"`, `"w0wa"`], whose entries correspond to a LCDM,
            :math:`w` CDM, and quintessence model, respectively.
            Defaults to `"lambda"`.
        w0: float, optional
            Dark energy equation of state at present time, :math:`w_0`.
            Only used if **de_model** is not `"lambda"`. Defaults to -1.0.
        wa: float, optional
            Negative derivative of the dark energy equation of state with
            respect to the scale factor, :math:`w_a`. Only used if
            **de_model** is `"w0wa"`. Defaults to 0.0.
        """
        self.Om0 = Om0
        self.Ok0 = Ok0
        self.Or0 = Or0
        self.H0 = H0

        self.Ode0 = 1.0 - self.Om0 - self.Ok0 - self.Or0
        self.hubble_distance = 2.998E5/self.H0

        self.de_model = de_model
        self.w0 = w0
        self.wa = wa

        self.flat = True if self.Ok0 == 0.0 else False
        self.relspecies = False if self.Or0 == 0.0 else True

    def update_cosmology(self, Om0, H0, Ok0=0.0, Or0=0.0, de_model='lambda',
                         w0=-1.0, wa=0.0):
        r"""
        Update the class to a new cosmology.

        Overwrites the current class attributes to assign the new cosmological
        parameters passed as input.

        Parameters
        ----------
        Om0: float
            Fractional total matter density at present time,
            :math:`\Omega_\mathrm{m,0}`.
        H0: float
            Hubble constant at present time, :math:`H_0`.
        Ok0: float, optional
            Fractional curvature density at present time,
            :math:`\Omega_\mathrm{k,0}`. Defaults to 0.0.
        Or0: float, optional
            Fractional radiation density at present time,
            :math:`\Omega_\mathrm{r,0}`, Defaults to 0.0.
        de_model: str, optional
            Selected dark energy model. It can be chosen from the list
            [`"lambda"`, `"w0"`, `"w0wa"`], whose entries correspond to a LCDM,
            :math:`w` CDM, and quintessence model, respectively.
            Defaults to `"lambda"`.
        w0: float, optional
            Dark energy equation of state at present time, :math:`w_0`.
            Only used if **de_model** is not `"lambda"`. Defaults to -1.0.
        wa: float, optional
            Negative derivative of the dark energy equation of state with
            respect to the scale factor, :math:`w_a`. Only used if
            **de_model** is `"w0wa"`. Defaults to 0.0.
        """
        self.Om0 = Om0
        self.Ok0 = Ok0
        self.Or0 = Or0
        self.H0 = H0

        self.Ode0 = 1.0 - self.Om0 - self.Ok0 - self.Or0
        self.hubble_distance = 2.998E5/self.H0

        self.de_model = de_model
        self.w0 = w0
        self.wa = wa

        self.flat = True if self.Ok0 == 0.0 else False
        self.relspecies = False if self.Or0 == 0.0 else True

    def DE_z(self, z):
        r"""
        Dark energy density as a function of redshift.

        Returns the energy density of the specified dark energy component,
        normalised to the value at present time,

        .. math::
             \frac{\rho_\mathrm{DE}(z)}{\rho_\mathrm{DE}(0)} = \
             \exp\left[-\int_0^z \frac{dz'}{1+z'}\left(1+w\left(z'\right)\
             \right) \right]\,.

        Parameters
        ----------
        z: float or numpy.ndarray
            Redshifts at which to evaluate the dark energy density.

        Returns
        -------
        de_z: float or numpy.ndarray
            Energy density of the dark energy component at the specified
            redshifts.
        """
        if self.de_model == 'lambda':
            de_z = 1.0
        elif self.de_model == 'w0':
            de_z = (1.0+z)**(3.0*(1.0+self.w0))
        elif self.de_model == 'w0wa':
            a = 1.0/(1.0+z)
            de_z = a**(-3.0*(1.0+self.w0+self.wa))*np.exp(-3.0*self.wa*(1.0-a))
        return de_z

    def wz(self, z):
        r"""
        Dark energy equation of state as a function of redshift.

        Returns the equation of state of the dark energy component,

        .. math::
             w(z) = w_0+w_a\frac{z}{1+z}\,.

        Parameters
        ----------
        z: float or numpy.ndarray
            Redshifts at which to evaluate the dark energy equation of state.

        Returns
        -------
        w: float or numpy.ndarray
            Dark energy equation of state at the specified redshifts.
        """
        if self.de_model == 'lambda':
            w = -1.0*np.ones_like(z)
        elif self.de_model == 'w0':
            w = self.w0*np.ones_like(z)
        elif self.de_model == 'w0wa':
            w = self.w0 + self.wa*z/(1.0+z)
        return w

    def Ez(self, z):
        r"""
        Normalised expansion factor as a function of redshift.

        Returns the value of the Hubble factor at the specified redshift,
        normalised to the value at present time,

        .. math::
            :nowrap:

                \begin{split}
                    E(z) = \frac{H(z)}{H_0} = \bigg[ \
                    &\Omega_\mathrm{m,0}(1+z)^3 \
                    + \Omega_\mathrm{r,0}(1+z)^4 \
                    + \Omega_\mathrm{k,0}(1+z)^2 \\
                    &+ \Omega_\mathrm{DE,0}\exp\Big[-\int_0^z \
                    \frac{dz'}{1+z'} \left(1+w\left(z'\right)\right) \Big]
                    \bigg]^\frac{1}{2}\,.
                \end{split}

        Parameters
        ----------
        z: float or numpy.ndarray
            Redshifts at which to evaluate the normalised expansion factor.

        Returns
        -------
        Ez: float or numpy.ndarray
            Normalised expansion factor at the specified redshifts.
        """
        ainv = 1.0+z
        Ez2 = self.Om0*ainv**3 + self.Ode0*self.DE_z(z)
        if not self.flat:
            Ez2 += self.Ok0*ainv**2
        if self.relspecies:
            Ez2 += self.Or0*ainv**4
        Ez = np.sqrt(Ez2)
        return Ez

    def one_over_Ez(self, z):
        r"""
        Inverse normalised expansion factor as a function of redshift.

        Return the inverse of the normalised expansion factor defined in
        **Ez**.

        Parameters
        ----------
        z: float or numpy.ndarray
            Redshifts at which to evaluate the inverse expansion factor.

        Returns
        -------
        inv_Ez: float or numpy.ndarray
            Inverse normalised expansion factor at the specified redshifts.
        """
        inv_Ez = 1.0/self.Ez(z)
        return inv_Ez

    def Hz(self, z):
        r"""
        Hubble expansion factor.

        Returns the Hubble parameter :math:`H(z)`, defined as

        .. math::
             H(z) = H_0\, E(z) \, .

        Parameters
        ----------
        z: float or numpy.ndarray
            Redshifts at which to evaluate the Hubble factor.

        Returns
        -------
        Hz: float or numpy.ndarray
            Hubble expansion factor at the specified redshifts.
        """
        Hz = self.H0*self.Ez(z)
        return Hz

    def Om(self, z):
        r"""
        Fractional total matter density as a function of redshift.

        Returns the total matter density at the specified redshifts in units
        of the critical density
        :math:`\rho_\mathrm{crit}(z)=\frac{3H^2(z)}{8\pi G}`, as

        .. math::
             \Omega_\mathrm{m}(z) = \
             \frac{\rho_\mathrm{m}(z)}{\rho_\mathrm{crit}(z)} = \
             \frac{\Omega_\mathrm{m,0}\,(1+z)^3}{E^2(z)} \, .

        Parameters
        ----------
        z: float or numpy.ndarray
            Redshifts at which to evaluate the fractional matter density.

        Returns
        -------
        Om: float or numpy.ndarray
            Fractional matter density at the specified redshifts.
        """
        Om = self.Om0*(1.0+z)**3/(self.Ez(z))**2
        return Om

    def Ode(self, z):
        r"""
        Fractional dark energy density as a function of redshift.

        Returns the dark energy density at the specified redshifts in units
        of the critical density
        :math:`\rho_\mathrm{crit}(z)=\frac{3H^2(z)}{8\pi G}`, as

        .. math::
             \Omega_\mathrm{DE}(z) = \
             \frac{\rho_\mathrm{DE}(z)}{\rho_\mathrm{crit}(z)} = \
             \frac{\Omega_\mathrm{DE,0}\exp\left[ \
             -\int_0^z \frac{dz'}{1+z'}\left(1+w\left(z'\right)\right) \
             \right]}{E^2(z)} \, .

        Parameters
        ----------
        z: float or numpy.ndarray
            Redshifts at which to evaluate the fractional dark energy density.

        Returns
        -------
        Ode: float or numpy.ndarray
            Fractional dark energy density at the specified redshifts.
        """
        Ode = self.Ode0/(self.Ez(z))**2*self.DE_z(z)
        return Ode

    def comoving_transverse_distance(self, z):
        r"""
        Transverse comoving distance as a function of redshift.

        Returns the transverse comoving distance defined as

        .. math::
            :nowrap:

            \begin{gather*}
                D_\mathrm{M}(z) =
                \begin{cases}
                   \frac{D_H}{\sqrt{-\Omega_\mathrm{k,0}}}\sin \left(\
                   \frac{\sqrt{-\Omega_\mathrm{k,0}}D_\mathrm{C}(z)}{D_H}\
                   \right), & \text{if}\ \Omega_\mathrm{k,0}<0 \\
                   D_\mathrm{C}(z), & \text{if}\ \Omega_\mathrm{k,0}=0 \, ,\\
                   \frac{D_H}{\sqrt{\Omega_\mathrm{k,0}}}\sinh \left(\
                   \frac{\sqrt{\Omega_\mathrm{k,0}}D_\mathrm{C}(z)}{D_H}\
                   \right), & \text{if}\ \Omega_\mathrm{k,0}>0
               \end{cases}
            \end{gather*}

        where

        .. math::
            D_\mathrm{C}(z) = D_H \int_0^z \frac{dz'}{E(z')}

        is the comoving distance, and

        .. math::

            D_H = \frac{c}{H_0}

        is the Hubble distance.

        Parameters
        ----------
        z: float
            Redshift at which to evaluate the transverse comoving distance.

        Returns
        -------
        dm: float
            Transverse comoving distance at the specified redshift.
        """
        r = quad(self.one_over_Ez, 0.0, z)[0]
        if self.flat:
            dm = r
        elif self.Ok0 > 0.0:
            sqrt_Ok0 = np.sqrt(self.Ok0)
            dm = np.sinh(sqrt_Ok0*r)/sqrt_Ok0
        else:
            sqrt_Ok0 = np.sqrt(-self.Ok0)
            dm = np.sin(sqrt_Ok0*r)/sqrt_Ok0
        dm *= self.hubble_distance
        return dm

    def angular_diameter_distance(self, z):
        r"""
        Angular diameter distance as a function of redshift.

        Returns the angular diameter distance defined as

        .. math::
            D_\mathrm{A}(z) = \frac{1}{1+z}D_\mathrm{M}(z) \, .

        Parameters
        ----------
        z: float
            Redshift at which to evaluate the angular diameter distance.

        Returns
        -------
        da: float
            Angular diameter distance at the specified redshift.
        """
        da = self.comoving_transverse_distance(z)/(1.0+z)
        return da

    def growth_factor(self, z, get_growth_rate=False):
        r"""
        Linear growth factor as a function of redshift.

        Returns the linear growth factor :math:`D(z)` (and the linear growth
        rate :math:`f(z)` at user request).

        Parameters
        ----------
        z: float
            Redshift at which to evaluate the growth factor.
        get_growth_rate: bool, optional
            Flag specifying whether to return only the linear growth factor
            (if **False**) or also the linear growth rate (if **True**).
            Defaults to **False**.

        Returns
        -------
        Dz: float or list
            Linear growth factor, or list containing both the linear growth
            factor and growth rate, at the specified redshift.
        """
        def Ez_for_D(z):
            ainv = 1.0+z
            Ez2 = self.Om0*ainv**3 + self.Ode0*self.DE_z(z)
            if not self.flat:
                Ez2 += self.Ok0*ainv**2
            if self.relspecies:
                Ez2 += self.Or0
            return np.sqrt(Ez2)

        def integrand(z):
            return (1.0+z)/(Ez_for_D(z))**3

        def growth_factor_from_ODE(z_eval):

            def derivatives_D(a, y):
                z = 1.0/a - 1.0
                D = y[0]
                Dp = y[1]

                wa = self.wz(z)
                Oma = self.Om(z)
                Odea = self.Ode(z)

                u1 = -(2.0 - 0.5*(Oma + (3.0*wa+1.0)*Odea))/a
                u2 = 1.5*Oma/a**2

                return [Dp, u1*Dp + u2*D]

            a_eval = np.array([1.0/(1.0 + z_eval)])
            a_min = np.fmin(a_eval, 1E-4)*0.99
            a_max = a_eval*1.01
            
            dic = solve_ivp(derivatives_D, (a_min[0], a_max[0]), [a_min[0], 1.0],
                            t_eval=a_eval, atol=1E-6, rtol=1E-6,
                            vectorized=True)
            D = dic['y'][0, :]

            if (dic['status'] != 0) or (D.shape[0] != a_eval.shape[0]):
                raise Exception('The calculation of the growth factor failed.')

            if get_growth_rate:
                Dp = dic['y'][1, :]
                f = np.float64(a_eval*Dp/D)

                return [D[0], f]
            else:
                return D[0]

        if self.de_model == 'lambda':
            if self.flat and not self.relspecies:
                a3 = 1.0/(1.0+z)**3
                Dz = (5.0/6.0*betainc(5.0/6.0, 2.0/3.0,
                                      self.Ode0*a3/(self.Om0+self.Ode0*a3)) *
                      (self.Om0 / self.Ode0)**(1.0 / 3.0) *
                      np.sqrt(1.0 + self.Om0 / (self.Ode0 * a3)) *
                      beta(5.0/6.0, 2.0/3.0))
            else:
                # integrate integral expression
                Dz = 2.5*self.Om0*Ez_for_D(z)*quad(integrand, z, np.inf)[0]
            if get_growth_rate:
                Omz = self.Om(z)
                f = -1.0 - Omz/2.0 + self.Ode(z) + 2.5*Omz/Dz/(1.0+z)
                Dz = [Dz, f]
        else:
            # do full differential equation integration
            Dz = growth_factor_from_ODE(z)

        return Dz

    def growth_rate(self, z):
        r"""
        Linear growth rate as a function of redshift.

        Returns the linear growth rate :math:`f(z)`.

        Parameters
        ----------
        z: float
            Redshift at which to evaluate the linear growth rate.

        Returns
        -------
        f: float
            Linear growth rate at the specified redshift.
        """
        def Ez_for_D(z):
            ainv = 1.0+z
            Ez2 = self.Om0*ainv**3 + self.Ode0*self.DE_z(z)
            if not self.flat:
                Ez2 += self.Ok0*ainv**2
            if self.relspecies:
                Ez2 += self.Or0
            return np.sqrt(Ez2)

        if self.de_model == 'lambda':
            Omz = self.Om(z)
            f = -1.0 - Omz/2.0 + self.Ode(z) \
                + 2.5*Omz/self.growth_factor(z)/(1.0+z)
        else:
            f = self.growth_factor(z, get_growth_rate=True)[1]
        return f

    def comoving_volume(self, zmin, zmax, fsky):
        r"""
        Comoving volume as a function of redshift and sky fraction.

        Computes the transverse comoving volume between :math:`z_\mathrm{min}`
        and :math:`z_\mathrm{max}` for a given sky fraction
        :math:`f_\mathrm{sky}`, as

        .. math::
            V_\mathrm{M}(z_\mathrm{min}, z_\mathrm{max}, f_\mathrm{sky}) = \
            \frac{4\pi f_\mathrm{sky}}{3} \left[D_\mathrm{M}^3 \
            (z_\mathrm{max}) - D_\mathrm{M}^3(z_\mathrm{min})\right]

        Parameters
        ----------
        zmin: float
            Lower redshift limit :math:`z_\mathrm{min}`.
        zmax: float
            Upper redshift limit :math:`z_\mathrm{max}`.
        fsky: float
            Sky fraction :math:`f_\mathrm{sky}`.

        Returns
        -------
        vol: float
            Transverse comoving volume between the specified minimum and
            maximum redshifts, and for the given sky fraction.
        """
        def differential_comoving_volume(z):
            dm = self.comoving_transverse_distance(z)
            return self.hubble_distance*dm**2/self.Ez(z)

        vol = fsky*4*np.pi*quad(differential_comoving_volume, zmin, zmax)[0]

        return vol
