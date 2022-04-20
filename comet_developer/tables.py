"""Tables module."""

import numpy as np
from scipy.spatial.distance import cdist
from astropy.io import fits
import GPy
from pyDOE import *


class Tables:
    r"""Class for handling the tables of the emulator.

    It handles both the training and validation sets, in the form of tables
    containing the different contributions to :math:`P_\mathrm{gg}(k,\mu)`.
    Provides routine to convert the raw tables into their transformed versions,
    which span a shorter dynamical range (to increase the accuracy of the
    emulator), to train the emulator, and to transform back the emulator
    predictions in the original metric.
    """

    def __init__(self, params, validation=False):
        r"""Class constructor.

        Parameters
        ----------
        params: list
            List of parameters used to train the emulator.
        validation: bool, optional
            A flag specifying if the class is meant for the validation of the
            emulator. Defaults to **False**.
        """
        self.params = params
        self.n_params = len(params)
        self.param_ranges = None
        self.validation = validation
        self.model = None
        self.model_transformed = None

        self.n_diagrams = 19
        self.names_diagrams = ['P0L_b1b1', 'PNL_b1', 'PNL_id', 'P1L_b1b1',
                               'P1L_b1b2', 'P1L_b1g2', 'P1L_b1g21', 'P1L_b2b2',
                               'P1L_b2g2', 'P1L_g2g2', 'P1L_b2', 'P1L_g2',
                               'P1L_g21', 'Pctr_c0', 'Pctr_c2', 'Pctr_c4',
                               'Pctr_b1b1cnlo', 'Pctr_b1cnlo', 'Pctr_cnlo']

    def set_param_ranges(self, ranges):
        r"""Set ranges for the parameters of the emulator.

        Sets the internal class attribute defining the prior for the
        parameters of the emulator.

        Parameters
        ----------
        ranges: dict
            Dictionary containing the parameter ranges. Keys correspond to the
            name of the parameters, while values are list with two entries,
            which correspond to the minimum and maximum value of the parameter.
        """
        if self.param_ranges is None:
            self.param_ranges = {}
        for p in self.params:
            self.param_ranges[p] = ranges[p]

    def generate_samples(self, ranges, n_samples, n_trials=0):
        r"""Generate parameter sample.

        The sample is built inside the specified parameter ranges and with the
        given size. Depending on the value of the class attribute
        **validation**, the sample is going to be generated using a Latin
        HyperCube (training) or randomly across the hypervolume (validation).

        Parameters
        ----------
        ranges: dict
            Dictionary containing the parameter ranges. Keys correspond to the
            name of the parameters, while values are list with two entries,
            which correspond to the minimum and maximum value of the parameter.
        n_samples: int
            Size of the sample. **n_samples** points in the hypervolume defined
            by the parameter ranges are going to be generated.
        n_trials: int, optional
            Number of resamplings of the Latin HyperCube when a training
            sample is requested. This is meant to obtain the best coverture of
            the hypervolume (with maxed minimum distance among points).
            Defaults to 0.
        """
        self.set_param_ranges(ranges)
        self.n_samples = n_samples

        if self.validation:
            self.samples = np.random.rand(self.n_samples, self.n_params)
        else:
            self.samples = lhs(self.n_params, samples=self.n_samples,
                               criterion='center')
            dist = cdist(self.samples, self.samples, metric='euclidean')
            min_dist = np.amin(dist[dist > 0.0])
            for n in range(n_trials):
                samples_new = lhs(self.n_params, samples=self.n_samples,
                                  criterion='center')
                dist = cdist(samples_new, samples_new, metric='euclidean')
                min_dist_new = np.amin(dist[dist > 0.0])
                if (min_dist_new > min_dist):
                    min_dist = min_dist_new
                    self.samples = samples_new

        for n, p in enumerate(self.params):
            self.samples[:, n] = (self.samples[:, n] *
                                  (self.param_ranges[p][1] -
                                   self.param_ranges[p][0]) +
                                  self.param_ranges[p][0])

    def save_samples(self, fname):
        r"""Save sample to file.

        Saves the parameter sample stored as class attribute into an external
        data file.

        Parameters
        ----------
        fname: str
            Name of output file.
        """
        np.savetxt(fname, self.samples)

    def load_samples(self, fname):
        r"""Load sample from file.

        Loads a parameter sample from an external data file, and stores it
        as a class attribute.

        Parameters
        ----------
        fname: str
            Name of output file to read from.
        """
        self.samples = np.loadtxt(fname)
        self.n_samples = self.samples.shape[0]

    def assign_samples(self, samples_hdu):
        r"""Read parameter sample from HDU object.

        Reads parameter sample from a Header Data Unit object, and stores it
        as class attribute. If the sample is meant for validation, determines
        the parameters from the header of the HDU object, otherwise uses the
        parameters defined as class attributes.

        Parameters
        ----------
        samples_hdu: astropy.io.fits.BinTableHDU
            Header Data Unit containing the parameter sample.
        """
        self.n_samples = samples_hdu.header['NAXIS2']
        if self.validation:
            self.samples = np.zeros([self.n_samples,
                                     samples_hdu.header['TFIELDS']])
            for i, p in enumerate([samples_hdu.header['TTYPE{}'.format(n+1)]
                                  for n in range(
                                    samples_hdu.header['TFIELDS'])]):
                self.samples[:, i] = samples_hdu.data[p]
        else:
            self.samples = np.zeros([self.n_samples, self.n_params])
            for i, p in enumerate(self.params):
                self.samples[:, i] = samples_hdu.data[p]

    def assign_table(self, table_hdu, nk, nkloop):
        r"""Read model table from HDU object.

        Reads a model table from a Header Data Unit object, and stores it as
        class attribute. If the sample is a training one, additionally calls
        the routine to convert the model table into its transformed version.

        Parameters
        ----------
        table_hdu: astropy.io.fits.BinTableHDU
            Header Data Unit containing the model table.
        nk: int
            Number of :math:`k` bins of the table.
        nkloop: int
            Number of :math:`k` bins of the table corresponding to the loop
            predictions.
        """
        self.nk = nk
        self.nkloop = nkloop

        if 'MODEL_SHAPE' == table_hdu.header['EXTNAME']:
            if self.model is None:
                self.model = {}
            for TYPE in [table_hdu.header['TTYPE{}'.format(i+1)]
                         for i in range(table_hdu.header['TFIELDS'])]:
                self.model[TYPE] = table_hdu.data[TYPE]
                if self.model[TYPE].ndim == 1:
                    self.model[TYPE] = self.model[TYPE][:, None]
            if not self.validation:
                self.transform_emulator_data(data_type=list(self.model.keys()))
        elif 'MODEL_FULL' == table_hdu.header['EXTNAME']:
            if self.model is None:
                self.model = {}
            self.model['PL'] = table_hdu.data['PL']
            for ell in [0, 2, 4]:
                for diagram in self.names_diagrams:
                    diagram_full = '{}_ell{}'.format(diagram, ell)
                    if diagram == 'PNL_b1':
                        # combine tree-level, one-loop
                        # and IR k^2 correction terms
                        self.model[diagram_full] = (
                            table_hdu.data['P0L_b1_ell{}'.format(ell)] +
                            table_hdu.data['P1L_b1_ell{}'.format(ell)] +
                            table_hdu.data['Pk2corr_b1_ell{}'.format(ell)])
                    elif diagram == 'PNL_id':
                        # combine tree-level, one-loop
                        # and IR k^2 correction terms
                        self.model[diagram_full] = (
                            table_hdu.data['P0L_id_ell{}'.format(ell)] +
                            table_hdu.data['P1L_id_ell{}'.format(ell)] +
                            table_hdu.data['Pk2corr_id_ell{}'.format(ell)])
                    elif diagram == 'P1L_b1b1':
                        # combine one-loop and IR k^2 correction terms
                        self.model[diagram_full] = (
                            table_hdu.data['P1L_b1b1_ell{}'.format(ell)] +
                            table_hdu.data['Pk2corr_b1b1_ell{}'.format(ell)])
                    else:
                        self.model[diagram_full] = table_hdu.data[diagram_full]
            if not self.validation:
                self.transform_emulator_data()
            else:
                self.convert_dictionary()
        else:
            raise KeyError('HDU table does not contain valid identifiers.')

    def get_flip_and_offset(self, table):
        r"""Compute flip and offset to rescale an input model table.

        Parameters
        ----------
        table: numpy.ndarray
            Model table containing a given term as a function of :math:`k`
            (first index) and the different parameter sample (second index).

        Returns
        -------
        flip: list
            Flip value for each :math:`k` entry.
        max_offset: list
            Maximum offset for each :math:`k` entry.
        """
        flip = []
        offset_list = []
        for i in range(table.shape[0]):
            idmax = np.abs(table[i, :]).argmax()
            flip.append(np.sign(table[i, idmax]))
            offset_list.append(np.abs(np.amin(flip[-1]*table[i, :])))
        if len(offset_list) == 0:
            max_offset = 0.0
        else:
            max_offset = max(offset_list)*1.1
        return flip, max_offset

    def transform(self, table, data_type):
        r"""Rescale the dynamical range of a table for the emulation.

        Applies flip, offset, and rescales the model table. This step is
        performed by first taking the logarithm of the model table, and then
        subtracting its mean and dividing by its standard deviation.

        Parameters
        ----------
        table: numpy.ndarray
            Model table containing a given term as a function of :math:`k`
            (first index) and the different parameter sample (second index).
        data_type: str
            Type of the table that is passed as input. This is used as a
            keyword for the dictionary where the values of flip, offset,
            mean and standard deviation are stored.

        Returns
        -------
        resc_table: numpy.ndarray
            Rescaled table.
        """
        if data_type not in ['PL', 's12', 'sv']:
            self.flip[data_type], self.offset[data_type] = \
                self.get_flip_and_offset(table.T)
        else:
            self.flip[data_type] = np.ones(table.shape[1])
            self.offset[data_type] = 0.0
        temp = np.log10(self.flip[data_type]*table+self.offset[data_type])
        self.mean[data_type] = np.mean(temp, axis=0)
        self.std[data_type] = np.std(temp, axis=0)
        return (temp-self.mean[data_type])/self.std[data_type]

    def transform_inv(self, table, data_type):
        r"""Rescale back the dynamical range of a table.

        Performs the inverse operations defined in the **transform** method.
        Needed to obtain predictions from the emulator in the original metric.

        Parameters
        ----------
        table: numpy.ndarray
            Table containing a given term as a function of :math:`k` (first
            index) and the different parameter sample (second index).
        data_type: str
            Type of the table that is passed as input. This is used as a
            keyword for the dictionary where the values of flip, offset,
            mean and standard deviation are stored.

        Returns
        -------
        resc_table: numpy.ndarray
            Rescaled table.
        """
        return (10**(table*self.std[data_type] + self.mean[data_type]) -
                self.offset[data_type]) * self.flip[data_type]

    def transform_emulator_data(self, data_type=None):
        r"""Rescale class attribute tables.

        Transforms the class attribute model tables from the original metric
        into the emulator one.

        Parameters
        ----------
        data_type: str or list, optional
            Types of the tables which have to be rescaled. If *None*, trasform
            the full set of model tables. Defaults to **None**.
        """
        if data_type is not None:
            data_type = ([data_type] if not isinstance(data_type, list)
                         else data_type)
            for dt in data_type:
                if self.model_transformed is None:
                    self.model_transformed = {}
                    self.mean = {}
                    self.std = {}
                    self.flip = {}
                    self.offset = {}
                self.model_transformed[dt] = self.transform(self.model[dt],
                                                            data_type=dt)
        else:
            self.model_transformed = {}
            self.mean = {}
            self.std = {}
            self.flip = {}
            self.offset = {}
            for ell in [0, 2, 4]:
                temp = np.zeros([self.n_samples, 7*self.nk + 10*self.nkloop])
                cnt = 0
                # only include the cell_ell counterterm
                # (the others are negligible without AP)
                for diagram in ['P0L_b1b1', 'PNL_b1', 'PNL_id',
                                'Pctr_c{}'.format(ell), 'Pctr_b1b1cnlo',
                                'Pctr_b1cnlo', 'Pctr_cnlo']:
                    diagram_full = '{}_ell{}'.format(diagram, ell)
                    temp[:, cnt*self.nk:(cnt+1)*self.nk] = \
                        self.model[diagram_full]/self.model['PL']
                    cnt += 1
                cnt = 0
                for diagram in ['P1L_b1b1', 'P1L_b1b2', 'P1L_b1g2',
                                'P1L_b1g21', 'P1L_b2b2', 'P1L_b2g2',
                                'P1L_g2g2', 'P1L_b2', 'P1L_g2', 'P1L_g21']:
                    diagram_full = '{}_ell{}'.format(diagram, ell)
                    temp[:,
                         7*self.nk + cnt*self.nkloop:7*self.nk +
                         (cnt+1)*self.nkloop] = (
                            self.model[diagram_full][:,
                                                     (self.nk-self.nkloop):] /
                            self.model['PL'][:, (self.nk-self.nkloop):])
                    cnt += 1

                self.model_transformed[ell] = self.transform(temp,
                                                             data_type=ell)

    def convert_dictionary(self):
        r"""Convert format of internal tables.

        Converts the format of the internal model tables, from dictionary
        to numpy.ndarray.
        """
        temp = np.zeros([self.n_samples, self.nk, 1+self.n_diagrams*3])
        temp[:, :, 0] = self.model['PL']
        for ell in [0, 2, 4]:
            for cnt, diagram in enumerate(self.names_diagrams):
                diagram_full = '{}_ell{}'.format(diagram, ell)
                temp[:, :, 1+cnt+self.n_diagrams*int(ell/2)] = \
                    self.model[diagram_full]
                cnt += 1
        self.model = temp

    def GPy_model(self, data_type):
        r"""Run the Gaussian Regression model to build the emulator.

        Defines a kernel for the covariance of the Gaussian process, that is
        obtained as a combination of a squared-exponential (i.e. gaussian),
        Matérn, and white noise kernels (see the `GPy documentation page
        <https://gpy.readthedocs.io/en/deploy/GPy.kern.html>`_ for more
        detailed informations), and trains the emulator using the rescaled
        model tables as input.

        Parameters
        ----------
        data_type: str
            Type of the model table that is used to build the emulator.

        Returns
        -------
        GPy_model: GPy.models.GPregression
            Emulator object.
        """
        kernel = (GPy.kern.RBF(input_dim=self.n_params,
                               variance=np.var(
                                self.model_transformed[data_type]),
                               lengthscale=np.ones(self.n_params),
                               ARD=True) +
                  GPy.kern.Matern32(input_dim=self.n_params,
                                    variance=np.var(
                                        self.model_transformed[data_type]),
                                    lengthscale=np.ones(self.n_params),
                                    ARD=True) +
                  GPy.kern.White(input_dim=self.n_params,
                                 variance=np.var(
                                    self.model_transformed[data_type])))
        # + GPy.kern.Matern32(input_dim=self.n_params,
        #                     variance=np.var(
        #                      self.model_transformed[data_type]),
        #                     lengthscale=np.ones(self.n_params),
        #                     ARD=True)
        # + GPy.kern.RatQuad(input_dim=self.n_params,
        #                    variance=np.var(
        #                      self.model_transformed[data_type]),
        #                    lengthscale=np.ones(self.n_params),
        #                    power=1., ARD=True) \
        return GPy.models.GPRegression(
            self.samples, self.model_transformed[data_type], kernel)
