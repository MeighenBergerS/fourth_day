# -*- coding: utf-8 -*-
# This module provided interfaces to PDFs and RNG

import abc
import logging
import numpy as np
import scipy.stats
from typing import Optional, Union, Dict, Any
from .config import config

_log = logging.getLogger(__name__)


class Probability(object, metaclass=abc.ABCMeta):
    """ Abstract class to interface to the probabilities

    Parameters
    ----------
    pdf : object
            The probability distribution class

    Returns
    -------
    None
    """

    @abc.abstractmethod
    def __call__(self, values: np.ndarray):
        """ Calls the pdf

        Parameters
        ----------
        values : np.array
            The values to flatten

        Returns
        -------
        None
        """
        pass


class PDF(object, metaclass=abc.ABCMeta):
    """ Interface class for the pdfs.
    One layer inbetween to allow different
    pdf packages such as scipy and numpy

    Parameters
    ----------
    pdf_interface : class
        Interface class to the pdf
        classes

    Returns
    -------
    None
    """

    @abc.abstractmethod
    def rvs(self, num: int) -> np.ndarray:
        """ Random variate sampling, filled with the subclasses definition

        Parameters
        ----------
        num : int
            Number of samples to draw

        Returns
        -------
        rvs : np.ndarray
            The drawn samples
        """
        pass

    @abc.abstractclassmethod
    def pdf(self, points: Union[float, np.ndarray]) -> np.ndarray:
        """ Calculates the probability of the given points

        Parameters
        ----------
        points : np.ndarray
            The points to evaluate

        Returns
        -------
        np.ndarray
            The probabilities
        """


class ScipyPDF(PDF, metaclass=abc.ABCMeta):
    """ Interface class for the scipy pdfs

    Parameters
    ----------
    None

    Returns
    None
    """
    def rvs(self, num: int, dtype: Optional[type] = None) -> np.ndarray:
        """ Calculates the random variates

        Parameters
        ----------
        num : int
            The number of samples
        dtype : optional
            Type of the output

        Returns
        -------
        rvs : np.ndarray
            The drawn samples
        """
        rvs = self._pdf.rvs(
            size=num, random_state=config["runtime"]["random state"])

        if dtype is not None:
            rvs = np.asarray(rvs, dtype=dtype)
        return rvs

    def pdf(self, points: Union[float, np.ndarray],
            dtype: Optional[type] = None) -> np.ndarray:
        """ Calculates the probabilities

        Parameters
        ----------
        points : np.ndarray
            The points to evaluate
        dtype : optional
            Type of the output

        Returns
        -------
        pdf : np.ndarray
            The probabilities
        """
        pdf = self._pdf.pdf(
            points
        )

        if dtype is not None:
            pdf = np.asarray(pdf, dtype=dtype)
        return np.nan_to_num(pdf)


class Normal(ScipyPDF):
    """ Class for the normal distributon

    Parameters
    ----------
    mean : Union[float, np.ndarray]
    sd : Union[float, np.ndarray]

    Returns
    -------
    None
    """

    def __init__(
            self,
            mean: Union[float, np.ndarray],
            sd: Union[float, np.ndarray],
            max_val=np.infty) -> None:
        """ Initializes the TruncatedNormal class

        Parameters
        ----------
        mean : Union[float, np.ndarray]
        sd : Union[float, np.ndarray]

        Returns
        -------
        None
        """
        super().__init__()
        self._mean = mean
        self._sd = sd

        self._pdf = scipy.stats.norm(
            loc=self._mean, scale=self._sd
        )

class LogNorm(ScipyPDF):
    """ Class for the lognormal distribution

    Parameters
    ----------
    mean : Union[float, np.array]
        The mean value
    sd : Union[float, np.array]
        The std

    Returns
    -------
    None
    """

    def __init__(
            self,
            mean: Union[float, np.ndarray],
            sd: Union[float, np.ndarray],
            max_val=np.infty) -> None:
        """ Initializes the LogNormal class

        Parameters
        ----------
        mean : Union[float, np.ndarray]
        sd : Union[float, np.ndarray]

        Returns
        -------
        None
        """
        super().__init__()
        self._mean = mean
        self._sd = sd

        self._pdf = scipy.stats.lognorm(
            self._mean, scale=self._sd
        )

class Gamma(ScipyPDF):
    """ Class for the scaled gamma distribution

    Parameters
    ----------
    mean : Union[float, np.array]
        The mean value
    sd : Union[float, np.array]
        The std

    Returns
    -------
    None
    """

    def __init__(
            self,
            mean: Union[float, np.ndarray],
            sd: Union[float, np.ndarray],
            max_val=None) -> None:
        """ Initializes the scaled gamma distribution

        Parameters
        ----------
        mean : Union[float, np.array]
            The mean value
        sd : Union[float, np.array]
            The std

        Returns
        -------
        None
        """
        self._mean = mean
        self._sd = sd
        self._beta = self._mean / self._sd**2.
        self._alpha = self._mean**2. / self._sd**2.
        # scipy parameters
        self._shape = self._alpha
        self._scale = 1. / self._beta
        local_gamma = scipy.stats.gamma(
            self._shape,
            scale=self._scale
        )
        self._pdf = local_gamma

def construct_pdf(conf_dict: Dict[str, Any]) -> PDF:
    """Convenience function to create a PDF from a config dict

    Parameters
    ----------
    conf_dict: Dict[str, Any]
        The dict should contain a `class` key with the name of the
        PDF to instantiate. Any further keys will be passed as kwargs

    Returns
    -------
    pdf : PDF
        The constructed pdf

    Raises
    -----
    Keyerror
        The pdf class is unknown
    """
    try:
        conf_dict = dict(conf_dict)
        class_name = conf_dict.pop("class")
        pdf_class = globals()[class_name]
        pdf = pdf_class(**conf_dict)
    except KeyError:
        raise KeyError("Unknown pdf class: %s", class_name)
    return pdf
