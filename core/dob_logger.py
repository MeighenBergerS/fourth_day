"""
dob_logger.py
Authors: Stephan Meighen-Berger
File to deal with the logger handling.
The user still needs to initialze the logger
to use this
"""
import logging
import inspect


class dob_logger():
    """
    class: dob_logger
    Class to inheret logger properties
    Parameters:
        -None
    Returns:
        -None
    """

    def __init__(self):
        """
        function: __init__
        Initializes
        Parameters:
            -None
        Returns:
            -None
        """

    @property
    def logger(self):
        """
        function: logging
        Function called to implement the logger
        Parameters:
            -None
        Returns:
            -None
        """
        loggerStr = '.'.join([__name__, self.__class__.__name__ + '.' +
                              inspect.stack()[1][3]])
        return logging.getLogger(loggerStr)
