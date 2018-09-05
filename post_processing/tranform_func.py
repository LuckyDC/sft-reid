import numpy as np
import scipy

CONFIG = None


def set_transform_args(args):
    global CONFIG
    CONFIG = args


def explicit_eta(data):
    return (1 - CONFIG.eta) * np.identity(data.shape[0]) + CONFIG.eta * data


def implicit_eta(data):
    return np.linalg.inv((1 + CONFIG.eta) * np.identity(data.shape) - CONFIG.eta * data)


def explicit_n(data):
    return np.linalg.matrix_power(data, CONFIG.n)


def exponential_eta(data):
    return scipy.linalg.expm(-CONFIG.eta * (np.identity(data.shape[0]) - data))
