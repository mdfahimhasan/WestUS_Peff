import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from Codes.utils.system_ops import makedirs


def calculate_rmse(Y_pred, Y_obsv):
    """
    Calculates RMSE value of model prediction vs observed data.

    :param Y_pred: prediction array or panda series object.
    :param Y_obsv: observed array or panda series object.

    :return: RMSE value.
    """
    if isinstance(Y_pred, np.ndarray):
        Y_pred = Y_pred.reshape(-1, 1)
        Y_obsv = Y_obsv.reshape(-1, 1)
        rmse_val = np.sqrt(np.mean((Y_obsv - Y_pred) ** 2))
    else:  # in case of pandas series
        rmse_val = np.sqrt(np.mean((Y_obsv - Y_pred) ** 2))
    return rmse_val

def calculate_r2(Y_pred, Y_obsv):
    """
    Calculates R2 value of model prediction vs observed data.

    :param Y_pred: prediction array or panda series object.
    :param Y_obsv: observed array or panda series object.

    :return: R2 value.
    """
    if isinstance(Y_pred, np.ndarray):
        Y_pred = Y_pred.reshape(-1, 1)
        Y_obsv = Y_obsv.reshape(-1, 1)
        r2_val = r2_score(Y_obsv, Y_pred)
    else:  # in case of pandas series
        r2_val = r2_score(Y_obsv, Y_pred)
    return r2_val


def scatter_plot(Y_pred, Y_obsv, plot_name, savedir='../Model_Run/Plots'):
    """
    Makes scatter plot of model prediction vs observed data.

    :param plot_name:
    :param Y_pred: flattened prediction array.
    :param Y_obsv: flattened observed array.
    :param savedir: filepath to save the plot.

    :return: A scatter plot of model prediction vs observed data.
    """
    fig, ax = plt.subplots()
    ax.plot(Y_obsv, Y_pred, 'o')
    ax.plot([0, 1], [0, 1], '-r', transform=ax.transAxes)
    ax.set_xlabel('GW Observed (mm)')
    ax.set_ylabel('GW Predicted (mm)')

    r2_val = round(calculate_r2(Y_pred, Y_obsv), 3)
    ax.text(0.1, 0.9, s=f'R2={r2_val}', transform=ax.transAxes)

    makedirs([savedir])
    fig_loc = os.path.join(savedir, plot_name)
    fig.savefig(fig_loc, dpi=300)


def calc_outlier_ranges_IQR(data, axis=None, decrease_lower_range_by=None, increase_upper_range_by=None):
    """
    calculate lower and upper range of outlier detection using IQR method.

    :param data: An array or list. Flattened array or list is preferred. If not flattened, adjust axis argument or
                 preprocess data before giving ito this function.
    :param axis: Axis or axes along which the percentiles are computed. Default set to None for flattened array or list.
    :param decrease_lower_range_by: A user-defined value to decrease lower range of outlier detection.
                                    Default set to None.
    :param increase_upper_range_by: A user-defined value to increase upper range of outlier detection.
                                    Default set to None.

    :return: lower_range, upper_range values of outlier detection.
    """
    q1 = np.nanpercentile(data, 25, axis=axis)
    median = np.nanpercentile(data, 50, axis=axis)
    q3 = np.nanpercentile(data, 75, axis=axis)

    iqr = q3 - q1

    lower_range = np.nanmin([i for i in data if i >= (q1 - 1.5 * iqr)])
    upper_range = np.nanmax([i for i in data if i <= (q3 + 1.5 * iqr)])

    # adjusts lower and upper values by an author-defined range
    if (decrease_lower_range_by is not None) | (increase_upper_range_by is not None):
        if (decrease_lower_range_by is not None) & (increase_upper_range_by is None):
            lower_range = lower_range - decrease_lower_range_by

        elif (increase_upper_range_by is not None) & (decrease_lower_range_by is None):
            upper_range = upper_range + increase_upper_range_by

        elif (increase_upper_range_by is not None) & (decrease_lower_range_by is not None):
            lower_range = lower_range - decrease_lower_range_by
            upper_range = upper_range + increase_upper_range_by

    return lower_range, upper_range, median


def calc_outlier_ranges_MAD(data, axis=None, threshold=3, decrease_lower_range_by=None, increase_upper_range_by=None):
    """
    calculate lower and upper range of outlier detection using Median Absolute Deviation (MAD) method.

    A good paper on MAD-based outlier detection:
    https://www.sciencedirect.com/science/article/pii/S0022103113000668

    :param data: An array or list. Flattened array or list is preferred. If not flattened, adjust axis argument or
                 preprocess data before giving ito this function.
    :param axis: Axis or axes along which the percentiles are computed. Default set to None for flattened array or list.
    :param threshold: Value of threshold to use in MAD method.
    :param decrease_lower_range_by: A user-defined value to decrease lower range of outlier detection.
                                    Default set to None.
    :param increase_upper_range_by: A user-defined value to increase upper range of outlier detection.
                                    Default set to None.

    :return: lower_range, upper_range values of outlier detection.
    """
    # Calculate the median along the specified axis
    median = np.nanmedian(data, axis=axis)

    # Calculate the absolute deviations from the median
    abs_deviation = np.abs(data - median)

    # Calculate the median of the absolute deviations
    MAD = np.nanmedian(abs_deviation, axis=axis)

    lower_range = median - threshold * MAD
    upper_range = median + threshold * MAD

    # adjusts lower and upper values by an author-defined range
    if (decrease_lower_range_by is not None) | (increase_upper_range_by is not None):
        if (decrease_lower_range_by is not None) & (increase_upper_range_by is None):
            lower_range = lower_range - decrease_lower_range_by

        elif (increase_upper_range_by is not None) & (decrease_lower_range_by is None):
            upper_range = upper_range + increase_upper_range_by

        elif (increase_upper_range_by is not None) & (decrease_lower_range_by is not None):
            lower_range = lower_range - decrease_lower_range_by
            upper_range = upper_range + increase_upper_range_by

    return lower_range, upper_range, median