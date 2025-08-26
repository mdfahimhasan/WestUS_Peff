import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


def make_scatter_line_plots(x1, y1,
                            fontsize, facecolor, edgecolor, alpha, marker_size,
                            xlabel_scatter, ylabel_scatter, x_y_lim_scatter,
                            make_line_plot=True,
                            x2=None, y2=None,
                            year=None,
                            xlabel=None, ylabel=None,
                            line_label_1=None, line_label_2=None):

    # R2 and RMSE calculation
    r2 = r2_score(y_true=y1, y_pred=x1)
    rmse = mean_squared_error(y_true=y1, y_pred=x1, squared=False)

    # for making scatter and line plots side-by-side
    if make_line_plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        # sns.set_style("darkgrid")
        plt.rcParams['font.size'] = fontsize

        # scatter plot (pixel-wise mm/year)
        ax[0].scatter(x1, y1, facecolor=facecolor, edgecolor=edgecolor, s=marker_size, alpha=alpha)
        ax[0].set_ylabel(ylabel_scatter)
        ax[0].set_xlabel(xlabel_scatter)
        ax[0].plot([0, 1], [0, 1], 'gray', transform=ax[0].transAxes)
        ax[0].set_xlim(x_y_lim_scatter)
        ax[0].set_ylim(x_y_lim_scatter)
        fig.text(0.05, 0.95, f'RMSE: {rmse:.2f}\nR²: {r2:.2f}', transform=ax[0].transAxes,
                 fontsize=(fontsize-2), verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgray', alpha=0.8))

        # line plot (annual mean mm/year)
        ax[1].plot(year, x2, label=line_label_1, marker='o')
        ax[1].plot(year, y2, label=line_label_2, marker='o')
        ax[1].set_xticks(year, rotation=45)
        ax[1].set_ylabel(ylabel)
        ax[1].set_xlabel(xlabel)
        ax[1].legend(loc='upper left', fontsize=(fontsize-2))

        plt.tight_layout()

    # for making scatter plot only (pixel-wise mm/year)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.rcParams['font.size'] = fontsize

        # scatter plot
        ax.scatter(x1, y1, facecolor=facecolor, edgecolor=edgecolor, s=marker_size)
        ax.set_ylabel(ylabel_scatter)
        ax.set_xlabel(xlabel_scatter)
        ax.plot([0, 1], [0, 1], 'gray', transform=ax.transAxes)
        ax.set_xlim(x_y_lim_scatter)
        ax.set_ylim(x_y_lim_scatter)
        fig.text(0.05, 0.95, f'RMSE: {rmse:.2f}\nR²: {r2:.2f}', transform=ax.transAxes,
                 fontsize=(fontsize-2), verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgray', alpha=0.8))


def make_line_plot_v1(y1, y2, year, fontsize, xlabel, ylabel, line_label_1, line_label_2,
                      figsize=(10, 4), y_lim=None, legend_pos='upper left',legend='on',
                      savepath=None, no_xticks=False, suptitle=None):

    # line plot (annual mean mm/year)
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(year, y1, label=line_label_1, color='tab:blue', marker='^', linewidth=1)
    ax.plot(year, y2, label=line_label_2, color='tab:green', marker='^', linewidth=1)
    ax.set_xticks(year)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(y_lim)

    # legend
    if legend == 'on':
        ax.legend(loc=legend_pos, fontsize=(fontsize - 2))
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # remove bounding box
    sns.despine(offset=10, trim=True)  # turning of bounding box around the plots

    # suptitle
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize)

    # xticks
    ax.set_xticklabels(labels=year, rotation=45, fontsize=fontsize)
    if no_xticks:
        ax.set_xticklabels(labels=[])

    plt.subplots_adjust(bottom=0.35)

    if savepath is not None:
        fig.savefig(savepath, dpi=300, transparent=True)


def make_line_plot_v2(y1, y2, y3, year, fontsize, xlabel, ylabel, line_label_1, line_label_2, line_label_3,
                      figsize=(10, 4), legend_pos='upper left', legend='on', y_lim=None,
                      savepath=None, no_xticks=False, suptitle=None):

    # line plot (annual mean mm/year)
    fig, ax = plt.subplots(figsize=figsize)
    plt.rcParams['font.size'] = fontsize

    ax.plot(year, y1, label=line_label_1, color='tab:blue', marker='^', linewidth=1)
    ax.plot(year, y2, label=line_label_2, color='tab:orange', marker='^', linewidth=1)
    ax.plot(year, y3, label=line_label_3, color='tab:green', marker='^', linewidth=1)
    ax.set_xticks(year)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # legend
    if legend == 'on':
        ax.legend(loc=legend_pos, fontsize=(fontsize-2))
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # remove bounding box
    sns.despine(offset=10, trim=True)  # turning of bounding box around the plots

    # xticks
    ax.set_xticklabels(labels=year, rotation=90, fontsize=fontsize)
    if no_xticks:
        ax.set_xticklabels(labels=[])

    # suptitle
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize)

    plt.subplots_adjust(bottom=0.35)

    if savepath is not None:
        fig.savefig(savepath, dpi=300, transparent=True)


def make_BOI_netGW_vs_pumping_vs_USGS_scatter_plot(df, x1, y1, x2, y2, error_xmin, error_xmax, hue,
                                                   xlabel1, ylabel1, xlabel2, ylabel2, fontsize, lim,
                                                   scientific_ticks=True, scilimits=(4, 4),
                                                   basin_labels=('GMD4, KS', 'GMD3, KS', 'Republican Basin, CO',
                                                                 'Harquahala INA, AZ', 'Douglas AMA, AZ', 'Diamond Valley, NV'),
                                                   figsize=(12, 8), savepath=None, text_dict=None, legend='on'):

    basin_colors = {'GMD4, KS': '#4c72b0',
                    'GMD3, KS': '#dd8452',
                    'Republican Basin, CO': '#55a868',
                    'Harquahala INA, AZ': '#c44e52',
                    'Douglas AMA, AZ': '#8172b3',
                    'Diamond Valley, NV': '#64b5cd',
                    'Parowan Valley, UT': '#ffb000'}

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    plt.rcParams['font.size'] = fontsize

    sns.scatterplot(data=df, x=x1, y=y1, hue=hue, marker='s', ax=ax[0], palette=basin_colors)
    ax[0].hlines(df[y1], df[error_xmin], df[error_xmax], color='gray', alpha=0.4)
    ax[0].legend_.remove()
    ax[0].plot([0, 1], [0, 1], 'gray', transform=ax[0].transAxes)
    ax[0].set_ylabel(ylabel1)
    ax[0].set_xlabel(xlabel1)
    ax[0].set_xlim(lim)
    ax[0].set_ylim(lim)

    sns.scatterplot(data=df, x=x2, y=y2, hue=hue, marker='s', ax=ax[1], palette=basin_colors)
    ax[1].legend_.remove()
    ax[1].plot([0, 1], [0, 1], 'gray', transform=ax[1].transAxes)
    ax[1].set_ylabel(ylabel2)
    ax[1].set_xlabel(xlabel2)
    ax[1].set_xlim(lim)
    ax[1].set_ylim(lim)

    if scientific_ticks:
        ax[0].ticklabel_format(style='sci', scilimits=scilimits)
        ax[0].tick_params(axis='both', labelsize=fontsize)

        ax[1].ticklabel_format(style='sci', scilimits=scilimits)
        ax[1].tick_params(axis='both', labelsize=fontsize)

    sns.despine(offset=10, trim=True)  # turning of bounding box around the plots

    if text_dict is not None:
        if isinstance(text_dict, dict):
            for i, s in text_dict.items():
                ax[int(i)].text(0.02, 0.98, s, transform=ax[int(i)].transAxes,
                           ha='left', va='top', fontsize=fontsize - 2)
        else:
            raise ValueError('text must be a dictionary')

    if legend == 'on':
        # Create a custom legend
        handles, labels = ax[0].get_legend_handles_labels()

        # Create legend with square markers, adjust marker size as needed
        new_handles = [plt.Line2D([], [], marker='s', color=handle.get_markerfacecolor(), linestyle='None') for handle
                       in handles]    # Skip the first handle as it's the legend title

        ax[0].legend(handles=new_handles, labels=list(basin_labels), title='Basin',
                     loc='lower right', fontsize=(fontsize-4))

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=400, transparent=True)


def make_scatter_plot_irr_area(df, x, y, hue, xlabel, ylabel, fontsize, lim,
                               basin_labels, figsize=(10, 6),
                               scientific_ticks=True, scilimits=(4, 4),
                               savepath=None, legend='on'):
    basin_colors = {'GMD4, KS': '#4c72b0',
                    'GMD3, KS': '#dd8452',
                    'Republican Basin, CO': '#55a868',
                    'Harquahala INA, AZ': '#c44e52',
                    'Douglas AMA, AZ': '#8172b3',
                    'Diamond Valley, NV': '#64b5cd'}

    fig, ax = plt.subplots(figsize=figsize)
    plt.rcParams['font.size'] = fontsize

    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=basin_colors)
    ax.legend_.remove()
    ax.plot([0, 1], [0, 1], 'gray', transform=ax.transAxes)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    sns.despine(offset=10, trim=True)  # turning of bounding box around the plots

    if scientific_ticks:
        ax.ticklabel_format(style='sci', scilimits=scilimits)
        ax.tick_params(axis='both', labelsize=fontsize)

    # legend
    if legend == 'on':
        handles, labels = ax.get_legend_handles_labels()

        # Create legend with square markers, adjust marker size as needed
        new_handles = [plt.Line2D([], [], marker='o', color=handle.get_facecolor()[0], linestyle='None') for handle in
                       handles[0:]]  # Skip the first handle as it's the legend title

        ax.legend(handles=new_handles, labels=list(basin_labels), title='Basin', loc='upper left', fontsize=fontsize)

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300)


def make_scatter_plot(df, x, y,
                      xlabel, ylabel, fontsize, lim,
                      alpha=0.4, edgecolor='blue', facecolor=None,
                      figsize=(10, 6),
                      scientific_ticks=True, scilimits=(4, 4),
                      savepath=None):

    fig, ax = plt.subplots(figsize=figsize)
    plt.rcParams['font.size'] = fontsize

    sns.scatterplot(data=df, x=x, y=y, alpha=alpha, edgecolor=edgecolor, facecolor=facecolor, ax=ax)
    ax.plot([0, 1], [0, 1], 'gray', transform=ax.transAxes)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    if scientific_ticks:
        ax.ticklabel_format(style='sci', scilimits=scilimits)
        ax.tick_params(axis='both', labelsize=fontsize)

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300)


def variable_correlation_plot(variables_to_include, dataframe_csv,
                              output_dir, method='spearman',
                              rename_dict=None):
    """
    Makes correlation heatmap of variables (predictors) used in the model using pearson's/spearman's correlation method.
    *** pearson's method consider linear relationship between variables while spearman's method consider non-linear
    relationship.

    :param variables_to_include: A list  of variables. Names should match with names available in the provided csv.
    :param dataframe_csv: Filepath of dataframe csv.
    :param output_dir: Filepath of output directory to save the image and resulting csv.
    :param method: Can be 'spearman' or 'pearson'. Default set to 'spearman' to measure strengths of non-linear relationship.
    :param rename_dict: A dictionary to change the default names of the variable. Set to None as default to not rename the variables.

    :return: None.
    """
    global df

    if '.csv' in dataframe_csv:
        df = pd.read_csv(dataframe_csv)
    elif '.parquet' in dataframe_csv:
        df = pd.read_parquet(dataframe_csv)

    df = df[variables_to_include]

    if rename_dict is not None:
        df = df.rename(columns=rename_dict)

    # estimating correlations
    corr_coef = round(df.corr(method=method), 2)

    # plotting
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.size'] = 12
    sns.heatmap(corr_coef, cmap='Purples', annot=True, annot_kws={"size": 10})
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variable_correlation.png'), dpi=200)

    # Calculating total value of absolute
    corr_coef = round(df.corr(method=method).abs(), 2)
    corr_coef['sum'] = corr_coef.sum() - 1  # deleting 1 to remove self correlation
    corr_coef.to_csv(os.path.join(output_dir, 'var_correlation.csv'))




