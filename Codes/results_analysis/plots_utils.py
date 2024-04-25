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
                      figsize=(10, 4), lim=None, legend_pos='upper left'):

    # line plot (annual mean mm/year)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(year, y1, label=line_label_1, color='tab:blue', marker='^', linewidth=1)
    ax.plot(year, y2, label=line_label_2, color='tab:green', marker='^', linewidth=1)
    ax.set_xticks(year)
    ax.set_xticklabels(labels=year, rotation=45)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(lim)
    ax.legend(loc=legend_pos, fontsize=(fontsize-2))


def make_line_plot_v2(y1, y2, y3, year, fontsize, xlabel, ylabel, line_label_1, line_label_2, line_label_3,
                      figsize=(10, 4), legend_pos='upper left'):

    # line plot (annual mean mm/year)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(year, y1, label=line_label_1, color='tab:blue', marker='^', linewidth=1)
    ax.plot(year, y2, label=line_label_2, color='tab:orange', marker='^', linewidth=1)
    ax.plot(year, y3, label=line_label_3, color='tab:green', marker='^', linewidth=1)
    ax.set_xticks(year)
    ax.set_xticklabels(labels=year, rotation=45)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend(loc=legend_pos, fontsize=(fontsize-2))


def make_BOI_netGW_vs_pumping_vs_USGS_scatter_plot(df, x1, y1, error_col, hue, xlabel1, ylabel1, fontsize, lim,
                                                   scientific_ticks=True, scilimits=(4, 4),
                                                   basin_labels=('GMD4, KS', 'GMD3, KS', 'Republican Basin, CO',
                                                                 'Harquahala INA, AZ', 'Douglas AMA, AZ', 'Diamond Valley, NV'),
                                                   x2=None, y2=None, xlabel2=None, ylabel2=None,
                                                   figsize=(12, 8), savepath=None):

    basin_colors = {'GMD4, KS': '#4c72b0',
                    'GMD3, KS': '#dd8452',
                    'Republican Basin, CO': '#55a868',
                    'Harquahala INA, AZ': '#c44e52',
                    'Douglas AMA, AZ': '#8172b3',
                    'Diamond Valley, NV': '#64b5cd'}

    if x2 is not None:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        plt.rcParams['font.size'] = fontsize

        ax[0].errorbar(df[x1], df[y1], xerr=df[error_col], fmt='none', ecolor='gray', alpha=0.2)
        sns.scatterplot(data=df, x=x1, y=y1, hue=hue, marker='s', ax=ax[0], palette=basin_colors)
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

        plt.tight_layout()

        # Create a custom legend
        handles, labels = ax[0].get_legend_handles_labels()

        # Create legend with square markers, adjust marker size as needed
        new_handles = [plt.Line2D([], [], marker='s', color=handle.get_facecolor()[0], linestyle='None') for handle in
                       handles[0:]]  # Skip the first handle as it's the legend title

        ax[0].legend(handles=new_handles, labels=list(basin_labels), title='basin', loc='upper left', fontsize=fontsize)

        if savepath is not None:
            fig.savefig(savepath, dpi=300)

    else:
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams['font.size'] = fontsize

        sns.scatterplot(data=df, x=x1, y=y1, hue=hue, marker='s', ax=ax, palette=basin_colors)
        ax.legend_.remove()
        ax.plot([0, 1], [0, 1], 'gray', transform=ax.transAxes)
        ax.set_ylabel(ylabel1)
        ax.set_xlabel(xlabel1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)

        if scientific_ticks:
            ax.ticklabel_format(style='sci', scilimits=scilimits)
            ax.tick_params(axis='both', labelsize=fontsize)

        handles, labels = ax.get_legend_handles_labels()

        # Create legend with square markers, adjust marker size as needed
        new_handles = [plt.Line2D([], [], marker='s', color=handle.get_facecolor()[0], linestyle='None') for handle in
                       handles[0:]]  # Skip the first handle as it's the legend title

        ax.legend(handles=new_handles, labels=list(basin_labels), title='basin', loc='upper left', fontsize=fontsize)

        if savepath is not None:
            fig.savefig(savepath, dpi=300)


def make_scatter_plot_irr_area(df, x, y, hue, xlabel, ylabel, fontsize, lim,
                               basin_labels, figsize=(10, 6),
                               scientific_ticks=True, scilimits=(4, 4),
                               savepath=None):
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

    if scientific_ticks:
        ax.ticklabel_format(style='sci', scilimits=scilimits)
        ax.tick_params(axis='both', labelsize=fontsize)

    handles, labels = ax.get_legend_handles_labels()

    # Create legend with square markers, adjust marker size as needed
    new_handles = [plt.Line2D([], [], marker='s', color=handle.get_facecolor()[0], linestyle='None') for handle in
                   handles[0:]]  # Skip the first handle as it's the legend title

    ax.legend(handles=new_handles, labels=list(basin_labels), title='basin', loc='upper left', fontsize=fontsize)

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