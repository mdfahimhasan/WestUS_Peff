import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


def make_scatter_line_plots(x1, y1,
                            fontsize, facecolor, edgecolor, alpha, marker_size,
                            xlabel_scatter, ylabel_scatter, x_y_lim_scatter,
                            make_line_plot=True,
                            x2=None, y2=None,
                            area_basin_mm2=None,
                            year=None,
                            xlabel_line=None, ylabel_line=None,
                            line_label_1=None, line_label_2=None,
                            area_2km_pixel=4809249000000):

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
        ax[1].plot(year, (x2 * area_2km_pixel / area_basin_mm2), label=line_label_1, marker='o')
        ax[1].plot(year, (y2 * area_2km_pixel / area_basin_mm2), label=line_label_2, marker='o')
        ax[1].set_xticks(year)
        ax[1].set_ylabel(ylabel_line)
        ax[1].set_xlabel(xlabel_line)
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
        ax.plot([0, 1], [0, 1], '-r', transform=ax.transAxes)
        ax.set_xlim(x_y_lim_scatter)
        ax.set_ylim(x_y_lim_scatter)
        fig.text(0.05, 0.95, f'RMSE: {rmse:.2f}\nR²: {r2:.2f}', transform=ax.transAxes,
                 fontsize=(fontsize-2), verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgray', alpha=0.8))


def make_line_plot(x, y,  year, fontsize,xlabel_line, ylabel_line, line_label_1, line_label_2):

    # line plot (annual mean mm/year)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(year, x, label=line_label_1, marker='o')
    ax.plot(year, y, label=line_label_2, marker='o')
    ax.set_xticks(year)
    ax.set_ylabel(ylabel_line)
    ax.set_xlabel(xlabel_line)
    ax.legend(loc='upper left', fontsize=(fontsize-2))


def make_BOI_AF_scatter_plot(df, x, y, hue, xlabel, ylabel, fontsize, lim):
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams['font.size'] = fontsize

    sns.scatterplot(data=df, x=x, y=y, hue=hue, marker='s')
    ax.plot([0, 1], [0, 1], '-r', transform=ax.transAxes)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.ticklabel_format(style='sci', scilimits=(4, 4))
    ax.tick_params(axis='both', labelsize=fontsize)

    # Create a custom legend
    handles, labels = ax.get_legend_handles_labels()

    # Create legend with square markers, adjust marker size as needed
    new_handles = [plt.Line2D([], [], marker='s', color=handle.get_facecolor()[0], linestyle='None') for handle in
                   handles[0:]]  # Skip the first handle as it's the legend title

    ax.legend(handles=new_handles, labels=labels[0:], title='Basin', loc='upper left', fontsize=fontsize)



