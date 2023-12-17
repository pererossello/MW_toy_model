
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



def plots_and_profiles(fig_size=1080, ratio=2.35, subplots=(2, 3), ts=2, pad=0.1, sw=0.15,
                       xlabel='', ylabel=[''], lpadx=0.15, lpady=0.15, dpi=300, wspace=0.2,
                       hspace=0.1):

    fig_w = fig_size * ratio
    fig_h = fig_size
    fig_width = fig_w / dpi
    fig_height = fig_h / dpi
    fig_size = fig_width * fig_height
    fs = np.sqrt(fig_size)

    I, J = subplots
    fig = plt.figure(
        figsize=(fig_width, fig_height),
        dpi=dpi,)  # Default dpi, will adjust later for saving
    outer_grid = mpl.gridspec.GridSpec(I, J, wspace=wspace, hspace=hspace)

    axes = []

    main_axes = ax = [[None] * J for _ in range(I)]
    sup_axes = ax = [[None] * J for _ in range(I)]

    for j in range(J):

        for i in range(I):
            inner_grid = mpl.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[i, j], wspace=0, width_ratios=[1, 0.3])
            # Create the main axes
            ax_main = fig.add_subplot(inner_grid[0])
            axes.append(ax_main)
            main_axes[i][j] = ax_main

            # ax_main.set_xlim(-lims[j], lims[j])
            # ax_main.set_ylim(-lims[j], lims[j])
            ax_main.set_aspect('equal')


            ax_additional = fig.add_subplot(inner_grid[1], sharey=ax_main)
            plt.setp(ax_additional.get_yticklabels(), visible=False)
            axes.append(ax_additional)
            sup_axes[i][j] = ax_additional
            #change y size of this subplot to match ax_main


            #set grid
            llw = 0.05
            ax_main.grid(color='white', lw=llw*fs, alpha = 0.5)
            #ax_additional.grid(color='white', lw=llw*fs)

            if i==0:
                ax_main.tick_params(labeltop=True, labelbottom=False)
                ax_additional.tick_params(labeltop=True, labelbottom=False)
            else:
                ax_main.set_xlabel(xlabel, fontsize=fs*ts, labelpad=lpadx)
                a=0


            if j==0:
                a=0
                # set xlabel
                ax_main.set_ylabel(ylabel[i], fontsize=fs*ts, labelpad=lpady)
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(
        axis="both",
        which="major",
        labelsize=ts * fs,
        size=fs * sw*4,
        width=fs * sw,
        pad= pad * fs,
        top=True,
        # labelbottom=labelbottom_bool,
        # labeltop=labeltop_bool,
        right=True,
        direction='inout'
        )
        ax.tick_params(axis='both', which="minor", 
        direction='inout',
        top=True,
        right=True,
        size=fs * sw*2.5, width=fs * sw,)

        for spine in ax.spines.values():
            spine.set_linewidth(fs * sw)
    return main_axes, sup_axes, fig, fs


def image_and_profile_plot(fig_size=720, ratio=1, c=0, dpi=300,
                           rat=0.1, hspace=0, wspace=0,
                           lw=0.015, ts=1.5, pad=0.5):

    """
    This plot function only works for hspace, wspace = 0, 0
    and for wr and hr equal to the same value

    Use the c parameter to finetune the ratio because its still not perfect. 
    """


    # so I have to define this ratio so that the scales between plots are not messed up
    ratio_2 = (ratio+rat)/(1+rat) + c

    # ratio_2 = ratio * (1+wr)/(1+hr)

    fig_w, fig_h = fig_size*ratio_2, fig_size
    fig_width = fig_w / dpi
    fig_height = fig_h / dpi
    size = fig_width * fig_height
    fs = np.sqrt(size)
    fig = plt.figure(
        figsize=(fig_width, fig_height),
        dpi=dpi,  # Default dpi, will adjust later for saving
        layout='none',
    )

    gs = mpl.gridspec.GridSpec(2, 2, width_ratios=[1, rat/ratio_2], height_ratios=[rat, 1], 
                               hspace=hspace, wspace=wspace/ratio_2)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    plt.setp(ax_top.get_xticklabels(), visible=False)
    #plt.setp(ax_top.get_yticklabels(), visible=False)
    #plt.setp(ax_right.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    # plt.setp(ax_main.get_yticklabels(), visible=False)
    # plt.setp(ax_main.get_xticklabels(), visible=False)


    axes = [ax_main, ax_top, ax_right]
    for ax in axes:

        ax.tick_params(axis='both', which="major", 
            direction='inout',
            labelsize=ts * fs,
            pad= pad * fs,
            size=fs * 1, width=fs * 0.15,)

        ax.minorticks_on()

        ax.tick_params(axis='both', which="minor", 
            direction='inout',
            size=fs * 0.55, width=fs * 0.125,)
        
        ax.grid(
            which="major",
            linewidth=fs * lw,
            color="black",
        )

        for spine in ax.spines.values():
            spine.set_linewidth(fs * 0.15)

    return fig, [ax_main, ax_top, ax_right], fs


def initialize_figure(
    fig_size=540,
    ratio=1.5,
    fig_w=None, fig_h=None,
    subplots=(1, 1), grid=True, 
    lw=0.015, ts=2, theme=None,
    pad=0.5,
    color='#222222',
    dpi=300,
    sw=0.15,
    wr=None, hr=None, hmerge=None, wmerge=None,
    ylabel='bottom',
    layout='constrained',
    hspace=None, wspace=None,
    tick_direction='inout',
    minor=True,
    top_bool=True,
    projection=None
):
    """
    Initialize a Matplotlib figure with a specified size, aspect ratio, text size, and theme.

    Parameters:
    fig_size (float): The size of the figure.
    ratio (float): The aspect ratio of the figure.
    text_size (float): The base text size for the figure.
    subplots (tuple): The number of subplots, specified as a tuple (rows, cols).
    grid (bool): Whether to display a grid on the figure.
    theme (str): The theme for the figure ("dark" or any other string for a light theme).

    Returns:
    fig (matplotlib.figure.Figure): The initialized Matplotlib figure.
    ax (list): A 2D list of axes for the subplots.
    fs (float): The scaling factor for the figure size.
    """
    if fig_w is None:
        fig_w = fig_size * ratio
        fig_h = fig_size

    fig_width = fig_w / dpi
    fig_height = fig_h / dpi
    fig_size = fig_width * fig_height
    fs = np.sqrt(fig_size)
    fig = plt.figure(
        figsize=(fig_width, fig_height),
        dpi=dpi,  # Default dpi, will adjust later for saving
        layout=layout,
    )

    if wr is None:
        wr_ = [1] * subplots[1]
    else:
        wr_ = wr
    if hr is None:
        hr_ = [1] * subplots[0]
    else:
        hr_ = hr
    

    gs = mpl.gridspec.GridSpec(subplots[0], subplots[1], figure=fig, width_ratios=wr_, height_ratios=hr_, hspace=hspace, wspace=wspace)


    ax = [[None] * subplots[1] for _ in range(subplots[0])]

    if theme == "dark":
        fig.patch.set_facecolor(color)
        plt.rcParams.update({"text.color": "white"})

    for i in range(subplots[0]):
        for j in range(subplots[1]):
            
            if hmerge is not None:
                if i in hmerge:
                    ax[i][j] = fig.add_subplot(gs[i, :])
                else:
                    ax[i][j] = fig.add_subplot(gs[i, j])
            elif wmerge is not None:
                if j in wmerge:
                    ax[i][j] = fig.add_subplot(gs[:, j])
                else:
                    ax[i][j] = fig.add_subplot(gs[i, j])
            else:
                ax[i][j] = fig.add_subplot(gs[i, j], projection=projection)

            if theme == "dark":
                ax[i][j].set_facecolor(color)
                ax[i][j].tick_params(colors="white")
                ax[i][j].spines["bottom"].set_color("white")
                ax[i][j].spines["top"].set_color("white")
                ax[i][j].spines["left"].set_color("white")
                ax[i][j].spines["right"].set_color("white")
                ax[i][j].xaxis.label.set_color("white")
                ax[i][j].yaxis.label.set_color("white")

            #ax[i][j].xaxis.set_tick_params(which="minor", bottom=False)

            if grid:
                ax[i][j].grid(
                    which="major",
                    linewidth=fs * lw,
                    color="white" if theme == "dark" else "black",
                )
            for spine in ax[i][j].spines.values():
                spine.set_linewidth(fs * sw)

            if ylabel == 'bottom':
                labeltop_bool = False
                labelbottom_bool = True
            elif ylabel == 'top':
                labeltop_bool = True
                labelbottom_bool = False
                ax[i][j].xaxis.set_label_position('top')

            else:
                labeltop_bool = True
                labelbottom_bool = True
                ax[i][j].xaxis.set_label_position('both')

            
            ax[i][j].tick_params(
                axis="both",
                which="major",
                labelsize=ts * fs,
                size=fs * sw*2,
                width=fs * sw,
                pad= pad * fs,
                top=top_bool,
                labelbottom=labelbottom_bool,
                labeltop=labeltop_bool,
                right=top_bool,
                direction=tick_direction
            )

            if minor:
                ax[i][j].minorticks_on()
                ax[i][j].tick_params(
                axis="both",
                which="major",
                labelsize=ts * fs,
                size=fs * sw*4,
                width=fs * sw,
                pad= pad * fs,
                top=top_bool,
                labelbottom=labelbottom_bool,
                labeltop=labeltop_bool,
                right=top_bool,
                direction=tick_direction
                )
                ax[i][j].tick_params(axis='both', which="minor", 
                direction=tick_direction,
                top=top_bool,
                right=top_bool,
                size=fs * sw*2.5, width=fs * sw,)

    if hmerge is not None:
        for k in hmerge:
            for l in range(1, subplots[1]):
                fig.delaxes(ax[k][l])

    if wmerge is not None:
        for k in wmerge:
            for l in range(1, subplots[0]):
                fig.delaxes(ax[l][k])
            
    
    return fig, ax, fs, gs

def get_hist_scatter_colors(x_rad, y_rad, 
                            x_lims=[0, 2*np.pi], 
                            y_lims=[-np.pi/2, np.pi/2], 
                            bins=[180, 90],
                            spherical=True):
    fact = 3
    h, xedges, yedges = np.histogram2d(x_rad, y_rad, 
                                       bins=[np.linspace(x_lims[0], x_lims[1], bins[0]*fact), 
                                             np.linspace(y_lims[0], y_lims[1], bins[1]*fact)])
    
    if spherical == True:
        xcenters = (xedges[:-1] + xedges[1:]) / 2 - np.pi
    else:
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    # Determine the bin indices for each point
    x_indices = np.digitize(x_rad, xedges) - 1
    y_indices = np.digitize(y_rad, yedges) - 1

    # Ensure indices are within the valid range
    x_indices = np.clip(x_indices, 0, len(xcenters) - 1)
    y_indices = np.clip(y_indices, 0, len(ycenters) - 1)

    # Retrieve the density values for each point
    colors = h.T[y_indices, x_indices]

    return colors



def initialize_figure_2(
    fig_size=20, 
    fig_w=512, fig_h=512,
    text_size=1, subplots=(1, 1), grid=True, theme="dark",
    color='#222222',
    dpi=300,
    wr=None, hr=None, hmerge=None, wmerge=None,
    layout='constrained',
    hspace=None, wspace=None,
    tick_direction='out',
    minor=False,
    top_bool=False
):
    

    dpi = dpi
    ratio = fig_w / fig_h
    fig_width = fig_w / dpi
    fig_height = fig_h / dpi
    fig_size = fig_width * fig_height
    fs = np.sqrt(fig_size)
    fig = plt.figure(
        figsize=(fig_width, fig_height),
        dpi=dpi,  # Default dpi, will adjust later for saving
        layout=layout,
    )

    if wr is None:
        wr_ = [1] * subplots[1]
    else:
        wr_ = wr
    if hr is None:
        hr_ = [1] * subplots[0]
    else:
        hr_ = hr

    gs = mpl.gridspec.GridSpec(subplots[0], subplots[1], figure=fig, width_ratios=wr_, height_ratios=hr_, hspace=hspace, wspace=wspace)

    axs = [[None] * subplots[1] for _ in range(subplots[0])]


    return fig, axs, fs, gs


def get_faces(center, dxy, dz):
    x, y, z = center
    dx = dxy
    dy = dxy
    # vertices centered at x,y,z with sides dx/2 and dz/2
    # bottom
    v1 = [x - dx / 2, y - dy / 2, z - dz / 2]
    v2 = [x + dx / 2, y - dy / 2, z - dz / 2]
    v3 = [x + dx / 2, y + dy / 2, z - dz / 2]
    v4 = [x - dx / 2, y + dy / 2, z - dz / 2]
    # top
    v5 = [x - dx / 2, y - dy / 2, z + dz / 2]
    v6 = [x + dx / 2, y - dy / 2, z + dz / 2]
    v7 = [x + dx / 2, y + dy / 2, z + dz / 2]
    v8 = [x - dx / 2, y + dy / 2, z + dz / 2]

    vertices = [v1, v2, v3, v4, v5, v6, v7, v8]

    faces = [[vertices[i] for i in [0, 1, 2, 3]],  # bottom
            [vertices[i] for i in [4, 5, 6, 7]],  # top
            [vertices[i] for i in [0, 3, 7, 4]],  # left
            [vertices[i] for i in [1, 2, 6, 5]],  # right
            [vertices[i] for i in [0, 1, 5, 4]],  # front
            [vertices[i] for i in [2, 3, 7, 6]]]  # back

    return faces