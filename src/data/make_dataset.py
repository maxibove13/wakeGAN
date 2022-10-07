#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Script to process raw CFD data from FL and GR files containing chaman (LES model) simulations of a WF. 
Slices of averaged Ux and Uy are obtained at hub's height. 
Each WT horizontal velocity slices are isolated and cropped in order to obtain two single-channel images. 
The images are not not normalized."""

__author__ = "Maximiliano Bove"
__email__ = "maxibove13@gmail.com"
__status__ = "Development"
__date__ = "08/22"

# Built-in modules
import json
import multiprocessing
import os
import time

# Third-party modules
import numpy as np
from scipy.io import loadmat
from matplotlib import cm, pyplot as plt
import yaml

# Local modules
try:
    from caffa3dMBRi_gz_21_0009 import caffa3dMBRi_gz
except:
    from src.data.caffa3dMBRi_gz_21_0009 import caffa3dMBRi_gz


# Load config file
with open('config.yaml') as file:
    config = yaml.safe_load(file)

# Figures linewidth
LW = config['data']['figures']['lw']

# Load WT coordinates from .mat
wt_coord = loadmat(os.path.join('data','aux', 'metadata', 'coord_layout.mat'))
# Load selected simulation time steps for temporal window in order to calculate UMean
with open(os.path.join('data','aux', 'metadata','sim_steps.json')) as f:
    t = json.load(f)
# Load turns
with open(os.path.join('data','aux', 'metadata','turns.json')) as f:
    turns = json.load(f)


def make_dataset(z, plot_wf_slice, wt_xy, clim):
    """
    Process raw CFD data of a LES WF simulation in order to obtain grayscale images of the horizontal velocity field of each WT.
    # Iterate over field
    Parameters
    ----------
    z : int
        Height of horizontal slices from the ground (Height of WT hub)

    plot_wf_slice : bool
        Whether to plot the horizontal slices or not.

    wt_xy : dict
        Coordinates of Wind Turbine

    clim : tuple
        clim of Ux and Uy

    Returns
    -------
    """

    processes = [] # Create list of processes
    # Iterate over all angles
    for key, value in turns.items():
        # Define the process
        p = multiprocessing.Process(target=make_dataset_per_prec, args=[prec, key, value, z, clim, plot_wf_slice, wt_xy])
        # Start the process
        p.start()
        # Append the process to list of processes
        processes.append(p)

    # Wait for all processes to finish before moving on
    for process in processes:
        process.join()

    return

def make_dataset_per_prec(prec, key, value, z, clim, plot_wf_slice, wt_xy):

    clim_ux = clim[0]
    clim_uy = clim[1]

    # Iterate over field
    for u in ['U', 'UMean']:

        # Initialize figure and subplots
        fig, axs = plt.subplots(2,1, sharex=True)

        # Define case name
        case = f"n{prec}{key}"

        # Iterate over velocity components:
        for comp, comp_names in enumerate(['Ux', 'Uy']):


            # Obtain image and grids from simulation
            image, limits, grid_x, grid_y = get_image_from_simulation(case, z, t[prec], u, comp)

            # Create matplotlib image
            im = axs[comp].imshow(
                np.flip(image, axis=0), 
                cmap=cm.coolwarm, 
                origin='lower', 
                extent=limits, 
                vmin=clim_ux[0] if comp == 0 else clim_uy[0], vmax=clim_ux[1] if comp == 0 else clim_uy[1],
                interpolation='none')

            # Plot WT position as a cross
            wt_xy_sim = wt_xy[key]
            axs[comp].plot(wt_xy_sim[:, 0], wt_xy_sim[:, 1], 'kx', markeredgewidth=1, markersize=4)

            # Iterate over all WT in this simulation
            for wt in range(wt_xy_sim.shape[0]):
                # Find corners around WT slice as indices of field slice and plot its limits
                ids_wt = isolate_wt_slice(wt_xy_sim, wt, axs[comp], grid_x, grid_y, plot=plot_wf_slice)

                # Adjust the indices in order to have the preselected shape
                a,b,c,d = 0,0,0,0
                while (ids_wt[3] - ids_wt[2]) > config['data']['shape'][0]:
                    ids_wt[3] -= 1
                    a += 1
                while (ids_wt[3] - ids_wt[2]) < config['data']['shape'][0]:
                    ids_wt[3] += 1
                    b += 1
                while (ids_wt[1] - ids_wt[0]) > config['data']['shape'][0]:
                    ids_wt[0] += 1
                    c += 1
                while (ids_wt[1] - ids_wt[0]) < config['data']['shape'][0]:
                    ids_wt[1] -= 1
                    d += 1
                # If the adjustment required more than 3 pixels is not acceptable
                assert a <= 3 or b <= 3 or c <= 3 or d <= 3, f"{a},{b},{c},{d}"

                # Crop slice around corners to have one WT image. This is our sample.
                sample = np.flip(image[ids_wt[2]:ids_wt[3], ids_wt[0]:ids_wt[1]], axis=0)
                # Check sample is square
                assert sample.shape[0] == sample.shape[1], f"Samples must be squared. {case}_{wt} shape is {sample.shape}"

                if wt == 0:
                    first_sample_shape = sample.shape
                else:
                    assert first_sample_shape == sample.shape, f"Samples do not have the same shape {first_sample_shape} != {sample.shape}"

                # Save it in gray scales without normalization.
                if comp == 0:
                    plt.imsave(
                        os.path.join('data', 'preprocessed', 'ux',f'{case}{wt}_ux.png'), 
                        sample, 
                        cmap=cm.gray,
                        vmin=config['data']['figures']['clim_ux'][0], vmax=config['data']['figures']['clim_ux'][1]
                        )
                else:
                    plt.imsave(
                        os.path.join('data', 'preprocessed', 'uy',f'{case}{wt}_uy.png'), 
                        sample, 
                        cmap=cm.gray, 
                        vmin=config['data']['figures']['clim_uy'][0], vmax=config['data']['figures']['clim_uy'][1]
                        )

            # Set labels
            if comp == 1:
                axs[comp].set(xlabel='$x$ [m]')
            axs[comp].set(ylabel='$y$ [m]')
            axs[comp].set_title(f'${comp_names}$')
            # Set colorbar
            cbar = fig.colorbar(im, ax=axs[comp], orientation='vertical', aspect=10, pad=0.03)
            cbar.set_label('[ms$^{-1}$]', labelpad=12, rotation=270)

        # Adjust subplots
        plt.subplots_adjust(hspace=0.3)

        if plot_wf_slice:
            # Save figure of Ux and Uy slices
            if u == 'UMean':
                fig.suptitle(f"Horizontal averaged velocity field at z ~ ${z}$ m\n${prec[0]}.{prec[-2:]}$ m/s | {key} = {value}° | {t[prec][0]} - {t[prec][1]}", y=1.05)
                fig.savefig(os.path.join('figures', 'wf_slices', f"um_{case}.png"), facecolor='white', transparent=False, bbox_inches='tight', dpi=300)
            else:
                fig.suptitle(f"Horizontal instantaneous velocity field at z ~ ${z}$ m\n${prec[0]}.{prec[-2:]}$ m/s | {key} = {value}°", y=1.05)
                fig.savefig(os.path.join('figures', 'wf_slices', f"u_{case}.png"), facecolor='white', transparent=False, bbox_inches='tight', dpi=300)

    print(f'{case} data processing finished. Image shape: {sample.shape}')

    return

def get_image_from_simulation(case, z, steps, field, comp):
    """
    Parameters
    ----------
        Create horizontal slices of a chaman simulation and compose them to have a single slice of the entire domain.

    case : str
        Simulation casename.

    z : int
        Height of planes.

    steps : array(3,)
        t[0]: start of simulation statistic calculations
        t[1]: start of temporal window
        t[2]: end of temporal window

    field : str
        Name of field to read.

    comp : int
        Number of velocity component: 0, 1 or 2.

    Returns
    -------
    image : ndarray(m,n)
        Slice of a single velocity component.
    limits : 
    grid_x : ndarray(m,n)
    grid_y : ndarray(m,n)
    """

    # Initialize WF domains limits
    limits = np.zeros(4)

    # Load tuple with number of regions per direction
    regions = config['data']['regions']

    # Iterate over regions in i direction
    for reg_i in range(regions[0]):
        # Iterate over regions in j direction
        for reg_j in range(regions[1]-1, -1, -1):
            # Define the regions number
            reg = regions[1] * reg_i + reg_j

            # Get horizontal slice of this region for field
            U_slice, grid_x_slice, grid_y_slice, limits = get_horizontal_slice(case, z, steps, field, reg, regions[0] * regions[1], limits, comp)

            # Concatenate different regions in j direction
            if reg_j == regions[1]-1:
                image_col = U_slice
                grid_x_col = grid_x_slice
                grid_y_col = grid_y_slice
            else:
                image_col = np.concatenate((image_col, U_slice), axis=0)
                grid_x_col = np.concatenate((grid_x_col, grid_x_slice), axis=0)
                grid_y_col = np.concatenate((grid_y_col, grid_y_slice), axis=0)


        # Concatenate different regions in i direction
        if reg_i == 0:
            image = image_col
            grid_x = grid_x_col
            grid_y = grid_y_col
        else:
            image = np.concatenate((image, image_col), axis=1)
            grid_x = np.concatenate((grid_x, grid_x_col), axis=1)
            grid_y = np.concatenate((grid_y, grid_y_col), axis=1)

        del image_col, grid_x_col, grid_y_col

    return image, limits, grid_x, grid_y

def get_horizontal_slice(case, z, steps, field, reg, n_regions, limits, comp):
    """
        Get a horizontal slice of a given case, field, region, step and height.
    """

     # Load FL and GR for both selected time steps
    if field == 'UMean':
        FL_0, _ = caffa3dMBRi_gz(
        os.path.join('data', 'raw', case, 'rgc'),
        case,
        [reg+1],
        [],
        steps[1],
        True,
        ['Xc','U', 'UMean', 'UUMean', 'CsgsC', 'betaCoef'],
        VTK=False)

    FL_1, GR = caffa3dMBRi_gz(
    os.path.join('data', 'raw', case, 'rgc'),
    case,
    [reg+1],
    [],
    steps[2],
    True,
    ['Xc','U', 'UMean', 'UUMean', 'CsgsC', 'betaCoef'],
    VTK=False)

    # Calculate index of corresponding selected height
    idz = np.abs(GR[reg].Xc[0, 0, :, 2] - z).argmin()

    # Calculate corrected UMean
    if field == 'UMean':
        U = (FL_1[reg].UMean * (steps[2] - steps[0]) - FL_0[reg].UMean * (steps[1] - steps[0])) / (steps[2] - steps[1])
    else:
        U = FL_1[reg].U

    # Get slice from field and adjust to display correctly
    U_slice = np.flip(U[:, :, idz, comp].T, axis=0)

    # Do the same for the grids
    grid_x_slice = np.flip(GR[reg].Xc[:, :, idz, 0].T, axis=0)
    grid_y_slice = np.flip(GR[reg].Xc[:, :, idz, 1].T, axis=0)

    # Calculate domain's limit
    if reg == 0:
        limits[0] = min(GR[reg].Xc[:, 0, idz, 0])
        limits[2] = min(GR[reg].Xc[0, :, idz, 1])
    elif reg == n_regions - 1:
        limits[1] = max(GR[reg].Xc[:, 0, idz, 0])
        limits[3] = max(GR[reg].Xc[0, :, idz, 1])

    return U_slice, grid_x_slice, grid_y_slice, limits
        
def isolate_wt_slice(wt_xy_sim, wt, ax, grid_x, grid_y, plot=True):

    # We want to limit the new domain around the WT a number of factors of its diameter
    lim_around_wt = config['data']['lim_around_wt'] # left, right, top, bottom of the plane around the WT

    # WT diameter
    d = wt_coord['coord_layout'][0, 3, 0]

    # Find corners of subdivision
    corners = [(wt_xy_sim[wt,0]-lim_around_wt[0]*d, wt_xy_sim[wt,1]+lim_around_wt[2]*d), # Top-left
    (wt_xy_sim[wt,0]+lim_around_wt[1]*d, wt_xy_sim[wt,1]+lim_around_wt[2]*d), # Top-right
    (wt_xy_sim[wt,0]-lim_around_wt[0]*d, wt_xy_sim[wt,1]-lim_around_wt[3]*d), # Bottom-left
    (wt_xy_sim[wt,0]+lim_around_wt[1]*d, wt_xy_sim[wt,1]-lim_around_wt[3]*d), # Bottom-right
    ] 
    # Plot subdivision lines
    if plot:
        ax.plot([corners[0][0], corners[1][0]], [corners[0][1], corners[1][1]], color='k', lw=LW) # Top
        ax.plot([corners[0][0], corners[1][0]], [corners[2][1], corners[3][1]], color='k', lw=LW) # Bottom
        ax.plot([corners[0][0], corners[0][0]], [corners[0][1], corners[3][1]], color='k', lw=LW) # Left
        ax.plot([corners[1][0], corners[1][0]], [corners[3][1], corners[1][1]], color='k', lw=LW) # Right

    # Identify indices of corners
    idx_0 = np.abs(grid_x[0, :] - corners[0][0]).argmin()
    idx_1 = np.abs(grid_x[0, :] - corners[1][0]).argmin()
    idy_0 = np.abs(grid_y[:, 0] - corners[0][1]).argmin()
    idy_1 = np.abs(grid_y[:, 0] - corners[3][1]).argmin()
    ids_wt = [idx_0, idx_1, idy_0, idy_1]

    return ids_wt



if __name__ == "__main__":
    tic = time.time()

    # Load WT coordinates
    wt_xy = {}
    for idx, turn in enumerate(turns.keys()):
        wt_xy[turn] = wt_coord['coord_layout'][:,0:2,idx]

    # Define clim for visualizing chaman simulation slices
    clim = (config['data']['figures']['clim_ux'], config['data']['figures']['clim_ux'])

    # Make dataset
    for prec in config['data']['precs']:
        make_dataset(config['data']['z'], config['data']['figures']['plot_wf_slices'], wt_xy, clim)

    toc = time.time()
    print(f"Data process duration: {((toc-tic)/60):.2f} m ")