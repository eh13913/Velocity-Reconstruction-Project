import os, sys, time 
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from numpy import fft




def compute_density_field(data, grid_size=None, n_grid=None, sample_fraction = 0.021):
    """
    A function that computes the density fluctuation field given the particle subsample for a cubic box

    :param data:        The (x, y, z) coords of the particles in the subsample (N x 3 numpy array)
    :param grid_size:   The size of the 3D grid cells in Mpc (OPTIONAL)
    :param n_grid:      The number of grid cells in 1 dimension such that n_total = n_grid^3 (OPTIONAL)
    """

    
    if grid_size is None and n_grid is None:
        raise NameError("Either the grid size or the number of grid cells must be specified")

    # compute the box dimensions
    box_lengths = [np.round((np.max(data[:, i]) - np.min(data[:, i]))) for i in range(np.shape(data)[1])]

    if box_lengths.count(box_lengths[0]) != len(box_lengths):
        raise RuntimeError("Input data is not a cubic box. All three side lengths must be equal. Currently (Lx, Ly, "
                           "Lz) = {}".format(box_lengths))

    # set the side length of the grid
    L = int(box_lengths[0])

    # compute the number of grid cells (if the grid size is provided)
    if grid_size is not None:
        ngrid = 500 / grid_size

        if not ngrid.is_integer():
            raise TypeError("Grid spacing is not compatible with the dimension of the simulation. Box size = {} and "
                            "Grid size = {}".format(L, grid_size))

        if n_grid is not None:
            raise NameError(
                "The grid size and the number of cells cannot be specified at the same time. Please specify only one "
                "of these variables.")

    else:
        ngrid = int(n_grid)
        grid_size = L / ngrid


    print('Computing the matter density field, with a grid size of {} Mpc...'.format(grid_size))

    # get an array of the bin edges for the grid 
    edge_array = np.linspace(-L / 2, L/2, int(ngrid + 1))

    # get a tuple of the bin edges for the histogram
    bin_edges = tuple([edge_array for _ in range(np.shape(data)[1])])

    # compute the density in each grid cell 
    density, _ = np.histogramdd(data, bins=bin_edges)

    return density, edge_array



def compute_density_fluctuation_field(data, grid_size=None, n_grid=None, edge_array=None, sample_fraction=0.021):
    """
    A function that computes the density fluctuation field given the particle subsample for a cubic box

    :param data:        Either the (x, y, z) coords of the particles in the subsample (N x 3 numpy array) or the gridded density field (N x N x N numpy array)
    :param grid_size:   The size of the 3D grid cells in Mpc (OPTIONAL)
    :param n_grid:      The number of grid cells in 1 dimension such that n_total = n_grid^3 (OPTIONAL)
    """

    # compute the input data shape 
    data_shape = np.array(data.shape)

    # if the data is a set of (x, y, z) coords compute the matter density field first
    if (data_shape != data_shape[0]).any():
        density_field, edges = compute_density_field(data=data, grid_size=grid_size, n_grid=n_grid, sample_fraction=sample_fraction)


    else: 
        edges = edge_array
        density_field = data

    if grid_size is None:
        grid_size = edges[1] - edges[0]


    print('Computing the matter density fluctuation field...')

    # compute the average particle density in the cubic box
    p_avg = (6912 ** 3) / (2000 ** 3)

    # compute the normalised density field in the cubic box
    p_box = (density_field / sample_fraction) / (grid_size ** len(density_field.shape))

    # compute the matter density fluctuation field
    delta = p_box / p_avg - 1

    return delta, density_field, edges



def compute_average_velocity_field(particle_sample, density_field=None, edge_array=None, ngrid=None):
    """
    A function that computes the average x, y, z components of the velocity field in each grid cell using the particle subsample.

    :param particle_sample: A (N, 6) array of the particle coordinates and velocities 
    :param density_field: A (M x M x M) grid containing the pre-computed particle counts for each grid cell (OPTIONAL) 
    :param edge_array: An array containing the values of the density grid edges (OPTIONAL) 
    :param ngrid: A integer number of grid cells to be used when computing the density and velocity fields if the density field has not been provided (OPTIONAL) 
    
    """
    
    if density_field is not None:
        
        if edge_array is None:
            raise RuntimeError("An array containing the grid edges must be input if a pre-computed density field is being used.")
            
        density_to_use = density_field

    
    else: 
        if ngrid is None: 
            raise RuntimeError("Either a pre-computed density field must be input, or the desired number of grid cells is required to compute the average velocity field.")

        # compute the density field and edge array 
        density_to_use, edge_array = compute_density_field(particle_sample[:, :3], n_grid=ngrid)
        
        
    print('Computing the average velocity field...')

    # set all 0 values in the density field to 1 so that division later does not raise errors     
    density = np.where(density_to_use == 0, 1, density_to_use)

    # get the tuple of bin edges for the velocity histograms
    bin_edges = tuple([edge_array for _ in range(3)])

    # compute the sum of the velocities in each grid cell 
    vx_grid, _ = np.histogramdd(particle_sample[:, :3], bins=bin_edges, weights=particle_sample[:, 3])
    vy_grid, _ = np.histogramdd(particle_sample[:, :3], bins=bin_edges, weights=particle_sample[:, 4])
    vz_grid, _ = np.histogramdd(particle_sample[:, :3], bins=bin_edges, weights=particle_sample[:, 5])

    # divide the total velocities by the density field to get the averages 
    velocity_field = [vx_grid, vy_grid, vz_grid] / density

    if density_field is not None: 
        return velocity_field

    else: 
        return velocity_field, density_to_use, edge_array

        


def wave_num(kgrid):
    """
    A function that computes the magnitude of the angular wave number k from it's cartesian components on a grid

    :param kgrid: The k-space meshgrid
    """

    kx, ky = kgrid[0], kgrid[1]
    kz = np.zeros_like(kx) if len(kgrid)==2 else kgrid[2]
    
    return np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)



def get_velocity_grids(field, edges):
    """
    A function that returns the grid over which the velocity field will be computed in both real and fourier 
    space. 

    :param field:    The grid corresponding to the matter density field (2D or 3D)
    :param edges:    The side-length of the grid cells used to compute the velocity field.
    """

    # compute the number of dimensions of the density_field 
    N = len(np.shape(field))
    
    if N not in [2, 3]: 
        raise ValueError("Density field does not appear to be either a 2D or 3D field. The density field has a shape {}".format(np.shape(density_field))) 
    

    # compute the side length of the density field (in Mpc) 
    L = np.max(edges) - np.min(edges)
    ngrid = len(edges) - 1

    # kgrid_arrays = [np.fft.fftfreq(ngrid, L / (ngrid-1)) for _ in range(N-1)]
    # kgrid_arrays.extend([np.fft.rfftfreq(ngrid, L / (ngrid-1))])

    # kgrid_arrays = [np.fft.fftfreq(ngrid, L / (ngrid-1)) for _ in range(N)]

    kgrid_arrays = [2 * np.pi * np.fft.fftfreq(ngrid, L / (ngrid)) for _ in range(N)]

    # generate the fourier-space velocity grid
    kgrid = np.meshgrid(*reversed(kgrid_arrays)) if N==2 else np.meshgrid(*kgrid_arrays)
    
    return kgrid



def compute_linear_velocity_field(delta, grid_edges, H0=67, omega_m=0.3):
    """
    A function that computes the linear peculiar velocity field across a grid, using a set cosmology
    and a given matter density field.

    :param delta:   The input matter density field (2D or 3D)
    :param grid_edges:    The edges of the bins used to compude the density field (1D array)
    :param H0:              The hubble constant used in the velocity field computation (default is 67
                            km/s/Mpc from Plank2018)
    :param omega_m:         The cosmological matter density parameter used in the velocity field 
                            computation (default is 0.3 from Plank2018)
    """

    
    print('Computing the linear velocity field...')
    
    # compute f
    f = omega_m ** 0.6

    # compute the number of dimensions of the density field 
    N = len(np.shape(delta))
    
    # compute the real space and fourier space grids needed to compute the velocity field.
    k_grid = get_velocity_grids(delta, grid_edges)

    # # compute the wavenumber at each grid point
    k = wave_num(k_grid)

    # compute the k-space density field
    # delta_k = fft.rfftn(delta)
    delta_k = fft.fftn(delta)
    
    # initialise the fourier space velocity array
    velocity_kx = np.zeros_like(delta_k)
    velocity_ky = np.zeros_like(delta_k)
    velocity_kz = np.zeros_like(delta_k)

    # create the mask for the non-zero wavenumbers
    mask = k != 0
    velocity_kx[mask] = 1j * f * H0 * delta_k[mask] * k_grid[0][mask] / k[mask] ** 2
    velocity_ky[mask] = 1j * f * H0 * delta_k[mask] * k_grid[1][mask] / k[mask] ** 2

    if N == 3: 
        velocity_kz[mask] = 1j * f * H0 * delta_k[mask] * k_grid[2][mask] / k[mask] ** 2
    
    # compute the inverse transformation to get the real space velocity field
    # vx = np.fft.irfftn(velocity_ky, delta.shape)
    # vy = np.fft.irfftn(velocity_kx, delta.shape)
    # vz = np.fft.irfftn(velocity_kz, delta.shape) if N == 3 else np.zeros_like(vx)

    vx = np.real(np.fft.ifftn(velocity_ky, delta.shape))
    vy = np.real(np.fft.ifftn(velocity_kx, delta.shape))
    vz = np.real(np.fft.ifftn(velocity_kz, delta.shape) if N == 3 else np.zeros_like(vx))

    return [vx, vy, vz]





    









