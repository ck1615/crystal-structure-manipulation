#!/usr/bin/env python3
"""
This module contains functions for extracting and plotting the total energy
as a function of order parameter angle.
"""
from matplotlib import cm
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, LinearLocator)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
import os
import re
from glob import glob
from ase.io import read
from copy import deepcopy
from scipy.interpolate import interp1d, interp2d

# Set default values for matplotlib rcParams
plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams["image.aspect"] = 'equal'

class ModeLandscape:
    """
    This class plots the mode interpolation values
    """

    def __init__(self, u=4, modelist= ['X3+_LTO', 'X3+_LTT'], verbose=False, savefig=True, figsize=10,
                 emin=-65, emax=25, npoints=10, coordmax=1, section='quadrant', fontsize=10):

        self.section = section
        self.modeList = modelist
        self.direcs = {
            key:
            '/Users/christopherkeegan/OneDrive - Imperial College London/Documents/PhD_Research/' + 
            'phd-project/Calculations/LBMAO/ModeInterp/q-e/LaU_{}.0/{}'.format(u, key)
            for key in self.modeList
            }
        self.modeTypes = ['X3+' if 'X3+' in mode else 'GM1+' for mode in self.modeList]
        self.phase_positions = None

        # Plot paramters
        self.grid_points = 100
        self.emin = emin
        self.emax = emax
        self.npoints = npoints
        self.cmap = 'jet'
        self.coordmax = coordmax
        self.savefig = savefig
        self.datapoints_linewidth = 0.2
        self.contour_levels_lw = 0.5
        self.fontsize = fontsize

        # Data
        self.energies = {mode: {} for mode in self.modeList}
        self.minimal_energies = {mode: {} for mode in self.modeList}
        self.data = {mode: None for mode in self.modeList}
        self.wd = os.getcwd()
        self.minE = {mode: None for mode in self.modeList}
        self.httE = {mode: None for mode in self.modeList}
        self.verbose = verbose

        # Get original LTO and LTT points
        self.get_original_points()

    def get_original_points(self):

        # Get LTT and LTO coordinates
        self.original_points = {mode: 
        list(np.load("{}/CIFs/modevals_dict.npy". format(self.direcs[mode]),
                    allow_pickle="TRUE"))
                                for mode in self.modeList}

        # Define scalar product between LTO and LTT
        self.alphas = {mode: 
                       np.dot(self.original_points[mode][0][self.modeType[i]], 
                              self.original_points[mode][1][self.modeType[i]])
                              for i, mode in enumerate(self.modeList)
        }

    def extract_energies(self, mode, verbose=False, uneven=False, scaling=False):

        # Get output file locations
        files = glob("{}/EvenSampling/*.scf.out".format(self.direc[mode]))
        if uneven:
            files += glob("{}/UnevenSampling/*.scf.out".format(self.direc[mode]))

        # Extract energies
        for fname in files:
            Atoms = read(fname)
            energy = 7 * Atoms.get_potential_energy() / \
                Atoms.get_global_number_of_atoms()
            coord_string = re.search('interpolated_(.*).scf.out',
                                     fname).group(1).split("_")
            x, y = float(coord_string[0]), float(coord_string[1])

            # Define the angle
            if x == 0:
                theta = 0
            else:
                theta = np.arctan(y/x)

            # Define the scaling coefficient (angle dependent)
            # Does not change angle of new coefficients
            # Arises due to non-orthogonality of basis (M_LTO and M_LTT)
            if scaling:
                scaling_coeff = 1
                scaling_coeff = np.sqrt(1 + self.alpha * np.sin(4 * theta))

                if abs(scaling_coeff - 1) > 1e-5:
                    self.original2weighted_coordinates[(x, y)] = \
                        (round(x * scaling_coeff, 3), round(y * scaling_coeff, 3))
                    x *= scaling_coeff
                    y *= scaling_coeff
                else:
                    self.original2weighted_coordinates[(x, y)] = (x, y)

            self.energies[mode][(round(x, 3), round(y, 3), theta)] = energy

        # Get minimum
        self.httE[mode] = self.energies[mode][(0.000, 0.000, 0.0)]
        for key in self.energies:
            # Shift by HTT energy and scale to meV (hence factor of 1e3)
            self.energies[mode][key] -= self.httE[mode]
            self.energies[mode][key] *= 1e3

        self.minE[mode] = min(list(self.energies[mode].values()))

        # Copy to minimal energies
        self.minimal_energies = deepcopy(self.energies)

        if verbose:
            return self.energies

    def symmetry_generate(self, mode):
        """
        This function generates all other polar coordinates related to the
        calculated ones using the symmetry of the modes.

        In particular:
            E(ρ, 45 - θ) =  E(ρ, 45 + θ)
            E(ρ, 90 - θ) =  E(ρ, 90 + θ)
            E(ρ, 360 - θ) = E(ρ, θ)
        """
        symmetry_energies = {}
        for (x, y, theta) in self.energies[mode]:
            energy = self.energies[mode][(x, y, theta)]

            rho = np.sqrt(x**2 + y**2)
            if self.section == 'quadrant':
                phis = [np.pi / 2 - theta]
            elif self.section == 'octant':
                phis = []
            elif self.section == 'semi':
                phis == [np.pi / 2 - theta, np.pi / 2 + theta, np.pi - theta]
            elif self.section == 'full':
                phis = [np.pi / 2 - theta, np.pi / 2 + theta, np.pi - theta,
                        np.pi + theta, 3*np.pi/2 - theta, 3*np.pi / 2 + theta,
                        2*np.pi - theta]

            for phi in phis:
                a, b = round(rho*np.cos(phi), 3), round(rho*np.sin(phi), 3)
                symmetry_energies[(a, b, phi)] = energy

        self.energies[mode].update(symmetry_energies)

    def get_phase_positions(self, mode):
        """
        This function gets the positions in cartesian coordinates of the
        relaxed HTT, LTT, LTO and Pccn phases.
        """
        if 'GM1+' in mode:
            lto_norm = round(norm(self.original_points[mode][0]['GM1+']),
                             3)
            ltt_norm = round(norm(self.original_points[mode][1]['GM1+']) /
                             np.sqrt(2), 3)
        elif 'X3+' in mode:
            lto_norm = round(norm(self.original_points[mode][0]['X3+']), 3)
            ltt_norm = round(norm(self.original_points[mode][1]['X3+']) /
                             np.sqrt(2), 3)

        if self.section == 'octant':
            lto_keys = [(lto_norm, 0)]
            ltt_keys = [(ltt_norm, ltt_norm)]
        elif self.section == 'quadrant':
            lto_keys = [(lto_norm, 0), (0, lto_norm)]
            ltt_keys = [(ltt_norm, ltt_norm)]
        elif self.section == 'semi':
            lto_keys = [(lto_norm, 0), (0, lto_norm), (-lto_norm, 0)]
            ltt_keys = [(ltt_norm, ltt_norm), (-ltt_norm, ltt_norm)]
        elif self.section == 'full':
            lto_keys = [(lto_norm, 0), (0, lto_norm), (-lto_norm, 0),
                        (0, -lto_norm)]
            ltt_keys = [(ltt_norm, ltt_norm), (-ltt_norm, -ltt_norm),
                        (-ltt_norm, ltt_norm), (ltt_norm, -ltt_norm)]
        else:
            ValueError("Value of section has to be one of: 'octant', " +
                       "'quadrant', 'semi' or 'full'.")

        return lto_keys, ltt_keys

    def data_arrays(self, mode):
        """
        This function converts the dictionary of data to lists.
        """
        x = []
        y = []
        z = []
        for (xs, ys, theta) in self.energies[mode]:
            x.append(xs)
            y.append(ys)
            z.append(self.energies[mode][(xs, ys, theta)])

        return x, y, z

    def get_data(self):

        for mode in self.modeList:
            self.extract_energies(mode)
            self.symmetry_generate(mode)
            self.data[mode] = self.data_arrays(mode)

        return None

    def get_high_symmetry_data(self):
        """
        This function extracts the data along the y = 0 (LTO) and y = x (LTT)
        sections of the contour plot.
        """

        self.high_sym_data = {}
        for mode in self.modeList:
            self.high_sym_data[mode] = {'LTO': {}, 'LTT': {}}
            for coords in self.energies[mode]:
                # Get LTO line data
                if coords[1] == 0:
                    self.high_sym_data[mode]['LTO'][coords[0]] = \
                    self.energies[mode][coords]
                    # If origin add to LTT data
                    if coords[0] == 0:
                        self.high_sym_data[mode]['LTT'][coords[0]] = \
                            self.energies[mode][coords]
                elif coords[0] == coords[1]:
                        coord = round(np.sqrt(2) * coords[0], 3)
                        self.high_sym_data[mode]['LTT'][coord] = self.\
                            energies[mode][coords]

    def plot_high_sym_data(self, ax, ymin=-80, ymax=20, xmin=0.0):
        """
        This function plots as line plots the data along the y = 0 (LTO) and
        y = x (LTT) sections of the energy landscape.
        """
         
        # Extract high-symmetry data
        self.get_high_symmetry_data() 

        # Define x-values for interpolation
        xi = np.linspace(0, self.coordmax)

        # Define labels
        lto_label = r'$\theta = 0$° ($X_{3}^{+}(a, 0)$) LTO strain'
        ltt_label = r'$\theta = 0$° ($X_{3}^{+}(a, a)$) LTT strain'

        # Extract LTO for LTO cell and LTT for LTT cell
        for mode in self.modeList:
            if 'X3+_LTO' in mode:
                lto_dict = self.high_sym_data[mode]['LTO'] 
                lto_key = self.get_phase_positions(mode)[0][0]
                (x_lto, y_lto) = list(lto_dict.keys()), list(lto_dict.values())
                lto_itp = interp1d(x_lto, y_lto)
            elif 'X3+_LTT' in mode:
                ltt_dict = self.high_sym_data[mode]['LTT']
                ltt_key = round(self.get_phase_positions(mode)[0][0] * 
                                np.sqrt(2), 3)
                (x_ltt, y_ltt) = list(ltt_dict.keys()), list(ltt_dict.values())
                ltt_itp = interp1d(x_ltt, y_ltt)

        # Plot data
        ax.scatter(x_lto, y_lto, s=10, color='r', marker='x', label=lto_label)
        ax.scatter(x_ltt, y_ltt, s=10, color='b', marker='o', label=ltt_label)
        ax.plot(xi, lto_itp(xi), color='r', linestyle='--', linewidth=1)
        ax.plot(xi, ltt_itp(xi), color='b', linestyle='-.', linewidth=1) 

        # Place positions of minima
        ax.scatter(lto_key, lto_dict[lto_key], color='k', marker='o', s=20)
        ax.scatter(ltt_key, ltt_dict[ltt_key], color='k', marker='o', s=20)

        # Label plot 
        ax.set_xlabel(r'$|X_{3}^{+}| / Å', fontsize=self.fontsize)
        ax.set_ylabel(r'$E _ E_{\mathrm{HTT}}$ / meV/(f.u.)', fontsize=self.fontsize)
        ax.axhline(y=0, color='k', linestyle='solid', linewidth=self.datapoints_linewidth)

        # Place limits
        ax.legend(loc=(0.01, 0.79))
        ax.set_xticks(np.arange(xmin, self.coordmax + 1e-5, 0.01))
        ax.set_xlim((xmin, self.coordmax))
        ax.set_ylim((ymin, ymax))

    def paper_plot(self, extent=100):

        # Define plot
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                figsize=(10, 10))

        # Get names of data
        contours = []

        # Get cutoff
        axis_cut = round(self.coordmax / 0.5) * 0.5

        # Create grid
        xi = np.linspace(-self.coordmax, self.coordmax, self.grid_points)
        Xi, Yi, = np.meshgrid(xi, xi)

        # Define levels
        levels = np.linspace(self.emin, self.emax, self.npoints)
        np.append(levels, 0)
        levels = np.around(levels, decimals=0)

        # Plot contour plots in upper-half
        for i, mode in enumerate(self.modeList):
            ax = axes[0, i]
            x, y, z = self.data[self.modeList[i]]

            # Interpolate
            triang = tri.Triangulation(x, y)
            interpolator = tri.LinearTriInterpolator(triang, z)
            zi = interpolator(Xi, Yi)

            # Plot
            ax.contour(xi, xi, zi, levels=levels,
            linewidth=self.contour_levels_lw, colours='k')
            cntr = ax.contourf(xi, xi, zi, levels=levels, cmap=self.cmap)

            # Plot raw data
            ax.plot(x, y, 'ko', ms=0.4)
            ax.plot(x, y, 'ko', ms=0.4)

            # Plot axes and diagonals (y = 0, x = 0, y = +/-x)
            ax.axhline(y=0, color='k', linewidth=self.datapoints_linewidth)
            ax.axvline(x=0, color='k', linewidth=self.datapoints_linewidth)
            ax.plot(x, x, 'k', linewidth=self.datapoints_linewidth)
            ax.plot(x, -np.array(x), 'k', linewidth=self.datapoints_linewidth)

            # Plot phase positions
            lto_keys, ltt_keys = self.get_phase_positions(mode)

            for lto_key in lto_keys:
                ax.scatter(lto_key[0], lto_key[1], marker='x', color='r', s=10)
                ax.text(lto_key[0] + 0.05, lto_key[1], 'LTO', fontsize=self.fontsize)
                continue
            for ltt_key in ltt_keys:
                ax.scatter(ltt_key[0], ltt_key[1], marker='x', color='r', s=10)
                ax.text(ltt_key[0] + 0.05, ltt_key[1], 'LTT', fontsize=self.fontsize)

            # HTT
            ax.scatter(0, 0, marker='x', color='g', s=10)
            ax.text(0 + 0.05, 0, 'HTT', fontsize=self.fontsize)

            # Plot minima according to structure
            #if data_name == "X3+_LTO":
            #    point = (1.0, 0.2)
            #    for i in range(2):
            #        for l in range(2):
            #            for k in range(2):
            #                ax.scatter((-1) ** k * (-1) ** l * point[i], (-1) ** (l + 1) * point[(i + 1) % 2], \
            #                       marker='x', color='k', s=15)
            #elif data_name == "X3+_LTT":
            #    point = (1.0, 0.3)
            #    for i in range(2):
            #        for l in range(2):
            #            for k in range(2):
            #                ax.scatter((-1) ** k * (-1) ** l * point[i], (-1) ** (l + 1) * point[(i + 1) % 2], \
            #                       marker='x', color='k', s=15)

            # Ticks 
            ax.set_xticks(np.arange(0.0, axis_cut + 0.5, 0.5))
            ax.set_yticks(np.arange(0.0, axis_cut + 0.5, 0.5))
            ax.set(adjustable='box', aspect='equal')
        
            ax.set_xlim((0, self.coordmax))
            ax.set_ylim((0, self.coordmax))

        # Plot cuts along LTO and LTT lines with LTO and LTT lattice respectively
        self.plot_cuts(axes[1, 0])

        # Align bottom figure to the left
        axes[1, 0].set_anchor((0.0, 0.0))

        fig.savefig('modelandscape_paper.pdf', bbox_inches='tight')


def main(modelist=['X3+_LTO_GM1+_LTT', 'X3+_LTT_GM1+_LTO']):
    MLs = ModeLandscape(u=4, modelist=modelist)
    return None
