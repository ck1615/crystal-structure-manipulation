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

    def __init__(self, direc='.', verbose=False, savefig=True, figsize=10,
                 emin=-65, emax=25, npoints=10, coordmax=1, section='quadrant'):

        self.section = section
        self.modeList = ['X3+_LTO', 'X3+_LTT']
        self.direcs = {
            key:
            '/Users/christopherkeegan/OneDrive - Imperial College London/Documents/PhD_Research/phd-project/Calculations/LBMAO/ModeInterp/q-e/{}/{}'.format(direc, key)
            for key in self.modeList
            }
        self.data = {}
        self.MLs = {}
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

        # Data
        self.energies = {mode: {} for mode in self.modeList}
        self.minimal_energies = {mode: {} for mode in self.modeList}
        self.wd = os.getcwd()
        self.minE = {mode: None for mode in self.modeList}
        self.httE = {mode: None for mode in self.modeList}
        self.verbose = verbose
        self.get_original_points()

    def get_original_points(self):

        # Get LTT and LTO coordinates
        try:
            self.original_points = list(np.load("{}/../CIFs/modevals_dict.npy".
                                        format(self.direc),
                                        allow_pickle="TRUE"))
        except FileNotFoundError:
            self.original_points = list(np.load("{}/CIFs/modevals_dict.npy".
                                        format(self.direc),
                                        allow_pickle="TRUE"))

        # Define scalar product between LTO and LTT
        self.alpha = np.dot(self.original_points[0][self.mode],
                            self.original_points[1][self.mode])

    def extract_energies(self, mode, verbose=False, uneven=False):

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
            #scaling_coeff = 1
            #scaling_coeff = np.sqrt(1 + self.alpha * np.sin(4 * theta))

            #if abs(scaling_coeff - 1) > 1e-5:
            #    self.original2weighted_coordinates[(x, y)] = \
            #        (round(x * scaling_coeff, 3), round(y * scaling_coeff, 3))
            #    x *= scaling_coeff
            #    y *= scaling_coeff
            #else:
            #    self.original2weighted_coordinates[(x, y)] = (x, y)

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

    def symmetry_generate(self, section='quadrant'):
        """
        This function generates all other polar coordinates related to the
        calculated ones using the symmetry of the modes.

        In particular:
            E(ρ, 45 - θ) =  E(ρ, 45 + θ)
            E(ρ, 90 - θ) =  E(ρ, 90 + θ)
            E(ρ, 360 - θ) = E(ρ, θ)

        Parameters:
        -----------
        section: string
            Section of entire energy landscape to generate. Allowed values:
                'quadrant', 'octant', 'semi', 'full'
        """

        symmetry_energies = {}
        for (x, y, theta) in self.energies:
            energy = self.energies[(x, y, theta)]

            rho = np.sqrt(x**2 + y**2)
            if section == 'quadrant':
                phis = [np.pi / 2 - theta]
            elif section == 'octant':
                phis = []
            elif section == 'semi':
                phis == [np.pi / 2 - theta, np.pi / 2 + theta, np.pi - theta]
            elif section == 'full':
                phis = [np.pi / 2 - theta, np.pi / 2 + theta, np.pi - theta,
                        np.pi + theta, 3*np.pi/2 - theta, 3*np.pi / 2 + theta,
                        2*np.pi - theta]

            for phi in phis:
                a, b = round(rho*np.cos(phi), 3), round(rho*np.sin(phi), 3)
                symmetry_energies[(a, b, phi)] = energy

        self.energies.update(symmetry_energies)

    def get_phase_positions(self, mode):
        """
        This function gets the positions in cartesian coordinates of the
        relaxed HTT, LTT, LTO and Pccn phases.
        """
        if 'GM1+' in mode:
            lto_norm = round(norm(self.MLs[mode].original_points[0]['GM1+']),
                             3)
            ltt_norm = round(norm(self.MLs[mode].original_points[1]['GM1+']) /
                             np.sqrt(2), 3)
        elif 'X3+' in mode:
            lto_norm = round(norm(self.MLs[mode].original_points[0]['X3+']), 3)
            ltt_norm = round(norm(self.MLs[mode].original_points[1]['X3+']) /
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

    def data_arrays(self):

        x = []
        y = []
        z = []
        for (xs, ys, theta) in self.energies:
            x.append(xs)
            y.append(ys)
            z.append(self.energies[(xs, ys, theta)])

        return x, y, z

    def plot_mexican_hat(self):

        x, y, z = self.data_arrays()
        ax = plt.axes(projection='3d')
        ax.set_zlim(self.minE*1.1, 0)
        xylim = max(x)
        ax.set_xlim(-xylim, xylim)
        ax.set_ylim(-xylim, xylim)

        ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
        plt.savefig("energy_landscape.pdf")

    def plot_contour(self, levels=40, cmap='jet', alpha=1):

        # Get original data
        lto_norm = round(norm(self.original_points[0][self.mode]), 3)
        ltt_norm = round(norm(self.original_points[1][self.mode]) / np.sqrt(2),
                         3)

        # Define cartesian coordinates for the LTO and LTT coordinates in the
        # reduced energy landscape
        lto_keys = [(lto_norm, 0), (0, lto_norm), (-lto_norm, 0), (0, -lto_norm
                                                                   )]
        ltt_keys = [(ltt_norm, ltt_norm), (-ltt_norm, -ltt_norm), (-ltt_norm,
                    ltt_norm), (ltt_norm, -ltt_norm)]

        ngridx = 100
        ngridy = 100
        x, y, z = self.data_arrays()

        fig, ax1 = plt.subplots(nrows=1)
        # -----------------------
        # Interpolation on a grid
        # -----------------------
        # A contour plot of irregularly spaced data coordinates
        # via interpolation on a grid.
        # Set maximum x and y value to plot [use max(x) and increase by 1%]
        rhocut = max(x) * 1.01

        # Create grid values first.
        xi = np.linspace(-rhocut, rhocut, ngridx)
        yi = np.linspace(-rhocut, rhocut, ngridy)

        # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)

        # Define the contour plot
        ax1.contour(xi, yi, zi, levels=levels, linewidths=0.5, colors='k')
        cntr1 = ax1.contourf(xi, yi, zi, levels=levels, cmap=cmap, vmin=-100,
                             vmax=600, extend='neither', alpha=0.5)

        fig.colorbar(cntr1, ax=ax1, label=r"$E - E_{\mathrm{HTT}}$ / meV/(f.u.)")
        ax1.set_xticks(np.arange(-1.5, 2, step=0.5))

        # Plot raw data points
        ax1.plot(x, y, 'ko', ms=0.4)
        # Plot x = 0, y = 0, y = x and y = -x axes
        ax1.plot(x, np.zeros(len(x)), 'k', linewidth=0.2)
        ax1.plot(np.zeros(len(y)), y, 'k', linewidth=0.2)
        ax1.plot(x, x, 'k', linewidth=0.2)
        ax1.plot(x, -np.array(x), 'k', linewidth=0.2)

        # Plot positions of LTO phase
        for lto_key in lto_keys:
            ax1.scatter(lto_key[0], lto_key[1], marker='x', color='r',
                        s=10)
            ax1.text(lto_key[0] + 0.05, lto_key[1], 'LTO', fontsize=10)
        for ltt_key in ltt_keys:
            ax1.scatter(ltt_key[0], ltt_key[1], marker='x', color='r',
                        s=10)
            ax1.text(ltt_key[0] + 0.05, ltt_key[1], 'LTT', fontsize=10)


        # Set axes limits
        ax1.set(xlim=(-rhocut, rhocut), ylim=(-rhocut, rhocut))

        # Set axis labels
        if mode =="X":
            ax1.set_xlabel(r"$|X_{I4/mmm}^{3+}|\cdot\cos(\theta)$")
            ax1.set_ylabel(r"$|X_{I4/mmm}^{3+}|\cdot\sin(\theta)$")
        elif mode == "G":
            ax1.set_xlabel(r"$|\Gamma_{Pccn}^{1+}|\cdot\cos(\theta)$")
            ax1.set_ylabel(r"$|\Gamma_{Pccn}^{1+}|\cdot\sin(\theta)$")


        plt.tight_layout()
        fig.savefig(self.wd + "/contour.pdf")

    def plot_e_vs_angle(self, title="", dir=""):

        fix, ax = plt.subplots()
        #Axis
        ax.set_xlabel("Order parameter angle between LTO and LTT / $^{\circ}$")
        ax.set_ylabel(r"$E - E_{HTT}(\theta = 0)$ / meV/atom")
        ax.set_xticks(np.arange(0,46.5,1.5),minor=True)
        ax.set_xticks(np.arange(0, 46, 3))
        ax.set_yticks(np.arange(self.minE, 0.1, 1))
        ax.set_yticks(np.arange(self.minE, 0, 0.25), minor=True)

        #Define and plot data
        for i, phase in enumerate(self.phases):
            x = [float(t) for t in self.energies[i].keys()]
            y = [self.energies[i][j] - self.energies[0]['0.0'] for j in \
                    self.energies[i].keys()]
            plt.scatter(x,y, label=phase, s=5)
        plt.legend()
        plt.xlim((0,45))
        plt.tight_layout()
        plt.savefig(self.wd + "/E_vs_OP.pdf")

    def plot_ediff(self):

        fig, ax = plt.subplots()
        #Axis labels
        ax.set_xlabel("Order parameter angle between LTO and LTT / $^{\circ}$")
        ax.set_ylabel("$E_{LTT} - E_{HTT}$ / meV/(f.u.)")
        ax.set_xticks(np.arange(0,46.5,1.5),minor=True)
        ax.set_xticks(np.arange(0,46,3))
        #Data definition and plotting
        x = list(self.energies[1].keys())
        y = [self.energies[1][t] - self.energies[0][t] for t in\
                self.energies[1].keys()]
        ax.scatter(x,y, s=5)
        plt.tight_layout()
        plt.savefig(self.wd + "/Ediff_vs_OP.pdf")

    def paper_plot(self, extent=100):

        # Define plot
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                figsize=(10, 10))

        # Get names of data
        contours = []
        names = list(self.data.keys())

        # Get cutoff
        grid_cut = max(self.data[names[0]][0]) * ( 1 + (extent - 100)
        / 100)
        axis_cut = round(grid_cut / 0.5) * 0.5

        # Create grid
        xi = np.linspace(-grid_cut, grid_cut, grid_points)
        Xi, Yi, = np.meshgrid(xi, xi)

        # Define levels
        levels = np.linspace(self.emin, self.emax, self.npoints)
        np.append(levels, 0)
        levels = np.around(levels, decimals=0)

        # Plot contour plots in upper-half
        for i in range(2):
            ax = axes[0, i]
            x, y, z = data[names[i]]

            # Interpolate
            triang = tri.Triangulation(x, y)
            interpolator = tri.LinearTriInterpolator(triang, z)
            zi = interpolator(Xi, Yi)

            # Plot
            ax.contour(xi, xi, zi, levels=levels,
            linewidth=self.contour_levels_lw, colours='k')
            cntr = ax.contourf(xi, xi, zi, levels=levels, cmap=cmap)

            # Plot raw data
            ax.plot(x, y, 'ko', ms=0.4)
            ax.plot(x, y, 'ko', ms=0.4)

            # Plot axes and diagonals (y = 0, x = 0, y = +/-x)
            ax.plot(x, np.zeros(len(x)), 'k', linewidth=self.datapoints_linewidth)
            ax.plot(np.zeros(len(y)), y, 'k', linewidth=self.datapoints_linewidth)
            ax.plot(x, x, 'k', linewidth=self.datapoints_linewidth)
            ax.plot(x, -np.array(x), 'k', linewidth=self.datapoints_linewidth)

            # Plot phase positions:

                lto_keys, ltt_keys = get_phase_positions(data_name)
        if True:
            for lto_key in lto_keys:
                ax.scatter(lto_key[0], lto_key[1], marker='x', color='r', s=10)
                ax.text(lto_key[0] + 0.05, lto_key[1], 'LTO', fontsize=10)
                continue
            for ltt_key in ltt_keys:
                ax.scatter(ltt_key[0], ltt_key[1], marker='x', color='r', s=10)
                ax.text(ltt_key[0] + 0.05, ltt_key[1], 'LTT', fontsize=10)
        # HTT
        ax.scatter(0, 0, marker='x', color='g', s=10)
        ax.text(0 + 0.05, 0, 'HTT', fontsize=10)

        # Plot minima according to structure
        if data_name == "X3+_LTO":
            point = (1.0, 0.2)
            for i in range(2):
                for l in range(2):
                    for k in range(2):
                        ax.scatter((-1) ** k * (-1) ** l * point[i], (-1) ** (l + 1) * point[(i + 1) % 2], \
                               marker='x', color='k', s=15)
        elif data_name == "X3+_LTT":
            point = (1.0, 0.3)
            for i in range(2):
                for l in range(2):
                    for k in range(2):
                        ax.scatter((-1) ** k * (-1) ** l * point[i], (-1) ** (l + 1) * point[(i + 1) % 2], \
                               marker='x', color='k', s=15)
        # Ticks 
        ax.set_xticks(np.arange(-axis_cut, axis_cut + 0.5, 0.5))
        ax.set_yticks(np.arange(-axis_cut, axis_cut + 0.5, 0.5))
        ax.set(adjustable='box', aspect='equal')
        
        ax.set_xlim((-coordmax, coordmax))
        ax.set_ylim((-coordmax, coordmax))


def main():

    Plots = ModeLandscape()

    for mode in Plots.modeList:
        Plots.extract_energies(mode)


if __name__ == "__main__":

    Plots = ModeLandscape()
    Plots.extract_energies()
    Plots.symmetry_generate()
    Plots.plot_contour()



