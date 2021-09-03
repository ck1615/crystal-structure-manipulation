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

plt.rcParams['font.size'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

class ModeLandscape:
    """
    This class plots the mode interpolation values
    """

    def __init__(self, direc='.'):
        self.direc = direc
        self.energies = {}
        self.minimal_energies = {}
        self.wd = os.getcwd()
        self.minE = None
        self.httE = None

    def extract_energies(self, verbose=False):

        # Change directory
        files = glob("{}/*.scf.out".format(self.direc))
        # Extract energies
        for fname in files:
            Atoms = read(fname)
            energy = 7 * Atoms.get_potential_energy() / \
                    Atoms.get_global_number_of_atoms()
            coord_string = re.search('interpolated_(.*).scf.out',
                                     fname).group(1).split("_")
            x, y = float(coord_string[0]), float(coord_string[1])
            self.energies[(x, y)] = energy

        #Get minimum
        self.httE = self.energies[(0.000, 0.000)]
        for key in self.energies:
            self.energies[key] -= self.httE
            self.energies[key] *= 1e3

        self.minE = min(list(self.energies.values()))

        # Get LTT and LTO coordinates
        self.original_points = list(np.load("{}/../../CIFs/modevals_dict.npy".
            format(self.direc),
            allow_pickle="TRUE"))

        # Copy to minimal energies
        self.minimal_energies = deepcopy(self.energies)

        if verbose:
            return self.energies

    def symmetry_generate(self):
        """
        This function generates all other polar coordinates related to the
        calculated ones using the symmetry of the modes.

        In particular:
            E(ρ, 45 - θ) =  E(ρ, 45 + θ)
            E(ρ, 90 - θ) =  E(ρ, 90 + θ)
            E(ρ, 360 - θ) = E(ρ, θ)
        """

        symmetry_energies = {}
        for (x, y) in self.energies:
            energy = self.energies[(x, y)]
            try:
                theta = np.arctan(y/x)
            except ZeroDivisionError:
                theta = 0
            rho = np.sqrt(x**2 + y**2)
            phis = [np.pi / 2 -theta, np.pi / 2 + theta, np.pi - theta, \
                    np.pi + theta, 3*np.pi/2 -theta, 3*np.pi / 2 + theta,\
                    2*np.pi - theta]
            for phi in phis:
                a, b = rho*np.cos(phi), rho*np.sin(phi)
                symmetry_energies[(a, b)] = energy

        self.energies.update(symmetry_energies)

    def data_arrays(self):

        x = []
        y = []
        z = []
        for (xs, ys) in self.energies:
            x.append(xs)
            y.append(ys)
            z.append(self.energies[(xs,ys)])

        return x,y,z

    def plot_mexican_hat(self):

        x, y, z = self.data_arrays()
        ax = plt.axes(projection='3d')
        ax.set_zlim(self.minE*1.1, 0)
        xylim = max(x)
        ax.set_xlim(-xylim, xylim)
        ax.set_ylim(-xylim, xylim)

        ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
        plt.savefig("energy_landscape.pdf")

    def plot_contour(self, mode="X", levels=40, cmap='binary', alpha=1):


        #Get original data
        if mode == "X":
            lto_norm = round(norm(self.original_points[0]['X3+']), 3)
            ltt_norm = round(norm(self.original_points[1]['X3+']) / np.sqrt(2),
                             3)
        elif mode == "G":
            lto_norm = round(norm(self.original_points[0]['GM1+']), 3)
            ltt_norm = round(norm(self.original_points[1]['GM1+']) /
                             np.sqrt(2), 3)

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



if __name__ == "__main__":

    Plots = ModeLandscape()
    Plots.extract_energies()
    Plots.symmetry_generate()
    Plots.plot_contour()



