#!/usr/bin/env python3
"""
This module contains functions for extracting and plotting the total energy
as a function of order parameter angle.
"""
from matplotlib import cm
import numpy as np
import readmixcastep as rc
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, \
        AutoMinorLocator, LinearLocator)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
import os

plt.rcParams['font.size']=14
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12

class ModeLandscape:
    """
    This class plots the mode interpolation values
    """

    def __init__(self):
        self.dirs = ['./']
        self.energies = {}
        self.wd = os.getcwd()
        self.minE = None
        self.httE = None
        self.cartEnergies = {}

    def extract_energies(self, rhocut=1.8, verbose=False):

        #Change directory
        files = [file for file in os.listdir(self.dirs[0]) if "castep" \
        in file]

        #Extract energies
        for fname in files:
            rho, theta = float(fname.split("_")[1]),\
                        (np.pi / 180) * float(fname.split("_")[-1].\
                        strip(".castep"))
            cas = rc.readcas(fname)
            if rho <= rhocut:
                self.energies[(rho, theta)] = 1e3*cas.get_energy()/cas.get_Nions()

        #Get minimum
        self.httE = self.energies[(0.000, 0.0)]
        for key in self.energies:
            self.energies[key] -= self.httE

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
        for (rho, theta) in self.energies:
            energy = self.energies[(rho, theta)]
            phis = [np.pi / 2 -theta, np.pi / 2 + theta, np.pi - theta, \
                    np.pi + theta, 3*np.pi/2 -theta, 3*np.pi / 2 + theta,\
                    2*np.pi - theta]
            for phi in phis:
                symmetry_energies[(rho, phi)] = energy

        self.energies.update(symmetry_energies)

    def polar2cartesian(self):
        x = []
        y = []
        z = []
        for (rho, theta) in self.energies:
            self.cartEnergies[(rho*np.cos(theta), rho*np.sin(theta))] =\
            self.energies[(rho, theta)]

            x.append(rho*np.cos(theta))
            y.append(rho*np.sin(theta))
            z.append(self.energies[(rho, theta)])
        return x,y,z

    def plot_mexican_hat(self):

        x,y,z = self.polar2cartesian()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_zlim(-20,0)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2,2)

        ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none');
        plt.savefig("energy_landscape.pdf")

    def plot_contour(self, rhocut=1.8):

        npts = 200
        ngridx = 100
        ngridy = 100
        x,y,z = self.polar2cartesian()

        fig, ax1 = plt.subplots(nrows=1)

        # -----------------------
        # Interpolation on a grid
        # -----------------------
        # A contour plot of irregularly spaced data coordinates
        # via interpolation on a grid.

        # Create grid values first.
        xi = np.linspace(-rhocut, rhocut, ngridx)
        yi = np.linspace(-rhocut, rhocut, ngridy)

        # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, z)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)

        # Note that scipy.interpolate provides means to interpolate data on a grid
        # as well. The following would be an alternative to the four lines above:
        #from scipy.interpolate import griddata
        #zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')

        ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
        cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")

        fig.colorbar(cntr1, ax=ax1)
        ax1.set_xticks(np.arange(-1.5, 2, step=0.5))
        ax1.plot(x, y, 'ko', ms=3)
        ax1.set(xlim=(-rhocut, rhocut), ylim=(-rhocut, rhocut))
        ax1.set_xlabel(r"$|X_{3}^{+}|\cdot\cos(\theta)$")
        ax1.set_ylabel(r"$|X_{3}^{+}|\cdot\sin(\theta)$")

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
        ax.set_ylabel("$E_{LTT} - E_{HTT}$ / meV/atom")
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



