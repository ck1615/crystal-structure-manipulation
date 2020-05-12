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
import os

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

    def extract_energies(self, rhocut=1.5, verbose=False):

        #Change directory
        files = [file for file in os.listdir(self.dirs[0]) if "castep" \
        in file]

        #Extract energies
        for fname in files:
            rho, theta = float(fname.split("_")[1]),\
                        float(fname.split("_")[-1].strip(".castep"))
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
            phis = [90-theta, 90 + theta, 180 - theta, 180 + theta, 270 -\
                    theta, 270 + theta, 360 - theta]
            for phi in phis:
                symmetry_energies[(rho, phi)] = energy

        self.energies.update(symmetry_energies)

    def polar2cartesian(self):

        rhos = []
        thetas = []
        energies = []

        for (rho, theta) in self.energies:
            rhos.append(rho)
            thetas.append(theta)
            energies.append(self.energies[(rho, theta)])

        rhos = np.array(rhos)
        thetas = np.array(thetas)
        energies = np.array(energies)

        x, y = rhos*np.cos(thetas), rhos*np.sin(thetas)

        return x, y, energies

    def plot_mexican_hat(self):

        x,y,z = self.polar2cartesian()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_zlim(-20,0)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2,2)

        ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none');
        plt.savefig("energy_landscape.pdf")

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

    Plots = PlotInterp()
    Plots.extract_energies()
    Plots.symmetry_generate()
    Plots.plot_mexican_hat()
    



