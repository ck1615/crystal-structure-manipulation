#!/usr/bin/env python3
"""
This module contains functions that interpolate between two low-symmetry
structures relative to a high-symmetry structure.
"""
from isodistortfile import isodistort
from convert2cell import cif2cell
import os
import time
import numpy as np
import numpy.linalg as la
from copy import deepcopy

class StructureInterp:
    """
    Class for the interpolation between two low-symmetry structures with
    different order parameters relative to a high-symmetry structure.

    Parameters
    ----------
    self.HS, self.LS1, self.LS2: str
        Names of the .cif files of the high-symmetry (HS) and two low-symmetry
        (LS) structures, subgroups of the high-symmetty phase, between which
        the interpolation is to be done.

    self.LS1_sub, self.LS2_sub: str
        Names of the .cif files of the LS structures reduced to a common
        subgroup.

    self.subgroup: int
        Space group number of subgroup.

    self.modeValues: list of two dicts
        List of two dictionaries mapping the distortion mode to the mode vector
        for the LS1 and LS2 structures relative to the HS structure.

    """

    def __init__(self, HSfile, LS1file, LS2file, subgroup, silent=True,
                 verbose=False):
        self.HS = HSfile
        self.LS1 = LS1file
        self.LS2 = LS2file
        self.subgroup = subgroup
        self.silent = silent
        self.modeValues = None
        self.interpList = {}
        self.verbose = verbose

    def reduce_common_subgroup(self):
        """
        This function reduces the two low-symmetry files to a common subgroup
        with common lattice parameters.

        Parameters:
        ----------
        subgroup: int
            The number corresponding to the space group of the common subgroup
            to both low-symmetry structures.

        cellparams: int
            The lattice parameters to use for the common reduction of both
            low-symmetry structures. 0: high-symmetry file, 1: low-symmetry
            file 1, 2: low-symmetry file 2.

        Returns:
        --------
        LS#file_"subgroup".cif: file
            Two CIF files in the same directory for the two reduced structures
        """

        for sfile in [self.LS1, self.LS2]:
            # Initialise ISODISTORT and get reduced structure
            iso = isodistort(sfile, silent=self.silent)
            iso.choose_by_spacegroup(self.subgroup)
            iso.select_space_group()

            # Save CIF file and get new name
            sfile = sfile.replace(".cif", "_" + str(self.subgroup) + ".cif")
            sfile = sfile.replace('.cif', '_{}.cif'.format(self.subgroup))
            iso.saveCif(fname=sfile, close=True)
            count=0
            while sfile not in os.listdir("."):
                time.sleep(1)
                assert count < 10, "Printing CIF file took too long."

        return None

    def get_extremal_mode_vectors(self, save2file=True):
        """
        This function gets the values of the displacements of the irreducible
        sites of the structures between which we are interpolating.
        """
        if self.modeValues is not None:
            print("Extremal modes have already been extracted.")
            return None

        if 'modevals_dict.npy' in os.listdir("."):
            if self.verbose:
                print("Reading end-member mode values from .npy file.")

            self.modeValues = list(np.load('modevals_dict.npy', allow_pickle=
                                           "TRUE"))
        else:
            # Initialise Isodistort instance with HTT
            iso = isodistort(self.HS, silent=self.silent)
            self.modeValues = []

            for lsFile in [self.LS1, self.LS2]:
                iso.load_lowsym_structure(lsFile)
                iso.get_mode_labels()
                self.modeValues.append(iso.modevalues)

            iso.close()

            if self.verbose:
                return self.modeValues
            elif save2file:
                #Save to binary file if requested
                np.save('modevals_dict.npy', self.modeValues)

    def interp_mode_vector(self, mode, rho, theta):

        # Define the vectors
        v1 = np.array(self.modeValues[0][mode])
        v2 = np.array(self.modeValues[1][mode])

        return list(rho * ((v1 / la.norm(v1)) * np.cos(2 * theta) +
                    (v2 / la.norm(v2)) * np.sin(2 * theta)))

    def get_interp_modevalues(self, mode, step=0.1, pc=50,
                              starting_structure=0, point_bounds=None,
                              add_original=True):
        """
        This function gets interpolated mode values by first creating a 2D grid
        of xy-points homogeneously sampling an eigth of the plane (0-45 degrees)
        where x and y run from 0 to the maximum magnitude of the mode vectors
        corresponding to LTT and LTO.

        x_max = max(rho_lto, rho_ltt)

        Each pair (x,y) is converted into a pair (ρ, θ) via the equations:
            ρ = sqrt(x² + y²)
            θ = atan(y/x)
        Conversely, we have:
            x = ρ cos(θ)
            y = ρ sin(θ)

        For each value pair (ρ, θ), a mode vector is generated for GM1+ via the
        formula:
            mode(ρ, θ) = ρ(mode_lto * cos(2θ) + mode_ltt * sin(2θ))
        """
        # Ensure the mode vectors of the low-symmetry structures are generated
        self.get_extremal_mode_vectors()

        # Get data for LTT and LTO structures
        v_lto = np.array(self.modeValues[0][mode])
        v_ltt = np.array(self.modeValues[1][mode])

        # Define the scalar product between the LTO and LTO mode vectors
        self.alpha = np.dot(v_lto, v_ltt)

        # Define the norm of the LTO and LTT structures
        rho_lto = la.norm(v_lto)
        rho_ltt = la.norm(v_ltt)
        x_lto, y_lto = rho_lto, 0.0
        x_ltt, y_ltt = rho_ltt / np.sqrt(2), rho_ltt / np.sqrt(2)

        # Define maximum magnitude to sample (relative)
        rho_max = max(rho_lto, rho_ltt) * (1 + pc / 100)

        # Define grid to sample 2D energy landscape
        if point_bounds is None:
            points = np.arange(0, rho_max, step)
            xy = [(round(xval, 3), round(yval, 3)) for yval in points for xval
                  in points if (yval <= xval)]
        else:
            xmin, xmax, ymin, ymax = point_bounds
            xy = [(xval, yval) for yval in np.arange(ymin, ymax + 1e-5, step)
                  for xval in np.arange(xmin, xmax + 1e-5, step) if
                  (yval <= xval)]

        # Add the LTT and LTO relaxed structures to the list
        if add_original:
            xy.append((x_lto, y_lto))
            xy.append((x_ltt, y_ltt))

        # Convert grid to polar coordinates
        self.cart2polar = {
                xy_vals: (
                    np.sqrt((xy_vals[0]**2 + xy_vals[1]**2) * 
                            (1 + self.alpha * np.sin(4 *
                             np.arctan(xy_vals[1] / xy_vals[0])))),
                    np.arctan(xy_vals[1] / xy_vals[0]))
                if (xy_vals[0] != 0) else
                (np.sqrt(xy_vals[0]**2 + xy_vals[1]**2), 0.0)
                for xy_vals in xy
                }

        # Initialise list of mode vectors
        self.interpList = {}
        for key, values in self.cart2polar.items():
            (rho, theta) = values
            self.interpList[key] = deepcopy(self.modeValues[starting_structure]
                                            )
            self.interpList[key][mode] = self.interp_mode_vector(mode, rho,
                                                                 theta)
        return None

    def print_interp_cif(self, extent='all'):
        # Launch ISODISTORT class instance
        # Load HS and starting-point LS structures
        itp = isodistort(self.HS, silent=self.silent)
        itp.load_lowsym_structure(self.LS1)
        itp.get_mode_labels()

        if extent == 'all':
            for key in self.interpList:
                if self.verbose:
                    print(key)
                # Define filename
                fname = "La2MgO4_interpolated_{:.3f}_{:.3f}.cif".\
                        format(key[0], key[1])

                # Pass if filename already in directory
                if fname in os.listdir('.'):
                    continue

                assert 'subgroup_cif(1).txt' not in os.listdir('.'),\
                    "New structure 'subgroup_cif.txt' downloaded " + \
                    "before previous one was converted."

                # Set the mode values for this interpolation
                # Iterate over interpolation entries
                itp.modevalues = self.interpList[key]
                itp.set_amplitudes()
                itp.save_cif(fname=fname)

        elif extent in self.interpList:
            key = extent
            if self.verbose:
                print(key)

            # Set the mode values for this interpolation
            # Iterate over interpolation entries
            itp.modevalues = self.interpList[key]
            itp.set_amplitudes()
            fname = "La2MgO4_interpolated_{:1.3f}_{:1.3f}.cif".\
                    format(key[0], key[1])
            itp.save_cif(fname=fname)


        # Close ISODISTORT page instance.
        time.sleep(2)
        itp.close()
        return None

    def check_mode_vector(self, filename, mode):
        """
        This function checks whether the mode amplitude of the generated file
        has the expected angle and magnitude.

        Parameters:
        -----------
        filename: str
            CIF file given as interpolated_rho_theta.cif

        Returns:
        """
        seed = filename.strip("interpolated_").strip(".cif")
        (rho, theta) = (float(seed.split("_")[0]), float(seed.split("_")[1]))

        isocheck = isodistort(self.HS, silent=self.silent)
        isocheck.load_lowsym_structure(filename)
        isocheck.get_mode_labels()

        modevalues = isocheck.modevalues[mode]
        isocheck.close()
        N = len(modevalues)
        norm = round(la.norm(modevalues),3)
        try:
            thetas = (180 / np.pi)*np.array([np.arctan(modevalues[i]/\
                modevalues[i+1]) for i in range(0, N, 2)])
        except ZeroDivisionError:
            thetas = np.array([0.0, 0.0, 0.0])
        #assert abs(rho - norm) < 1e-3, "The magnitudes of the modes are not"+\
        #        "the same as what is expected. The expected norm is" + str(rho) + \
         #       " and the calculated norm is " + str(norm)

        #assert abs(thetas[0] - theta) > 1e-3, "The angles are not the same "+\
                #"The expected angle is {.1f} and the calculated angle is {.1f}." % \
                #(theta, thetas[0])
        return modevalues, rho, theta, norm, thetas
