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
import copy
import sys

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

    def __init__(self, HSfile, LS1file, LS2file, subgroup, silent=True, \
            verbose=False):
        self.HS = HSfile
        self.LS1 = LS1file
        self.LS2 = LS2file
        self.LS1sub = self.LS1.replace(".cif", "_" + str(subgroup) + ".cif")
        self.LS2sub = self.LS2.replace(".cif", "_" + str(subgroup) + ".cif")
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
            #Initialise ISODISTORT and get reduced structure
            iso = isodistort(sfile, silent=self.silent)
            iso.choose_by_spacegroup(self.subgroup)
            iso.select_space_group()

            #Save CIF file and get new name
            sfile = sfile.replace(".cif", "_" + str(self.subgroup) + ".cif")
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
            self.modeValues = list(np.load('modevals_dict.npy', allow_pickle=\
                    "TRUE"))
        else:
            #Initialise Isodistort instance with HTT
            iso = isodistort(self.HS, silent=self.silent)

            self.modeValues = []

            for lsFile in [self.LS1sub, self.LS2sub]:
                iso.load_lowsym_structure(lsFile)
                iso.get_mode_labels()
                self.modeValues.append(iso.modevalues)
            if self.verbose:
                return self.modeValues
            elif save2file:
                #Save to binary file if requested
                np.save('modevals_dict.npy', self.modeValues)

    def get_interp_modevalues(self, mode, spimin=1, spimax=1, dspi=2, \
            thetamin=0, thetamax=45.0, dtheta=1.5, save2file=True):
        """
        This function interpolates between the values of the chosen mode.

        Parameters:
        -----------
        mode: str
            The name of the mode whose values are to be interpolated. Note that
            this mode must be present in both end structures.

        theta: float
            The range of "angles" to be considered.

        dtheta: float
            The spacing between the angles of the order parameter for the
            interpolation.
        """
        #Convert to radians
        thetamin *= np.pi / 180
        thetamax *= np.pi / 180
        dtheta *= np.pi / 180

        #Construct range of angles
        tvals = np.arange(thetamin, thetamax + 1e-4, dtheta)
        #Construct range of magnitudes
        spivals = np.arange(spimin, spimax + 1e-4, dspi)

        #Get mode values if None
        if self.modeValues is None:
            self.get_extremal_mode_vectors()

        #Get initial and final order parameter vector
        initial = self.modeValues[0][mode]
        final = self.modeValues[1][mode]
        #Get initial and final mode magnitude
        rho_init = la.norm(np.array(initial))
        rho_final = la.norm(np.array(final))

        if self.verbose:
            print("Initial vector and magnitude:", initial, rho_init)
            print("Final vector and magnitude:", final, rho_final)

        #Get scale factor
        delta = [np.sqrt(2)*final[k]/initial[k] - 1 for k in\
                range(1, len(initial), 2)]
        if self.verbose:
            print('Scale factors: ', delta)

        for t in tvals:
            for spiral_mag in spivals:
                #Get vector of values
                itpList = [0 for i in range(len(initial))]
                itpList[1::2] = ['{0:.6f}'.format(spiral_mag*(even*(np.cos(t) \
                        + delta[i]*np.sin(t)))) for i, even in \
                        enumerate(initial[1::2])]
                itpList[::2] = ['{0:.6f}'.format(spiral_mag*(even*(np.sin(t) +\
                delta[i]*np.sin(t)*np.tan(t)))) for i, even in \
                enumerate(initial[1::2])]
                #Get norm
                norm = la.norm(itpList)
                if t == 0.0:
                    assert abs(norm - spiral_mag*rho_init) < 1e-8, r"Norm for \
                        $\theta = 0$ not equal to LTO magnitude."

                #Extract initial dictionary and set the mode values to interpolated
                #values
                itpDict = copy.deepcopy(self.modeValues[0])
                itpDict[mode] = itpList
                self.interpList[(norm, 180*t/np.pi)] = itpDict

        if self.verbose:
            return self.interpList

    def print_interp_cif(self, print_cell=True):

        #Launch ISODISTORT class instance
        #Load HS and starting-point LS structures
        itp = isodistort(self.HS, silent=self.silent)
        itp.load_lowsym_structure(self.LS1sub)
        itp.get_mode_labels()

        #Set the mode values for this interpolation
            #Iterate over interpolation entries
        for key in self.interpList:
            if self.verbose:
                print(key)

            itp.modevalues = self.interpList[key]
            itp.set_amplitudes()
            fname="interpolated_%.3f_%.1f.cif" % (key[0], key[1])
            itp.saveCif(fname=fname)
            #Make sure file is in directory
            if fname not in os.listdir('.'):
                print("File not in directory, trying to save again.")
                itp.saveCif(fname=fname)
            #Make sure file is not empty
            if os.path.getsize(os.getcwd() + '/' + fname) == 0:
                print("File %s is empty.\nWill delete and try again." % fname)
                itp.saveCif(fname=fname)
            #Convert to a .cell file 
            if print_cell:
                cif2cell(fname)
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

if __name__ == "__main__":

    itp = StructureInterp('La2MgO4_HTT.cif', 'La2MgO4_LTO.cif', \
            'La2MgO4_LTT.cif', 56, silent=True)
    from glob import glob
    mode = 'X3+'
    fname = str(sys.argv[1])

    seed = fname.split(".cif")[0]
    print(seed)
    modevalues, rho, theta, realrho, realthetas = itp.check_mode_vector(fname, mode)
    print("The computed magnitude and angle are: ", realrho, realthetas)
    newseed = "interpolated_%.3f_%.1f" % (realrho, realthetas[0])
    for ext in ["castep", "cell", "cif"]:
        oldf = seed + ".%s" % ext
        newf = newseed + ".%s" % ext
        os.rename(oldf, newf)


