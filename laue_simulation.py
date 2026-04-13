#!/usr/bin/env python3
import os
import mplstereonet
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from abc import ABC, abstractmethod
from scipy.spatial import cKDTree
from skimage.draw import disk

from laue_utils import rodrigues

def map_structures(crystal_structure, hkl_targets):
    """
    Maps str objects of simulation objects to their class instances.

    Parameters
    ----------
    crystal_structure : str
        Crystal structure
    hkl_targets : list[str]
        List of (h,k,l) targets

    Returns
    -------
    cs : CrystalStructure
        Crystal structure instance
    hklt : list[str]
        List of (h,k,l) target keys
    """
    # Map crystal stucture
    if crystal_structure.lower() == "cubic":
        cs = CubicCrystalStructure()
    elif crystal_structure.lower() == "tetragonal":
        cs = TetragonalCrystalStructure()
    elif crystal_structure.lower() == "hexagonal":
        cs = HexagonalCrystalStructure()
    else:
        raise Exception(f"Unsupported crystal structure '{crystal_structure}' provided. Allowed are: [cubic, tetragonal, hexagonal].")   

    # Map symmetry targets
    hklt = []
    allowed_hkl_targets = cs.symmetries.keys()
    for t in hkl_targets:
        if t.upper() not in allowed_hkl_targets:
            raise Exception(f"Unkown or forbidden symmetry target '{t}' provided for {cs}. Allowed are {list(allowed_hkl_targets)}.")
        hklt.append(next(key for key in allowed_hkl_targets if key == t.upper()))

    return cs, hklt

class CrystalStructure(ABC):
    """
    Base class for crystal structure for Laue diffraction pattern simulation.
    Lattice: (a,b,c,alpha,beta,gamma)
    NOTE: As of now, (alpha,beta,gamma) are not implemented.
    """
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    @property
    @abstractmethod
    def symmetries(self):
        """
        Allowed high-symmetry points with corresponding symmetry operations 
        (i.e. C_n symmetry (i.e. n-fold rotational symmetry)).

        Returns
        -------
        dict
            Dictionary with symmetry elements
        """
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def space_groups(self):
        """
        Allowed space groups (international notation).

        Returns
        -------
        list[int]
            List with allowed space groups
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def sample_lattice_parameters(min_values, max_values, **kwargs):
        """
        Sample lattice parameters (a,b,c).

        Parameters
        ----------
        min_values : float
            Minimum lattice parameters (Angstrom and deg)
        max_values : float
            Maximum lattice parameters (Angstrom and deg)

        Returns
        -------
        list[float]
            List with lattice parameters
        """
        return NotImplementedError()
    
class CubicCrystalStructure(CrystalStructure):
    """
    Cubic lattice: (a,a,a,90,90,90)
    """
    def __init__(self):
        super().__init__()

    @property
    def symmetries(self):
        return {
            '100': {'C_n': 4, 'hkl': np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float32)},
            '010': {'C_n': 4, 'hkl': np.array([[0, 1, 0], [0, -1, 0]], dtype=np.float32)},
            '001': {'C_n': 4, 'hkl': np.array([[0, 0, 1], [0, 0, -1]], dtype=np.float32)},
            '011': {'C_n': 2, 'hkl': np.array([[0, 1, 1], [0, -1, -1], [0, -1, 1], [0, 1, -1]], dtype=np.float32)},
            '101': {'C_n': 2, 'hkl': np.array([[1, 0, 1], [-1, 0, -1], [-1, 0, 1], [1, 0, -1]], dtype=np.float32)},
            '110': {'C_n': 2, 'hkl': np.array([[1, 1, 0], [-1, -1, 0], [-1, 1, 0], [1, -1, 0]], dtype=np.float32)},
            '111': {'C_n': 3, 'hkl': np.array([[1, 1, 1], [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.float32)}
        }
    
    @property
    def space_groups(self):
        return [221, 225, 229]
    
    @staticmethod
    def sample_lattice_parameters(min_values, max_values):
        a_min, a_max = min_values[0], max_values[0]
        a = np.random.uniform(low=min_values[0], high=max_values[0])
        return [a, a, a, 90, 90, 90]
    
class TetragonalCrystalStructure(CrystalStructure):
    """
    Tetragonal lattice: (a,a,c,90,90,90)
    """
    def __init__(self):
        super().__init__()

    @property
    def symmetries(self):
        return {
            '001': {'C_n': 4, 'hkl': np.array([[0, 0, 1], [0, 0, -1]], dtype=np.float32)},
            '010': {'C_n': 2, 'hkl': np.array([[0, 1, 0], [0, -1, 0]], dtype=np.float32)},
            '100': {'C_n': 2, 'hkl': np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float32)},
            '110': {'C_n': 2, 'hkl': np.array([[1, 1, 0], [-1, -1, 0], [-1, 1, 0], [1, -1, 0]], dtype=np.float32)},
        }
    
    @property
    def space_groups(self):
        return [139]
    
    @staticmethod
    def sample_lattice_parameters(min_values, max_values):
        a, c = np.random.uniform(
            low=[min_values[0], min_values[2]], 
            high=[max_values[0], max_values[2]]
        )
        return [a, a, c, 90, 90, 90]
    
class HexagonalCrystalStructure(CrystalStructure):
    """
    Hexagonal lattice: (a,a,c,90,90,120)
    """
    def __init__(self):
        super().__init__()

    @property
    def symmetries(self):
        return {
            '001': {'C_n': 6, 'hkl': np.array([[0, 0, 1], [0, 0, -1]], dtype=np.float32)},
            '110': {'C_n': 2, 'hkl': np.array([[1, 1, 0], [-1, -1, 0]], dtype=np.float32)}, # note that there are two (110) but one of them looks the same as (010)
        }
    
    @property
    def space_groups(self):
        return [191]
    
    @staticmethod
    def sample_lattice_parameters(min_values, max_values):
        a, c = np.random.uniform(
            low=[min_values[0], min_values[2]], 
            high=[max_values[0], max_values[2]]
        )
        return [a, a, c, 90, 90, 120]

class LaueSimulator:
    """
    Class for simulating back-scattering Laue diffraction patterns.

    Based on: 
    https://www.physics.utoronto.ca/~phy326/laue/Laue_Spot_Patterns.py
    released under the MIT license by David Bailey, University of Toronto
    (access on 2022-10-07)
    """
    def __init__(
            self, 
            space_groups,
            space_groups_weights,
            lat_constants, 
            d_film, 
            y_lim, 
            z_lim, 
            beam_yz, 
            energy_beam=50000, 
            max_plane_index=20
        ):
        """
        Class constructor.

        Parameters
        ----------
        space_groups : list[int]
            Space group of the crystal
        space_groups_weights : list[float]
            Weights of the space groups
        lat_constants : list[float]
            Crystal lattice constants (a,b,c,alpha,beta,gamma)
        d_film : float
            Distance from the X-ray source to the Laue film
        y_lim : list[float]
            Size of the Laue film in y-direction
        z_lim : list[float]
            Size of the Laue film in z-direction
        beam_zy : list[float]
            The beam location on the Laue film
        energy_beam : float
            Energy of the X-ray beam (eV)
        max_plane_index : int
            The number of planes to consider for the simulation
        """
        if len(lat_constants) != 6:
            raise Exception(f"Lattice constants require 6 parameters (a,b,c,α,β,γ).")
        
        assert len(space_groups) == len(space_groups_weights)

        self.space_groups = space_groups
        self.space_groups_weights = np.array(space_groups_weights) / np.sum(space_groups_weights)
        self.d_film = float(d_film)
        self.lat_constants = np.array(lat_constants, dtype=np.float32)
        self.y_lim = np.sort(y_lim)
        self.z_lim = np.sort(z_lim)
        self.beam_yz = np.array(beam_yz)
        self.energy_beam = float(energy_beam)
        self.max_plane_index = int(max_plane_index)

        # Initialize the axes
        self.hkl = None
        self.x_axis = None
        self.y_axis = None
        self.z_axis = None
        self.R = None

        # Initialize array results (simplifies function calls)
        self.y_spots = None
        self.z_spots = None
        self.spot_intensities = None

        # Define real and reciprocal vectors 
        a, b, c, alpha, beta, gamma = self.lat_constants
        alpha, beta, gamma = np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)

        a_vec = np.array([a, 0.0, 0.0])
        b_vec = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])
        c_vec = np.array([
            c * np.cos(beta), 
            c * (np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma), 
            c * np.sqrt(max(0, 1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))/np.sin(gamma))
        ])

        volume = np.dot(a_vec, np.cross(b_vec, c_vec))
        a_star = np.cross(b_vec, c_vec) / volume
        b_star = np.cross(c_vec, a_vec) / volume
        c_star = np.cross(a_vec, b_vec) / volume

        # Define crystal basis in real and reciprocal space
        self.basis = np.column_stack((a_vec, b_vec, c_vec))
        self.basis_star = np.column_stack((a_star, b_star, c_star))

        # Map reciprocal vectors and calculate some constant values

        # ChatGPT
        grid = np.mgrid[
            -self.max_plane_index:self.max_plane_index+1,
            -self.max_plane_index:self.max_plane_index+1,
            -self.max_plane_index:self.max_plane_index+1
        ].reshape(3, -1).T
        self.hkl_pairs = grid[np.any(grid != 0, axis=1)]
        
        G = self.hkl_pairs @ self.basis_star.T
        self.d_hkl = 1.0 / np.linalg.norm(G, axis=1) # planar spacing
        self.hkl_rec_pairs = G * self.d_hkl[...,np.newaxis]
        self.hkl_rec_pairs /= np.linalg.norm(self.hkl_rec_pairs, axis=1, keepdims=True) # normalize vectors
        self.min_wave = 12398.419 / self.energy_beam
        self.hkl2_allowed = np.where(np.sum((self.hkl_pairs / self.lat_constants[:3])**2, axis=1) < (2/self.min_wave)**2)[0]
        self.y_phys_size = np.abs(np.diff(self.y_lim))[0]
        self.z_phys_size = np.abs(np.diff(self.z_lim))[0]

    def set_crystal_axis(self, ax, from_hkl=False):
        """
        Setting the crystal axis. 
        - If from_hkl=True: Follows the plane normal N(h,k,l) convention (reciprocal lattice vector direction) if from_hkl is True.
        - If from_hkl=False: Assumes an orthonormal basis 
        Note that for cubic structures the (h,k,l) values are already orthogonal.

        Parameters
        ----------
        ax : 1D-array[float]
            (h,k,l) crystal axis
        from_hkl : bool
            Whether to use (h,k,l) values
        """
        self.x_axis = self.basis_star @ ax if from_hkl else ax
        self.x_axis = self.x_axis/np.linalg.norm(self.x_axis)

        if self.z_axis is not None:
            self._update_y_axis()
            self._update_z_axis()
            self._update_R_matrix()

    def set_z_axis(self, ax, from_hkl=False):
        """
        Setting the z-axis.

        Parameters
        ----------
        ax : 1D-array[float]
            Coordinates of the z-axis (i.e. beam axis perpendicular to crystal axis)
        from_hkl : bool
            Whether to use (h,k,l) values
        """
        self.z_axis = self.basis_star @ ax if from_hkl else ax
        self.z_axis /= np.linalg.norm(self.z_axis)

        if self.x_axis is not None:
            self._update_y_axis()
            self._update_z_axis()
            self._update_R_matrix()

    def run_simulation(self):
        """
        Running the main Laue spot simulation.
        """
        if self.R is None:
            raise Exception("Please provide both crystal- and z-axis.")

        self._calculate_laue_spots()
        self._sum_spot_intensities()

    def generate_pattern(self, shape, max_spots, remove_fraction=0.0, spot_shift_sigma=0.0, spot_radius=10):
        """
        Generates a Laue pattern.

        Parameters
        ----------
        shape : tuple[int]
            Shape of discretized 2D-array
        max_spots : int
            Maximum number of Laue spots to include
        remove_fraction : float
            Fraction of Laue spots to be randomly removed (between 0 and 1) (still ensures 'max_spots' Laue spots)
        spot_shift_sigma : float
            Spot shift standard deviation used to add random shifts to the Laue spot positions
        spot_radius : int
            (Constant) radii of the Laue spots
        
        Returns
        -------
        2D-array[float]
            Laue diffraction pattern
        """
        # Discretize spot positions
        ypx, zpx, intensities = self._prepare_array_information(shape, max_spots, remove_fraction)

        # Add random shifts to spots
        ypx += np.random.normal(loc=0.0, scale=spot_shift_sigma, size=ypx.shape)
        zpx += np.random.normal(loc=0.0, scale=spot_shift_sigma, size=zpx.shape)

        # Initialize discretized array
        img = np.zeros(shape=shape, dtype=np.float32) 

        for i, (y0, z0) in enumerate(zip(ypx, zpx)):
            rr, cc = disk((z0, y0), radius=spot_radius, shape=shape)
            img[rr, cc] = 1.0

        return img

    def _prepare_array_information(self, shape, max_spots, remove_fraction):
        """
        Prepares the mapping of the (y,z,I) Laue spot information to a 2D array of certain shape.

        Parameters
        ----------
        shape : tuple[int]
            Shape of discretized 2D-array
        max_spots : int
            Maximum number of Laue spots to consider
        remove_fraction : float
            Fraction of Laue spots to be randomly removed (between 0 and 1)

        Returns
        -------
        1D-array[float]
            y-coordinates of the Laue spots
        1D-array[float]
            z-coordinates of the Laue spots
        1D-array[int]
            Spot intensities that fulfill the intensity threshold condition
        """
        y_scaling = shape[0]/self.y_phys_size
        z_scaling = shape[1]/self.z_phys_size
        ypx = y_scaling*(self.y_spots-np.min(self.y_lim))
        zpx = z_scaling*(self.z_spots-np.min(self.z_lim))
        n_spots = int(np.min([self.spot_intensities.size-1, int(max_spots * (1 + remove_fraction))]))

        # Peaks separated by distance
        min_dist = int(min(0.035 * np.array(shape))) # minimum separation distance of peaks
        idx_sorted = np.argsort(self.spot_intensities)[::-1]
        selected = []
        selected_coords = []
        tree = None

        for idx in idx_sorted:
            if len(selected) >= n_spots:
                break
            point = (zpx[idx], ypx[idx])
            # if we've got a tree, query it
            if tree is not None:
                if tree.query_ball_point(point, min_dist):
                    continue
            selected.append(idx)
            selected_coords.append(point)
            # rebuild tree (cheap for small n_peaks)
            tree = cKDTree(selected_coords)

        ids_spots = np.array(selected)

        # Remove random spots (weighted probabilitites based on intensities)
        if remove_fraction > 0:
            n_to_rmv = max(n_spots - max_spots, 0)
            probs = np.ones_like(ids_spots) # probability of removing spots
            ids_to_rmv = np.random.choice(ids_spots.size, size=(n_to_rmv,), replace=False, p=probs/probs.sum())
            ids_spots = np.delete(ids_spots, ids_to_rmv)

        return ypx[ids_spots].squeeze(), zpx[ids_spots].squeeze(), self.spot_intensities[ids_spots].squeeze()

    def _sum_spot_intensities(self, epsilon=0.001):
        """
        Loops over the Laue spots and adds their intensities if they are within a certain range.
        
        Parameters
        ---------
        epsilon : float
            Within this range the intensities are summed
        """
        coords = np.column_stack((self.y_spots, self.z_spots))
        tree = cKDTree(coords)

        # Build adjacency list (who’s within epsilon of who)
        pairs = tree.query_pairs(epsilon)

        # Union–find to group
        parent = np.arange(len(coords))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pj] = pi

        for i, j in pairs:
            union(i, j)

        # Flatten roots
        for i in range(len(parent)):
            parent[i] = find(i)

        # Aggregate intensities by root
        root_ids, inverse = np.unique(parent, return_inverse=True)
        summed = np.zeros(len(root_ids), dtype=self.spot_intensities.dtype)
        np.add.at(summed, inverse, self.spot_intensities)

        # Assign back (first point in cluster keeps summed value)
        spot_intensities = np.zeros_like(self.spot_intensities)
        first_idx = {}
        for idx, root in enumerate(parent):
            if root not in first_idx:
                first_idx[root] = idx
                spot_intensities[idx] = summed[np.where(root_ids == root)[0][0]]

        self.spot_intensities = spot_intensities

    def _update_y_axis(self):
        """
        Updates the y-axis.
        """
        self.y_axis = np.cross(self.z_axis, self.x_axis)
        self.y_axis /= np.linalg.norm(self.y_axis)

    def _update_z_axis(self):
        """
        Updates the z-axis.
        """
        self.z_axis = np.cross(self.x_axis, self.y_axis)
        self.z_axis /= np.linalg.norm(self.z_axis)

    def _update_R_matrix(self):
        """
        Updates the rotation matrix.
        """
        self.R = np.vstack([self.x_axis, self.y_axis, self.z_axis])

    def _get_cosine_angles(self):
        """
        Calculates the cosine angles.

        Returns
        -------
        hkl_all : 2D-array[float]
            Valid (h,k,l) planes for Laue back-scattering (in the lab frame) 
            (within 45 degrees of the beam direction)
        structure_factors : 1D-array[float]
            Structure factors for given (h,k,l) spots
        wavelengths : 1D-array[float]
            Wavelengths corresponding to the valid (h,k,l) coordinates
        """
        hkl_lab = (self.R @ self.hkl_rec_pairs.T).T
        cosang = hkl_lab[:,0]
        ids = np.intersect1d(np.where(cosang > 1/np.sqrt(2))[0], self.hkl2_allowed)
        hkl_all = hkl_lab[ids]
        structure_factors = np.sum([w * self._structure_factor(sg, self.hkl_pairs[ids,:]).astype(np.float32) for sg, w in zip(self.space_groups, self.space_groups_weights)], axis=0)
        wavelengths = cosang[ids] / self.d_hkl[ids]

        return hkl_all, structure_factors, wavelengths

    def _calculate_laue_spots(self):
        """
        Calculates the reflected Laue spots.
        """
        hkl_all, structure_factors, wavelengths = self._get_cosine_angles()
        y, z, theta = self._spot_yz_from_hkl(hkl_all)

        # Calculate the intensity
        theta = np.pi/2 - theta[...,np.newaxis]
        wavelengths = wavelengths[...,np.newaxis]

        I = np.divide(
                np.multiply(
                    (1 + np.cos(2*theta)**2) * (wavelengths/self.min_wave - 1) / wavelengths**3,
                    structure_factors * (np.sin(theta)**2 - np.cos(theta)**2)**3
                ), 
            np.sin(theta)**4
        )

        # Assign value of 0 to intensities with theta = np.pi/2 (peak damped by direct beam)
        I[theta == np.pi/2] = 0.0

        # Remove entries with invalid y or z position
        ids_to_rmv = np.where(
            np.logical_or.reduce(
                (
                    np.isnan(y), np.isnan(z),
                    y > self.y_lim[1], y < self.y_lim[0], 
                    z > self.z_lim[1], z < self.z_lim[0],
                    wavelengths.squeeze() < self.min_wave
                )
            ) == True
        )[0][...,np.newaxis]

        valid_mask = np.ones(len(y), dtype=bool)
        valid_mask[ids_to_rmv] = False
        y = y[valid_mask]
        z = z[valid_mask]
        I = I[valid_mask]

        self.y_spots = np.asarray(y).squeeze()
        self.z_spots = np.asarray(z).squeeze()
        self.spot_intensities = np.asarray(I).squeeze()

    def _structure_factor(self, space_group, hkl):
        """
        Calculates the structure factor for the given space group.

        Parameters
        ----------
        space_group : int
            Space group of the crystal
        hkl : 2D-array[float]
            (h,k,l) planes

        Returns
        -------
        allowed : 2D-array[bool]
            Logical array describing the allowed (h,k,l) planes based on the space group
        """
        h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]

        if space_group == 191 or space_group == 221: # primitive cubic e.g. for SrTiO3
            # Any h,k,l are allowed.
            return np.ones((hkl.shape[0], 1), dtype=bool)
        elif space_group == 225: # face centered cubic
            # h,k,l must be all even or all odd, not mixed
            even_hkl =  (h % 2 == 0) & (k % 2 == 0) & (l % 2 == 0)
            odd_hkl = (h % 2 != 0) & (k % 2 != 0) & (l % 2 != 0)
            allowed = even_hkl | odd_hkl
        elif space_group == 227: # face centered diamond
            # Either h,k,l all even and h+k+l = 4N (N an integer) or h,k,l all odd and h+k+l = 4NÂ±1
            sum_hkl = h + k + l
            even_hkl =  (h % 2 == 0) & (k % 2 == 0) & (l % 2 == 0)
            odd_hkl = (h % 2 != 0) & (k % 2 != 0) & (l % 2 != 0)
            hkl_4N = sum_hkl % 4 == 0
            hkl_4Np1 = (sum_hkl % 4 == 1) | (sum_hkl % 4 == 3)
            allowed = (even_hkl & hkl_4N) | (odd_hkl & hkl_4Np1)
        elif space_group == 229: # body centered cubic
            # Sum of h,k,l must be even
            allowed = (h + k + l) % 2 == 0
        elif space_group == 167: # e.g. sapphire
            # General rhombohedral centering condition: -h + k + l = 3n , space group 167 (sapphire)
            i = -h - k  # for completeness, if needed for hkil checks
            # 1) General condition: hkil: -h + k + l = 3n
            cond_hkil = (-h + k + l) % 3 == 0
            # 2) hki0: -h + k = 3n
            cond_hki0 = ((-h + k) % 3 == 0) & (l == 0)
            # 3) hh(-2h)l: l = 3n
            cond_hh2hl = (h == k) & (i == -2*h) & (l % 3 == 0)
            # 4) h-h0l: h + l = 3n, l = 2n
            cond_hh0l = (h == -k) & (i == 0) & ((h + l) % 3 == 0) & (l % 2 == 0)
            # 5) 000l: l = 6n
            cond_000l = (h == 0) & (k == 0) & (l % 6 == 0)
            # 6) h-h00: h = 3n
            cond_hh00 = (h == -k) & (l == 0) & (h % 3 == 0)
            # Combine all conditions (logical OR)
            allowed = cond_hkil | cond_hki0 | cond_hh2hl | cond_hh0l | cond_000l | cond_hh00
        elif space_group == 139: # e.g. LSCO
            # According to http://img.chem.ucl.ac.uk/sgp/LARGE/139az2.htm
            # 1) General condition: hkl: h + k + l = 2n (usually sufficient)
            cond_hkl = (h + k + l) % 2 == 0
            # 2) 0kl: k + l = 2n
            cond_0kl = (h == 0) & ((k + l) % 2 == 0)
            # 3) h0l: h + l = 2n
            cond_h0l = (k == 0) & ((h + l) % 2 == 0)
            # 4) hk0: h + k = 2n
            cond_hk0 = (l == 0) & ((h + k) % 2 == 0)
            # 5) hhl: l = 2n
            cond_hhl = (h == k) & (l % 2 == 0)
            # 6) h00: h = 2n
            cond_h00 = (k == 0) & (l == 0) & (h % 2 == 0)
            # 7) 0k0: k = 2n
            cond_0k0 = (h == 0) & (l == 0) & (k % 2 == 0)
            # 8) 00l: l = 2n
            cond_00l = (h == 0) & (k == 0) & (l % 2 == 0)
            # Combine all conditions (logical OR)
            allowed = cond_hkl | cond_0kl | cond_h0l | cond_hk0 | cond_hhl | cond_h00 | cond_0k0 | cond_00l
        elif space_group == 12: # monoclinic
            # C-centered lattice: h + k must be even
            allowed = ((h + k) % 2 == 0)
        else:
            raise Exception(f"Unkown space group '{space_group}'.")
        
        return allowed[...,np.newaxis]

    def _spot_yz_from_hkl(self, hkl):
        """
        Calculates the (y,z) coordinates for a given set of (h,k,l) planes.

        Parameters
        ----------
        hkl : 2D-array[float]
            (h,k,l) planes

        Returns
        -------
        y : 1D-array[float]
            y-coordinates corresponding to the given (h,k,l) planes
        z : 1D-array[float]
            z-coordinates corresponding to the given (h,k,l) planes
        theta : 1D-array[float]
            theta angles corresponding to the given (h,k,l) planes
        """
        theta, phi = self._plane_angles_from_hkl(hkl)
        y, z = self._spot_yz_from_plane_angles(theta, phi)

        return y, z, theta

    def _spot_yz_from_plane_angles(self, theta, phi):
        """
        Calculates the (y,z) coordinates for given plane angles and Laue film distance.

        Parameters
        ----------
        theta : 1D-array[float]
            Theta angles
        phi : 1D-array[float]
            Phi angles

        Returns
        -------
        y : 1D-array[float]
            y-coordinates of the Laue spots
        z : 1D-array[float]
            z-coordinates of the Laue spots
        """
        y = self.d_film*np.tan(2*theta)*np.cos(phi) + self.beam_yz[0]
        z = self.d_film*np.tan(2*theta)*np.sin(phi) + self.beam_yz[1]

        ids = np.where(theta > np.pi/2)
        y[ids] = np.nan
        z[ids] = np.nan
        
        return y, z

    @staticmethod
    def _plane_angles_from_hkl(hkl):
        """
        Calculates the plane angles (theta, phi) for a given set of (h,k,l) planes.

        Parameters
        ----------
        hkl : 2D-array[float]
            (h,k,l) planes

        Returns
        -------
        theta : 1D-array[float]
            Theta angles
        phi : 1D-array[float]
            Phi angles
        """
        theta = np.arctan2(np.sqrt(hkl[:,1]**2 + hkl[:,2]**2), hkl[:,0])
        phi = np.arctan2(hkl[:,2], hkl[:,1])

        return theta, phi

class StereographicProjection:
    """
    Stereographic projection of the Laue pattern.
    """
    def __init__(self, crystal_structure, hkl_targets, basis_star, render_shape=(256,256)):
        """
        Parameters
        ----------
        crystal_structure : CrystalStructure
            Crystal structure
        hkl_targets : list[str]
            List of (h,k,l) targets
        render_shape : tuple[int]
            Shape of the rendered Laue pattern
        """
        self.crystal_structure = crystal_structure
        self.hkl_targets = list(hkl_targets)
        self.basis_star = basis_star # matrix for mapping to cartesian (h,k,l) coordinates
        self.render_shape = render_shape

        # Calculate hkl targets
        self._hkl_dict = {}
        for hs_label, hs_value in self.crystal_structure.symmetries.items():
            hkls = hs_value['hkl']
            hkls /= np.linalg.norm(hkls, axis=1, keepdims=True) # normalize vectors
            cart = (self.basis_star @ hkls.T).T # apply reciprocal basis transformation to get cartesian coordinates
            cart = np.where(np.abs(cart) < 1e-6, 0, cart) # set small values to zero
            cart /= np.linalg.norm(cart, axis=1, keepdims=True) # normalize vectors
            self._hkl_dict[hs_label] = cart

        # Initialize strike and dip lists
        self.reset_trajectory()

        # Set desired steregraphic projection center and calculate the steregraphic rotation matrix
        center_axis = np.array([0.0, 0.0, 1.0])
        vertical_axis = np.array([0.0, 0.0, 1.0])
        angle = np.arccos(np.clip(np.dot(center_axis, vertical_axis), -1.0, 1.0))
        axis = np.cross(center_axis, vertical_axis)

        if np.linalg.norm(axis) < 1e-8:
            self.R_stereo = np.eye(3)
        else:
            self.R_stereo = rodrigues(axis=axis/np.linalg.norm(axis), angle=angle)

        # Remaining class attributes
        self.strike_sign = -1
        self.dip_sign = -1
        self.line_ofs = int(0.1*max(self.render_shape))
        self._fontfamily = 'Courier New' if os.name == 'nt' else 'DejaVu Sans Mono' # former: Windows, latter: Unix
        self._fontsize = 15 if os.name == 'nt' else 13.5 # 'Noto Mono' roughly 10% smaller

    def reset_trajectory(self):
        """
        Resets the coordinates of the trajectory.
        """
        self.s_angles = []
        self.d_angles = []

    def project(self, env, savedir=None, savename_prefix=''):
        """
        Calculates and plots the stereographic projection for the current environment state.

        Parameters
        ----------
        env : LaueEnv
            Laue environment instance
        savedir : str
            (Optional) Directory to save the figure(s) (if None: plt.show() will be used)
        savename_prefix : str
            (Optional) prefix for the savenames

        Returns
        -------
        fig : Figure
            Rendered figure with stereographic projection and Laue pattern
        """
        # Render the observation
        obs = env._simulate_observation(shape=self.render_shape)
        
        # Extract environment states
        hkl_ax = env.sim.x_axis

        if env.done:
            flag = "SUCCESS"
        elif env.out_of_range:
            flag = "OUT OF RANGE"
        elif env.out_of_actions:
            flag = "OUT OF ACTIONS"
        else:
            flag = False

        step = env.step_nr

        # Prepare title
        ang_str = self.fmt_tuple(np.rad2deg(env.state), width=5, prec=1)
        chkl_str = self.fmt_tuple(hkl_ax, width=5, prec=2) # NOTE: these are the "cartesian" (h,k,l) values
        _line1 = f"Step: {step:03d}"
        _line2 = f"Angles: {ang_str} deg" 
        _line3 = f"cHKL: {chkl_str} r.l.u." # NOTE: these are the "cartesian" (h,k,l) values
        _line4 = f"Distance (rotation): {np.rad2deg(env.currentdist):4.1f} ({env.currentrot:5.1f}) deg"

        # Plotting of stereographic projection
        fig = plt.figure(figsize=(9,6))
        fig.suptitle(f"{_line1}\n{_line2}\n{_line3}\n{_line4}", y=0.95, fontsize=self._fontsize, linespacing=self._fontsize/10, fontfamily=self._fontfamily)
        ax_s = fig.add_subplot(1,2,1,projection='equal_angle_stereonet', rotation=0)
        # ax_s = fig.add_subplot(1,2,1,projection='equal_area_stereonet', rotation=0)
        ax_i = fig.add_subplot(1,2,2)
        ax_s.set_azimuth_ticklabels([])
        ax_s.grid(color='lightgray')

        for spine in ax_s.spines.values():
            spine.set_edgecolor('lightgray')

        # Plot the high-symmetry points
        min_dist = np.inf
        for hs_label, hkls in self._hkl_dict.items():
            if hs_label in self.hkl_targets:
                dists = np.sqrt(np.sum((hkls - hkl_ax)**2, axis=1))
                ct = np.argmin(dists)
                if dists[ct] < min_dist:
                    closest_targets = hkls
                    min_dist = dists[ct]

            for hkl in hkls:
                strike, dip = mplstereonet.vector2pole(*(self.R_stereo @ hkl))
                strike = (strike - 90) % 360 # adjust the strike to re-orient the pole to match the Laue pattern convention
                lon, lat = mplstereonet.pole(self.strike_sign*strike, self.dip_sign*dip)

                if hs_label in self.hkl_targets:
                    ax_s.pole(self.strike_sign*strike, self.dip_sign*dip, markersize=6, color='gray', marker='o')
                else:
                    ax_s.pole(self.strike_sign*strike, self.dip_sign*dip, markersize=6, markeredgecolor='gray', markerfacecolor='none', marker='o')
                
                ax_s.annotate(
                    f"{hs_label}", ha='center', va='bottom', color='gray',
                    xy=(lon, lat), xytext=(0, 5), textcoords='offset points',
                    fontsize=8
                )

        # Highlight the closest high-symmetry target
        closest_targets = (self.basis_star.T @ closest_targets.T).T # convert back
        for cts in closest_targets:
            strike, dip = mplstereonet.vector2pole(*(self.R_stereo @ cts))
            strike = (strike - 90) % 360 # adjust the strike to re-orient the pole to match the Laue pattern convention
            ax_s.pole(self.strike_sign*strike, self.dip_sign*dip, markersize=8, color='blue', marker='o')

        # Plot the current state (negative strike and dip!)
        s_angle, d_angle = mplstereonet.vector2pole(*(self.R_stereo @ hkl_ax))
        s_angle = (s_angle - 90) % 360 # adjust the strike to re-orient the pole to match the Laue pattern convention
        self.s_angles.append(self.strike_sign*s_angle[0])
        self.d_angles.append(self.dip_sign*d_angle[0])

        # Plot the Laue pattern
        ax_i.imshow(obs, vmin=0., vmax=1., cmap=plt.cm.gray)

        # Plotting the final trajectories
        # ax_s.pole(self.s_angles, self.d_angles, linestyle='--', color='r', marker='none', linewidth=1, alpha=0.5)
        ax_s.pole(self.s_angles[:-1], self.d_angles[:-1], linestyle='', marker='.', markerfacecolor='r', markeredgecolor='none', markersize=8)
        ax_s.pole(self.s_angles[-1], self.d_angles[-1], marker='x', color='r', markersize=8, markeredgewidth=2)

        ax_i.vlines(
            x=[self.render_shape[1]/2,self.render_shape[1]/2], 
            ymin=self.line_ofs, ymax=self.render_shape[0]-self.line_ofs, 
            colors='red', linewidth=1, alpha=0.25
        )
        ax_i.hlines(
            y=[self.render_shape[0]/2,self.render_shape[0]/2], 
            xmin=self.line_ofs, xmax=self.render_shape[1]-self.line_ofs, 
            colors='red', linewidth=1, alpha=0.25
        )

        circle = plt.Circle(
            xy=(self.render_shape[0]/2,self.render_shape[1]/2), 
            radius=env._proj_target_radius, 
            color='red', fill=False, linewidth=2, alpha=0.25)
        
        ax_i.add_patch(circle)

        if flag:
            color = 'yellow' if flag == "SUCCESS" else 'red'
            ax_i.text(0.5, 0.925, flag, transform=ax_i.transAxes, ha='center', va='bottom', fontsize=12, color=color)

        ax_i.set(xticks=[], yticks=[], xlim=[0,self.render_shape[1]-1], ylim=[self.render_shape[0]-1,0])
        
        if savedir is not None:
            fig.savefig(os.path.join(savedir, f"{savename_prefix}{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.png"), dpi=200, bbox_inches='tight', pad_inches=0.1)
            plt.close()

        return fig
    
    @staticmethod
    def fmt_tuple(tup, width, prec):
        return f"({','.join(f'{val:{width}.{prec}f}' for val in tup)})"

if __name__ == "__main__":
    
    # Testing the LaueSimulator
    sim = LaueSimulator(
        space_groups=[221],
        space_groups_weights=[1],
        d_film=40,
        lat_constants=[3.905,3.905,3.905,90,90,90],
        y_lim=[-50,50],
        z_lim=[-50,50],
        beam_yz=[0,0],
        energy_beam=50000,
        max_plane_index=20
    )

    # Define the (h,k,l) crystal axis and a reference beam axis
    crystal_axis = np.array([0.0, 0.0, 1.0])
    beam_axis = np.array([1.0, 0.0, 0.0])

    # Run the Laue spot simulation
    sim.set_crystal_axis(crystal_axis, from_hkl=True)
    sim.set_z_axis(beam_axis, from_hkl=True)
    sim.run_simulation()
    
    # Generate the discretized Laue pattern
    observation = sim.generate_pattern(
        shape=(1284,1284), 
        max_spots=200, 
        remove_fraction=0.0,
        spot_shift_sigma=0.0,
        spot_radius=10
    )

    # Plot the Laue pattern
    plt.figure(figsize=(5,5))
    ax = plt.subplot(1,1,1)
    ax.imshow(observation, cmap=plt.cm.gray)
    ax.set(xticks=[], yticks=[], xlim=[0,observation.shape[1]-1], ylim=[observation.shape[0]-1,0])

    plt.tight_layout()
    plt.show()
