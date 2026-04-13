#!/usr/bin/env python3
import os
import json
import numpy as np

from PIL import Image
from datetime import datetime

from scipy.ndimage import gaussian_filter, median_filter
from scipy.spatial import cKDTree
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import blob_log, match_template
from skimage.draw import disk

class JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to consider more special types.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, type):
            return str(obj)
        else:
            return str(obj)

def get_timestamp():
    """
    Returns the current date and time.
    """
    return datetime.now().strftime('%Y/%m/%d %H:%M:%S') 

def rot_matrix_x(angle):
    """
    Rotation matrix around the x-axis.

    Parameters
    ----------
    angle : float
        Angle of rotation (in radians)

    Returns
    -------
    2D-array[float]
        Rotation matrix
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)], 
        [0, np.sin(angle), np.cos(angle)]
    ])

def rot_matrix_y(angle):
    """
    Rotation matrix around the y-axis.

    Parameters
    ----------
    angle : float
        Angle of rotation (in radians)

    Returns
    -------
    2D-array[float]
        Rotation matrix
    """
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

def rot_matrix_z(angle):
    """
    Rotation matrix around the z-axis.

    Parameters
    ----------
    angle : float
        Angle of rotation (in radians)

    Returns
    -------
    2D-array[float]
        Rotation matrix
    """
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

def rodrigues(axis, angle):
    """
    Creating a rotation matrix around an arbitrary axis (Rodrigues formula).

    Parameters
    ----------
    axis : 1D-array[float]
        Axis of rotation
    angle : float
        Angle of rotation (in radians)

    Returns
    -------
    R : 2D-array[float]
        Rotation matrix
    """

    if np.all(axis==0):
        return np.eye(3)
    
    axis = np.array(axis, dtype=np.float32)
    axis /= np.linalg.norm(axis)
    ux, uy, uz = axis

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    # Compute the outer product of axis vector with itself
    outer = np.outer(axis, axis)
    
    # Create the skew-symmetric matrix for the axis
    K = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])

    # Rodrigues' rotation formula
    R = np.eye(3) * cos_theta + (1 - cos_theta) * outer + sin_theta * K
    
    return R

def sample_laue_parameters(laue_pars_range):
    """
    Sample Laue simulation parameters.

    Parameters
    ----------
    laue_pars_range : dict
        Dictionary containing fixed Laue parameters or ranges of parameters

    Returns
    -------
    laue_pars : dict
        Sampled Laue parameters
    """
    # Sample space group
    id_sg = np.random.randint(low=0, high=len(laue_pars_range['space_groups']))
    space_groups = laue_pars_range['space_groups'][id_sg]
    space_groups_weights = laue_pars_range['space_groups_weights'][id_sg]

    # Sample lattice constants based on crystal structure
    lat_constants = laue_pars_range['crystal_structure'].sample_lattice_parameters(
        min_values=laue_pars_range['lat_constant_range_min'],
        max_values=laue_pars_range['lat_constant_range_max'],
    )

    # Sample remaining parameters
    d_film = np.random.uniform(
        low=min(laue_pars_range['film_range']), 
        high=max(laue_pars_range['film_range'])
    )
    max_planes = np.random.randint(
        low=min(laue_pars_range['max_planes_range']), 
        high=max(laue_pars_range['max_planes_range'])
    )
    max_spots = np.random.choice(
        a=np.arange(min(laue_pars_range['max_spots_range']), max(laue_pars_range['max_spots_range'])),
        p=laue_pars_range['max_spots_weights']
    )

    # Create final dictionary with Laue parameters
    laue_pars = {
        'space_groups': space_groups if isinstance(space_groups, list) else [space_groups],
        'space_groups_weights': space_groups_weights if isinstance(space_groups_weights, list) else [space_groups_weights],
        'd_film': d_film,
        'lat_constants': lat_constants,
        'max_planes': max_planes,
        'max_spots': max_spots,
        'crystal_structure': laue_pars_range['crystal_structure'],
        'y_lim': laue_pars_range['y_lim'],
        'z_lim': laue_pars_range['z_lim'],
        'beam_yz': laue_pars_range['beam_yz'],
        'energy_beam': laue_pars_range['energy_beam'],
        'r_camera': laue_pars_range['r_camera']
    }

    return laue_pars

def max_spots_sample_weights(min_value, max_value, weights_type):
    """
    Calculates max. spots sample weights for a given range of values (min/max) based on a given weights type.

    Parameters
    ----------
    min_value : float
        Minimum value of max. spots
    max_value : float
        Maximum value of max. spots
    weights_type : str
        Weights type

    Returns
    -------
    1D-array[float]
        Weights (sum up to 1)
    """
    ms_range = np.arange(min_value, max_value)

    if weights_type == "constant":
        weights = np.ones_like(ms_range)
    elif weights_type == "log":
        weights = np.log(ms_range) / ms_range
    else:
        raise Exception(f"Unknown max. spots. sampling weight '{weights_type}'. Implemented are 'constant', 'log'.")

    return weights / np.sum(weights)

def create_filtered_image(image, gauss_sigma=25, gauss_trunc=3, median_size=5):
    """
    Applies a Gaussian and Median filter to the input image.

    Parameters
    ----------
    image : 2D array[float]
        Input image (raw)
    gauss_sigma : float
        Width of the Gaussian kernel
    gauss_trunc : float
        Truncate the kernel at this many standard deviations
    median_size : float
        Size of the median filter
    
    Returns
    -------
    2D array[float]
        Filtered image
    """
    sigma = min(image.shape)/gauss_sigma
    imagef = image / gaussian_filter(input=image, sigma=sigma, truncate=gauss_trunc)
    return median_filter(input=np.nan_to_num(imagef), size=median_size)

def create_binary_image(image, thres=0.01):
    """
    Creates a binary image using a certain threshold value.

    Parameters
    ----------
    image : 2D array[float]
        Input image (filtered)
    thres : float
        Threshold for creating the binary image.

    Returns
    -------
    imageb : 2D array[float]
        Binary image
    """
    if thres > 1 or thres < 0:
        raise ValueError(f"Threshold needs to be between 0 and 1.")
    max_vals = int(thres*image.size)
    indices = (-image.ravel()).argsort()[:max_vals]
    r, c = np.unravel_index(indices, shape=image.shape)
    imageb = np.zeros_like(image)
    imageb[r,c] = 1
    return imageb

def line_detection(image, n_lines, n_angles=720, min_dist=50, min_angle=20, com=False):
    """
    Find lines in a Laue diffraction pattern using Hough transform.

    Hough transform: r = x * cos(theta) + y * sin(theta)

    Parameters
    ----------
    image : 2D array[float]
        Image (ideally binary)
    n_lines : int
        Maximal number of lines to detect
    n_angles : int
        Number of angles to use for Hough transform (more = finer search)
    min_dist : int
        Hough transform hyperparameter regarding distance r
    min_angle : int
        Hough transform hyperparameter regarding angle theta
    com : bool
        Whether to calculate the pattern center as the center of mass of all line crossings
    
    Returns
    -------
    angle_main : float
        Angle between the x-axis and the most intense line
    eff_center : tuple[float]
        Center of the diffraction pattern
    projections : list[float]
        Projections of the image center onto the detected lines
    slopes : list[float]
        Slopes of the detected lines
    """
    def project_point(point, line_point, line_dir):
        v = line_dir / np.linalg.norm(line_dir)
        return line_point + (np.dot(point - line_point, v)) * v

    def intersect_lines(p1, d1, p2, d2):
        A = np.column_stack((d1, -d2))
        t, _ = np.linalg.solve(A, p2 - p1)
        return p1 + t * d1

    center = np.float32(np.array(image.shape)/2)
    angles = np.linspace(-np.pi/2, np.pi/2, n_angles)
    hspace, theta, dist = hough_line(image, theta=angles)
    _, theta, dist = hough_line_peaks(hspace, theta, dist, min_distance=min_dist, min_angle=min_angle, num_peaks=n_lines)

    angle_main = np.rad2deg(np.amin(np.pi/2 - theta)) # choose angle of the most intense line

    projections = np.zeros((theta.size, 2))
    slopes = np.zeros((theta.size,))
    lines = []

    for i, (angle, r) in enumerate(zip(theta, dist)):
        origin = r * np.array([np.cos(angle), np.sin(angle)])
        dir_vec = np.array([np.cos(angle+np.pi/2), np.sin(angle+np.pi/2)])
        projections[i] = project_point(center, origin, dir_vec)
        slopes[i] = np.tan(angle+np.pi/2)
        lines.append((origin, dir_vec))

    # Calculate the center of the diffraction pattern
    if theta.size < 2:
        print(f"WARNING: less than two lines detected. Center cannot be determined.")
        eff_center = [0,0]
    else:
        if com:
            crossings = []
            for i in range(len(lines)):
                p1, d1 = lines[i]
                for j in range(i+1, len(lines)):
                    p2, d2 = lines[j]
                    crossings.append(intersect_lines(p1, d1, p2, d2))

                eff_center = np.mean(crossings, axis=0)
        else:
            # Calculate the center as the crossing of the two most intense lines
            (p1, d1), (p2, d2) = lines[0], lines[1]
            eff_center = intersect_lines(p1, d1, p2, d2)

    return angle_main, eff_center, projections, slopes

def load_image(filename):
    """
    Loading a (Laue) image.

    Parameters
    ----------
    filename : str
        Name of the Laue file (.png or .tif)
    
    Returns
    -------
    img : 2D-array[float] or None
        Image
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' does not exist.")
    
    if not os.path.splitext(filename)[-1] in ['.png', '.tif']:
        raise Exception(f"Only 'png' or 'tif' files are supported.")

    img = Image.open(filename)
    
    if filename.endswith("png"):
        img = np.asarray(img.convert("L")).astype(np.float32)
    elif filename.endswith("tif"):
        img = np.asarray(img, dtype=np.int32).astype(np.float32)
    
    return img

def crop_or_pad_image(image, crop_size):
    """
    Crops or pads an image with zeros.

    Parameters
    ----------
    image : 2D-array[float]
        Image array to be cropped or padded
    crop_size : tuple[int]
        Final output shape

    Returns
    -------
    image_cropped : 2D-array[float]
        Cropped or padded image array
    """
    H, W = image.shape
    
    target_H, target_W = crop_size
    
    image_cropped = np.zeros((target_H, target_W), dtype=image.dtype)

    # Compute cropping indices
    start_H = max((H - target_H) // 2, 0)
    end_H   = start_H + min(H, target_H)

    start_W = max((W - target_W) // 2, 0)
    end_W   = start_W + min(W, target_W)

    # Compute placement indices in output
    out_start_H = max((target_H - H) // 2, 0)
    out_end_H   = out_start_H + min(H, target_H)

    out_start_W = max((target_W - W) // 2, 0)
    out_end_W   = out_start_W + min(W, target_W)

    # Crop or pad
    image_cropped[out_start_H:out_end_H, out_start_W:out_end_W] = image[start_H:end_H, start_W:end_W]

    return image_cropped

def make_gaussian_template(radius, pad_factor=3):
    """
    2D Gaussian template.

    Parameters
    ----------
    radius : float
        Radius of the Gaussian
    pad_factor : int
        Padding

    Returns
    -------
    tpl : 2D-array[float]
        Template
    """
    size = int(max(3, radius * pad_factor))
    if size % 2 == 0:
        size += 1
    x = np.arange(size) - size//2
    xx, yy = np.meshgrid(x, x)
    tpl = np.exp(-(xx**2 + yy**2) / (2 * (radius**2)))
    tpl = (tpl - tpl.mean()) / (tpl.std() + 1e-10)
    return tpl

def preprocess_binary(
        image, 
        crop_size, 
        input_size, 
        radius_camera,
        median_filter_size, 
        gaussian_filter_size, 
        lower_percentile=1, 
        upper_percentile=99, 
        min_sigma=5.0,
        max_sigma=15.0,
        num_sigma=3,
        min_spot_distance=0.02,
        threshold=0.4,
        overlap=0.5,
        use_ncc=False,
        num_ncc_radii=5,
        constant_scale=1,
    ):
    """
    Converting a Laue pattern into a (downsampled) binary version without background.

    Parameters
    ----------
    image : 2D-array[float]
        Original Laue pattern (with or without flatfield correction)
    crop_size : tuple[int]
        Crop size
    input_size : tuple[int]
        Downsampled size of the agent input
    radius_camera : float
        Radius of camera hole (fraction of image length)
    median_filter_size : int
        Median filter size
    gaussian_filter_size : float
        Gaussian filter size (sigma)
    lower_percentile : float
        Lower percentile
    upper_percentile : float
        Upper precentile
    min_sigma : float
        Minimum blob sigma value of the blobs
    max_sigma : float
        Maximum blob sigma value of the blobs
    num_sigma : int
        Number of sigma values between min and max sigma values
    min_spot_distance : float
        Minimum spot separation distance
    threshold : float
        Absolute threshold for blob search
    overlap : float
        Overlap fraction of blobs to eliminate overlapping blobs
    use_ncc : bool
        Whether to apply normalized cross correlation and Gaussian template matching
    num_ncc_radii : int
        Number of spot radii (Gaussian standard deviation) for multiscale NCC
    constant_scale : float
        Factor to scale the downsampled agent input image pixel values
        
    Returns
    -------
    image_binary : 2D-array[float]
        Binary image containing spots
    image_proc : 2D-array[float]
        Downsampled binary image
    """
    # Crop the image
    image = crop_or_pad_image(image, crop_size=crop_size)
    orig_size = image.shape

    # Apply a median and gaussian filter
    image = median_filter(image, size=int(max(1, median_filter_size)))
    image = gaussian_filter(image, sigma=0.5)

    # Threshold image
    lower = np.percentile(image, max(lower_percentile, 0.0))
    upper = np.percentile(image, min(upper_percentile, 100.0))
    image = np.clip(image, a_min=lower, a_max=upper)

    if use_ncc:
        # allocate arrays for responses
        radii = np.linspace(min_sigma, max_sigma, num_ncc_radii, endpoint=True)
        ncc_stack = np.zeros((len(radii), *image.shape), dtype=np.float32)

        # 2) compute NCC for each template
        for i, r in enumerate(radii):
            tpl = make_gaussian_template(radius=r)
            # match_template returns normalized cross-correlation
            resp = match_template(image, tpl, pad_input=True)
            ncc_stack[i] = resp.astype(np.float32)

        # 4) optional smoothing of ncc_max to reduce tiny peaks
        image = np.max(ncc_stack, axis=0)

    # Spot identification
    blobs = blob_log(
        image,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold_rel=threshold,
        overlap=overlap
    )

    # Create binary array
    image_binary = np.zeros_like(image, dtype=np.float32)

    # Filter spots by some minimum distance
    min_dist = int(min(min_spot_distance * np.array(orig_size))) # minimum separation distance of peaks (0.035 tweaked for images of shape 1284x1284 --> 84x84)
    selected = []
    selected_coords = []
    tree = None

    for idx in range(len(blobs)):
        point = (blobs[idx,0], blobs[idx,1])
        # if we've got a tree, query it
        if tree is not None:
            if tree.query_ball_point(point, min_dist):
                continue
        selected.append(idx)
        selected_coords.append(point)
        # rebuild tree (cheap for small n_peaks)
        tree = cKDTree(selected_coords)

    ids_spots = np.array(selected)
    blobs = blobs[ids_spots,:] # select blobs

    # Mapping
    for y, x, r in blobs:
        r = 10.0 # NOTE fixed radius
        r_pos, c_pos = disk((y, x), radius=max(1, r), shape=orig_size)
        try:
            image_binary[r_pos, c_pos] = 1.0
        except:
            # spot is on the border, ignore
            pass

    # Create downsampled agent input
    image_proc = np.array(Image.fromarray(image_binary).resize(input_size, resample=Image.Resampling.BOX))
    image_proc = gaussian_filter(image_proc, sigma=gaussian_filter_size)
    
    # Scale image pixel values according to the binary downsampling factor
    image_proc *= constant_scale

    # Remove camera spot
    if radius_camera > 0:
        r_cut, c_cut = disk((input_size[0]//2, input_size[1]//2), int(radius_camera*min(input_size)), shape=input_size)
        image_proc[r_cut, c_cut] = 0.0

    return image_binary, image_proc

if __name__ == "__main__":
    pass
