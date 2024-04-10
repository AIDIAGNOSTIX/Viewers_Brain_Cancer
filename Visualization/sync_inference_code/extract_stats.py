import os
import sys
import time
import traceback

from time import time as ctime
import torch
import torchio as tio
from classes import Patient
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy
from scipy.spatial import ConvexHull, distance_matrix
import scipy
from sklearn.cluster import KMeans

def load_subject(subject_dir):
        """
        Load DICOM images into a TorchIO Subject.

        Parameters:
        - subject_dir: Path to the directory containing.

        Returns:
        - A TorchIO Subject containing all the loaded images.
        """
        subject_dict  = {}
        for modality_file in os.listdir(subject_dir):
            modality = modality_file.split('.nii')[0].split('_')[-1]
            if 'seg' in modality:
                subject_dict[modality] = tio.LabelMap(os.path.join(subject_dir, modality_file))
            else:
                subject_dict[modality] = tio.ScalarImage(os.path.join(subject_dir, modality_file))
        subject = tio.Subject(**subject_dict)
        return subject

def calculate_diameters(segmentation, voxel_spacing_mm, num_clusters=2000):
    # You can change num_clusters where higher number gives more accurate results
    # But it uses more memory and more computations

    # Convert the segmentation tensor to a numpy array and remove the channel dimension
    segmentation_array = segmentation.squeeze(0).numpy()

    # Extract the tumor boundary points
    boundary_points = np.argwhere(segmentation_array > 0)

    # Convert voxel spacing to cm
    voxel_spacing_cm = np.array(voxel_spacing_mm) / 10

    # Reduce the number of boundary points using K-means clustering for the longest diameter calculation
    num_clusters = min(len(boundary_points), num_clusters)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(boundary_points)
    reduced_boundary_points = kmeans.cluster_centers_

    # Calculate the longest diameter in voxel space and then convert to cm
    # using the reduced set of boundary points
    reduced_boundary_points_cm = reduced_boundary_points * voxel_spacing_cm
    dist_matrix = distance_matrix(reduced_boundary_points_cm, reduced_boundary_points_cm)
    longest_diameter_cm = np.max(dist_matrix)

    # Calculate the shortest diameter
    # Here, the full set of boundary points can be used or reduced, depending on memory constraints
    boundary_points_cm = boundary_points * voxel_spacing_cm
    try:
        hull = ConvexHull(boundary_points_cm)
        shortest_diameter_cm = np.max([np.linalg.norm(hull.points[edge[0]] - hull.points[edge[1]]) for edge in hull.simplices])
    except scipy.spatial.qhull.QhullError as e:
        print("Convex hull calculation failed due to coplanar points; consider an alternative method for calculating the shortest diameter.")
        print(e)
        shortest_diameter_cm = None

    return longest_diameter_cm, shortest_diameter_cm

def calculate_RECIST_diameter(segmentation, voxel_spacing_mm, lesion_type='non-nodal', num_clusters=2000):
    """
    Calculates the RECIST diameter of lesions, distinguishing between nodal and non-nodal lesions.

    Parameters:
    - segmentation (numpy.ndarray): The segmentation mask of the lesion.
    - voxel_spacing_mm (list or numpy.ndarray): The spacing of the voxels in mm.
    - lesion_type (str): The type of lesion ('nodal' or 'non-nodal').
    - num_clusters (int): Number of clusters for K-means to reduce boundary points.

    Returns:
    - float: The RECIST diameter in mm.
    """

    segmentation_array = segmentation.squeeze(0).numpy()
    boundary_points = np.argwhere(segmentation_array > 0)
    voxel_spacing_mm = np.array(voxel_spacing_mm)

    # Convert boundary points to physical space in millimeters
    boundary_points_mm = boundary_points * voxel_spacing_mm

    if lesion_type == 'non-nodal':
        num_clusters = min(len(boundary_points_mm), num_clusters)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(boundary_points_mm)
        reduced_points_mm = kmeans.cluster_centers_

        distances = pdist(reduced_points_mm)
        recist_diameter_mm = np.max(distances)

    elif lesion_type == 'nodal':
        try:
            hull = ConvexHull(boundary_points_mm)
            edges = np.array([hull.points[simplex] for simplex in hull.simplices])
            lengths = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1)
            recist_diameter_mm = np.min(lengths)
        except scipy.spatial.qhull.QhullError as e:
            print("Convex hull calculation failed.")
            recist_diameter_mm = None
    else:
        raise ValueError("Lesion type must be either 'nodal' or 'non-nodal'.")

    return recist_diameter_mm

def calculate_mean_intensity(mri_array, mask_array):
    """
    Calculate the mean intensity value of a segmented region in an MRI scan.

    Args:
    mri_array: The MRI image data.
    mask_array: The binary segmentation mask of the region.

    Returns:
    float: The mean intensity value of the region.
    """
    # Apply the mask to the MRI array to isolate the region's voxels
    region_voxels = mri_array[mask_array > 0].numpy().astype(float)

    # Calculate the mean intensity value of the region
    mean_intensity = region_voxels.mean()

    return mean_intensity

def get_tumor_voulme_stats(subject, verbose=False):
    spacing = subject.spacing
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    seg_data = subject['seg'].data
    volume_stats = {}
    """
    1 -> Tumor core
    2 -> Whole tumor
    4 -> Enhancing tumor
    """
    tc = (seg_data == 1).sum()
    wt = (seg_data == 2).sum()
    et = (seg_data == 4).sum()
    total_voxels = tc + wt + et
    volume_stats['num_voxels'] = {
        'tc': tc,
        'wt': wt,
        'et': et,
        'total': total_voxels
        }
    tc_volume = (tc * voxel_volume)/1000 # in cm^3
    wt_volume = (wt * voxel_volume)/1000 # in cm^3
    et_volume = (et * voxel_volume)/1000 # in cm^3
    total_volume = (total_voxels * voxel_volume)/1000 # in cm^3
    volume_stats['tumor_volume'] = {
        'tc': tc_volume,
        'wt': wt_volume,
        'et': et_volume,
        'total': total_volume
    }
    if verbose:
        print(f"Tumor core volume: {tc_volume:.2f} cm^3")
        print(f"Whole tumor volume: {wt_volume:.2f} cm^3")
        print(f"Enhancing tumor volume: {et_volume:.2f} cm^3")
        print(f"Total tumor volume: {total_volume:.2f} cm^3")
        print('-'*50,'\n')
    return volume_stats

def get_tumor_diameter_stats(subject, num_clusters=2000, verbose=False):
    """
    Calculate the longest and shortest tumor diameters.
    """
    # segmentation = (subject['seg'].data>0).squeeze(0).numpy()
    segmentation = subject['seg'].data
    spacing = subject.spacing
    longest_diameter, shortest_diameter = calculate_diameters(segmentation, spacing, num_clusters)
    diameter_stats = {
        'longest_diameter': longest_diameter,
        'shortest_diameter': shortest_diameter
        }
    if verbose:
        print(f'Longest Diameter: {longest_diameter:.2f}, Shortest Diameter: {shortest_diameter:.2f}')
        print('-'*50,'\n')
    return diameter_stats

def get_mean_intensity_stats(subject, verbose=False):
    mean_intensity_stats = {}
    for modality in subject:
        if modality != 'seg':
            mean_intensity = calculate_mean_intensity(subject[modality].data, subject['seg'].data)
            mean_intensity_stats[modality] = mean_intensity
            if verbose:
                print(f'Mean Intensity for {modality}: {mean_intensity:.2f}')
    if verbose:
        print('-'*50,'\n')
    return mean_intensity_stats

def get_tumor_RECIST_diameter_stats(subject, num_clusters=2000, verbose=False):
    """
    Calculate the longest and shortest tumor diameters.
    """
    segmentation = subject['seg'].data
    spacing = subject.spacing
    RECIST_diameter = calculate_RECIST_diameter(segmentation, spacing, lesion_type='non-nodal', num_clusters=num_clusters)
    RECIST_diameter_stats = {
        'RECIST_diameter': RECIST_diameter,
        }
    if verbose:
        print(f'RECIST Diameter: {RECIST_diameter:.2f} mm')
        print('-'*50,'\n')
    return RECIST_diameter_stats

def calculate_doubling_time(V1, V2, T, verbose=False):
    """
    Calculate the doubling time of a tumor given two volume measurements and the time between them.

    Parameters:
    - V1 (float): Volume at the first measurement.
    - V2 (float): Volume at the second measurement.
    - T (float): Time in days between the two measurements.

    Returns:
    - float: The doubling time in days.
    """
    DT = (T * np.log(2)) / (np.log(V2) - np.log(V1))
    if verbose:
        print(f'Doubling Time: {DT:.2f} days')
    return DT

def main():
    data_path = '/home/girgis-kubuntu/Desktop/Work/Freelacing/Cancer_Detection/brats_data'
    subjects = []
    for example_dir in os.listdir(data_path):
        subject = load_subject(os.path.join(data_path, example_dir))
        subjects.append(subject)
    transforms = [
        tio.ToCanonical(),
        tio.RescaleIntensity(out_min_max=(0, 1024))
        ]
    transform = tio.Compose(transforms)
    subjects_dataset = tio.SubjectsDataset(subjects,transform=transform)
    for subject in subjects_dataset:
        volume_stats = get_tumor_voulme_stats(subject, verbose=False)
        diameter_stats = get_tumor_diameter_stats(subject, num_clusters=2000, verbose=False)
        mean_intensity_stats = get_mean_intensity_stats(subject, verbose=False)
        RECIST_diameter_stats = get_tumor_RECIST_diameter_stats(subject, num_clusters=2000, verbose=False)

if __name__ == "__main__":
    main()






