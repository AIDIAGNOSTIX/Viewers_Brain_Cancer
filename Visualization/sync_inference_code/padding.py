import os
import torchio as tio

def load_subject(dicom_dir):
    """
    Load DICOM images into a TorchIO Subject.

    Parameters:
    - dicom_dir: Path to the directory containing DICOM series.

    Returns:
    - A TorchIO Subject containing all the loaded images.
    """
    return tio.Subject(
        t1=tio.ScalarImage(os.path.join(dicom_dir, 'T1w')),
        t2=tio.ScalarImage(os.path.join(dicom_dir, 'T2w')),
        t1ce=tio.ScalarImage(os.path.join(dicom_dir, 'T1wCE')),
        flair=tio.ScalarImage(os.path.join(dicom_dir, 'FLAIR')),
    )

def find_max_dimensions(subject):
    """
    Find the maximum dimensions across all images in the subject.

    Parameters:
    - subject: A TorchIO Subject.

    Returns:
    - A tuple containing the maximum height, width, and depth.
    """
    max_H = max_W = max_D = 0
    for key in subject.keys():
        _, H, W, D = subject[key].shape
        max_H = max(max_H, H)
        max_W = max(max_W, W)
        max_D = max(max_D, D)
    return max_H, max_W, max_D

def apply_transformations(subject, max_dimensions):
    """
    Apply transformations to the subject.

    Parameters:
    - subject: A TorchIO Subject.
    - max_dimensions: A tuple of the maximum dimensions to pad/crop to.

    Returns:
    - A transformed TorchIO Subject.
    """
    transform = tio.Compose([
        tio.ToCanonical(),  # Ensure all images are in RAS+ orientation first
        tio.Resample(1, image_interpolation='bspline'),
        tio.CropOrPad(max_dimensions),  # Standardize spatial shape
    ])
    return transform(subject)

def save_transformed_subject(transformed_subject, output_dir):
    """
    Save the transformed subject images to disk.

    Parameters:
    - transformed_subject: A transformed TorchIO Subject.
    - output_dir: The directory to save the images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for modality, image in transformed_subject.items():
        image_path = os.path.join(output_dir, f"{modality}.nii.gz")
        image.save(image_path)

# Example usage
dicom_dir = '/mnt/sda/downloads/rsna-miccai-brain-tumor-radiogenomic-classification/exp/00001'
output_dir = '/mnt/sda/downloads/rsna-miccai-brain-tumor-radiogenomic-classification/exp/subject'

subject = load_subject(dicom_dir)
max_dimensions = find_max_dimensions(subject)
transformed_subject = apply_transformations(subject, max_dimensions)
save_transformed_subject(transformed_subject, output_dir)
