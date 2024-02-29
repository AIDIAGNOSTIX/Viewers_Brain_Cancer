import requests
import numpy as np
import pydicom
import os
import shutil
import traceback

import pydicom_seg
import SimpleITK as sitk
import io
import tempfile
import uuid
import nibabel as nib
from params import orthanc_url
import torchio as tio

class Patient:
    def __init__(self, patient_id):
        self.id = patient_id
        self.info = self.get_patient_info()
        self.studies = [Study(s) for s in self.info["Studies"]]
        self.data_dir_path = None
        self.data_path = None
        self.reference_dicom_dir_paths = []
        self.save = True
        self.crop_inverted = False
        self.series = []
        self.get_all_series()
        self.inferred = self.contains_seg()
        self.correct_order = ["flair", "t1ce", "t1", "t2"]
        self.inference_modes = []
        self.segmentation_results = []
        self.dcm_seg_paths = []
        # Create a DICOM SEG template
        self.template = pydicom_seg.template.from_dcmqi_metainfo(
            os.path.join(os.path.dirname(__file__), "metainfo.json")
        )
        self.to_be_deleted = []

        # Create a DICOM SEG writer
        self.writer = pydicom_seg.MultiClassWriter(template=self.template)

        self.reader = sitk.ImageSeriesReader()

    def delete_temp(self):
        for tmp_dir in self.to_be_deleted:
            shutil.rmtree(tmp_dir)

    def contains_seg(self):
        for series in self.series:
            if series.modality == "SEG":
                return True
        return False

    def get_all_series(self):
        for study in self.studies:
            self.series.extend(study.series)

    def load_subject(self):
        """
        Load DICOM images into a TorchIO Subject.

        Parameters:
        - dicom_dir: Path to the directory containing DICOM series.

        Returns:
        - A TorchIO Subject containing all the loaded images.
        """
        subject_dict  = {}
        for series in self.series:
            reference_dicom_dir_path = self.get_dicom_reference(series)
            self.reference_dicom_dir_paths.append(
                (series.mode, reference_dicom_dir_path)
            )
            subject_dict[series.mode] = tio.ScalarImage(reference_dicom_dir_path)
        self.subject = tio.Subject(**subject_dict)
        return self.subject

    def find_max_dimensions(self, subject):
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

    def invert_crop_or_pad(self, subject, original_sizes):
        """
        Invert the CropOrPad transformation for each image in the subject.

        Parameters:
        - subject: A TorchIO Subject that has been transformed and potentially padded.
        - original_sizes: A dictionary mapping image names to their original sizes.

        Returns:
        - A TorchIO Subject with images cropped back to their original sizes.
        """
        self.crop_inverted = True
        for image_name, image in subject.items():
            if image_name in original_sizes:
                original_size = original_sizes[image_name]
                current_size = image.shape[1:]  # Exclude batch dimension

                # Calculate cropping margins if the current size is larger than the original
                if all(cs >= os for cs, os in zip(current_size, original_size)):
                    crop_margins = [(cs - os) // 2 for cs, os in zip(current_size, original_size)]
                    crop_transform = tio.Crop(crop_margins)
                    subject[image_name] = crop_transform(image)
                # If the current size matches the original, no action is needed
        return subject

    def apply_transformations(self, subject, max_dimensions):
        """
        Apply transformations to the subject.

        Parameters:
        - subject: A TorchIO Subject.
        - max_dimensions: A tuple of the maximum dimensions to pad/crop to.

        Returns:
        - A transformed TorchIO Subject.
        """
        def get_original_sizes(subject):
            original_sizes = {}
            for image_name, image in subject.items():
                original_sizes[image_name] = image.shape[1:]  # Exclude batch dimension
            return original_sizes
        self.subject_original_sizes = get_original_sizes(subject)
        reference_image = next(iter(subject.values()), None)
        transform = tio.Compose([
            tio.ToCanonical(),  # Ensure all images are in RAS+ orientation first
            tio.Resample(1, image_interpolation='bspline'),
            tio.Resample(reference_image, image_interpolation='nearest'),
            tio.CropOrPad(max_dimensions),  # Standardize spatial shape
        ])
        return transform(subject)

    def save_transformed_subject(self, transformed_subject):
        """
        Save the transformed subject images to disk.

        Parameters:
        - transformed_subject: A transformed TorchIO Subject.
        """
        # Generate a unique ID
        unique_id = uuid.uuid4()
        # Create a temporary directory
        self.data_dir_path = tempfile.mkdtemp()
        self.to_be_deleted.append(self.data_dir_path)
        # Create a subdirectory named "BraTS2021_<unique_id>"
        file_name = f"BraTS2021_{unique_id}"
        self.data_path = os.path.join(self.data_dir_path, file_name)
        os.makedirs(self.data_path)
        for modality, series in transformed_subject.items():
            series_path = os.path.join(
                self.data_path, f"{file_name}_mode_{modality}.nii.gz" # leave _mode to be able to find it later in inference
            )
            series.save(series_path)

    def download_and_preprocess(self):
        self.subject = self.load_subject()
        max_dimensions = self.find_max_dimensions(self.subject)
        self.subject = self.apply_transformations(self.subject, max_dimensions)
        self.save_transformed_subject(self.subject)

    def get_dicom_reference(self, series):
        series_id = series.id
        output_dir_path = tempfile.mkdtemp()
        # Ensure the output directory exists
        os.makedirs(output_dir_path, exist_ok=True)
        self.to_be_deleted.append(output_dir_path)

        # Get the list of instances in the series
        response = requests.get(f"{orthanc_url}/series/{series_id}/instances")
        if response.status_code != 200:
            print("Error fetching series instances")
            return

        instances = response.json()
        # Download a reference instance
        for i, instance in enumerate(instances):
            instance_id = instance["ID"]
            dicom_response = requests.get(
                f"{orthanc_url}/instances/{instance_id}/file", stream=True
            )

            if dicom_response.status_code == 200:
                # output_file_path = os.path.join(output_dir_path, f"{instance_id}.dcm")
                output_file_path = os.path.join(output_dir_path, f"{i}.dcm")
                with open(output_file_path, "wb") as f:
                    f.write(dicom_response.content)
            else:
                print(f"Error downloading instance {instance_id}")
        series.dir = output_dir_path
        return output_dir_path

    def validate(self):  # do not run if seg already exists for patient (for now)
        if self.inferred:
            self.save = False
            return False
        else:
            self.save = True # set before just making sure
            return True

    def get_patient_info(self):
        return requests.get(f"{orthanc_url}/patients/{self.id}").json()

    def get_inference_modes(self):
        for series in self.series:
            self.inference_modes.append(series.mode)
        self.inference_modes = list(set(self.inference_modes))
        self.inference_modes = sorted(self.inference_modes, key=lambda x: self.correct_order.index(x) if x in self.correct_order else len(self.correct_order))
        return self.inference_modes

    def set_segmentation_result(self, results):
        for result in results:
            self.segmentation_results.append(
                (
                    result[0],
                    # np.transpose(result[1].copy()[0], (2, 0, 1)).astype(np.uint8),
                    result[1].copy()[0].astype(np.uint8),
                )
            )  # (mode, result)
        self.inferred = True
        # implement this where you vote on the final result using all results, and set final segmentation for each existing mode
        self.combine_results()

    def combine_results(self): # TODO: Modify after implementing merging policy
        # implement this where you vote on the final result using all results, but for now just use mode to get each mode's result
        # the result of this function is the final segmentation result for each of the main inference modes (single existing modes)
        # and should be the result of the merging policies to be added after models training
        self.final_segmentation_results = {}
        largest_seg, seg_model_name = self.find_largest_seg(self.segmentation_results)
        print(f'largest segmentation was from model: {seg_model_name}')
        for mode in self.inference_modes:
            self.final_segmentation_results[mode] = largest_seg

    def find_largest_seg(self, segmentation_results):
        # Initialize the largest sum and the corresponding segmentation mask
        largest_sum = -1
        largest_seg = None
        seg_model_name = None
        # Iterate through the segmentation results to find the mask with the largest sum
        for model_name, seg_mask in segmentation_results:
            current_sum = np.sum(seg_mask>0)  # Calculate the sum of the current mask
            if current_sum > largest_sum:
                largest_sum = current_sum
                largest_seg = seg_mask  # Update the mask with the largest sum
                seg_model_name = model_name
        return largest_seg, seg_model_name

    def get_segmentation_result_per_mode(self, mode):
        if self.crop_inverted==False:
            subject_seg_dict  = {}
            for mode in list(self.final_segmentation_results.keys()):
                subject_seg_dict[mode] = tio.ScalarImage(tensor=np.expand_dims(self.final_segmentation_results[mode],0))
            self.subject_seg = tio.Subject(**subject_seg_dict)
            self.subject_seg = self.invert_crop_or_pad(self.subject_seg, self.subject_original_sizes)
            for key in list(self.final_segmentation_results):
                self.final_segmentation_results[key] = np.transpose(self.subject_seg[key].data[0], (2, 1, 0))
        return self.final_segmentation_results[mode]

    # TODO : MAKE THIS SEPERATE FOR EACH MODE'S OUTPUT
    def create_dicom_seg(self, reference_dicom_dir_path):
        mode, reference_dicom_dir_path = reference_dicom_dir_path
        temp_dir = tempfile.mkdtemp()
        self.to_be_deleted.append(temp_dir)

        dcm_files = self.reader.GetGDCMSeriesFileNames(reference_dicom_dir_path)
        source_images = [pydicom.dcmread(x, stop_before_pixels=True) for x in dcm_files]
        if source_images[0].Modality == "SEG":
            return None
        self.reader.SetFileNames(dcm_files)
        image = self.reader.Execute()

        # Create a SimpleITK image from the numpy array
        segmentation_image = sitk.GetImageFromArray(
            self.get_segmentation_result_per_mode(mode)
        )
        segmentation_image.CopyInformation(image)

        # Write the DICOM SEG file
        try:
            dcm_seg = self.writer.write(segmentation_image, source_images)
        except Exception as e:
            print(f'segmentation_image sum: {self.get_segmentation_result_per_mode(mode).sum()}')
            return None

        # Save the DICOM SEG file
        dcm_seg_path = os.path.join(temp_dir, "output_seg.dcm")
        dcm_seg.save_as(dcm_seg_path)
        return dcm_seg_path

    def prepare_dicom_seg(self):
        for i, reference_dicom_dir_path in enumerate(self.reference_dicom_dir_paths):
            dcm_seg_path = self.create_dicom_seg(reference_dicom_dir_path)
            if dcm_seg_path is not None:
                self.dcm_seg_paths.append(dcm_seg_path)

    def upload_dcm_seg_worker(self, dcm_seg_path):
        with open(dcm_seg_path, "rb") as file:
            files = {"file": file}
            response = requests.post(f"{orthanc_url}/instances", files=files)
        return response.status_code == 200

    def upload_dicom_seg(self):
        # Upload segmentation result back to Orthanc
        for i, dcm_seg_path in enumerate(self.dcm_seg_paths):
            if self.upload_dcm_seg_worker(dcm_seg_path):
                print(f"Sucessfully added segmentation series {i}")
            else:
                print("Couldn't add segmentation series {i}")

    def get_series(self):
        return self.series


class Study:
    def __init__(self, study_id):
        self.id = study_id
        self.info = self.get_study_info()
        self.series = [Series(s) for s in self.info["Series"]]
        self.pad_series()

    def get_study_info(self):
        return requests.get(f"{orthanc_url}/studies/{self.id}").json()
    def pad_series(self):
        pass

class Series:
    def __init__(self, series_id):
        self.id = series_id
        self.info = self.get_series_info()
        self.instances = self.info['Instances']
        self.num_instances = len(self.instances)
        # self.data = self.get_series_data()
        self.contributes = True # unused
        self.used = False # unused
        self.modality = self.info["MainDicomTags"]["Modality"]
        self.mode = self.get_familiar_mode(
            self.info["MainDicomTags"]["SeriesDescription"]
        )
        self.dir = None

    def get_series_info(self):
        return requests.get(f"{orthanc_url}/series/{self.id}").json()

    def get_familiar_mode(self, input_mode):
        out_mode = input_mode.lower().replace("w", "")
        return out_mode

    def get_series_data(self):
        response = requests.get(f"{orthanc_url}/series/{self.id}/numpy")
        response.raise_for_status()
        series_data = np.load(io.BytesIO(response.content))
        series_data = np.transpose(series_data, (3, 1, 2, 0)).squeeze(0)
        return series_data


class Instance: # unused, untested yet
    def __init__(self, id):
        self.id = id
        self.info = self.get_instance_info()
        self.data = None

    def get_instance_info(self):
        return requests.get(f"{orthanc_url}/instances/{self.id}").json()

    def get_instance_data(self):
        response = requests.get(f"{orthanc_url}/instances/{self.id}/numpy")
        response.raise_for_status()
        instance_data = np.load(io.BytesIO(response.content))
        instance_data = np.transpose(instance_data, (3, 1, 2, 0)).squeeze(0)
        return instance_data