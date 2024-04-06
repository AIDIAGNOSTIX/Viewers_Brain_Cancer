import requests
import numpy as np
import pydicom
import os
import shutil
import traceback
import torch

import pydicom_seg
import SimpleITK as sitk
import io
import tempfile
import uuid
import nibabel as nib
from params import orthanc_url, auth
import torchio as tio
# from utils.upload_to_orthanc import upload_dicom_directory
class Patient:
    def __init__(self, patient_id):
        self.id = patient_id
        self.info = self.get_patient_info()
        self.short_id = self.info.get('MainDicomTags', {}).get('PatientID')
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
    # def delete_patient(self, seg_only=True):
    #     for series in self.series:
    #         modality = series.info.get('MainDicomTags', {}).get('Modality')
    #         if seg_only:
    #             if modality == 'SEG':
    #                 self.delete_series(series.id)
    #         else:
    #             patient_short_id = self.info.get('MainDicomTags', {}).get('PatientID')
    #             if patient_short_id == '00492':
    # #                 self.delete_series(series.id)

    # def delete_series(self, series_id):
    #     series_url = f"{orthanc_url}/series/{series_id}"
    #     print(f"Deleting series {series_id} with modality 'SEG'")
    #     delete_response = requests.delete(series_url, auth=auth)
    #     if delete_response.status_code == 200:
    #         print(f"Series {series_id} deleted successfully")
    #     else:
    #         print(f"Failed to delete series {series_id}")
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
        subject = tio.Subject(**subject_dict)
        return subject

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

    def invert_resample_and_to_canonical(self, subject_seg, subject):

        self.crop_inverted = True
        # Iterate over the images in subject_seg
        original_subject_seg = subject_seg.copy()
        for image_name, image in subject_seg.items():
            if image_name in subject:
                # Get the corresponding reference image from subject
                reference_image = subject[image_name]
                # Create the Resample transformation with the reference image
                resample_1 = tio.Resample(1, image_interpolation='nearest')
                # resample = tio.Resample(target=reference_image, image_interpolation='nearest')
                resample = tio.Resample((reference_image.shape[1:], reference_image.affine), image_interpolation='bspline')

                # Apply the resampling to the image
                original_subject_seg[image_name] = resample(resample_1(image))
            else:
                # Handle the case where the image_name is not in the reference subject
                print(f"Warning: {image_name} not found in reference subject. Skipping.")
        return original_subject_seg

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
        def select_biggest_image(subject):
            # Initialize variables to store the biggest image information
            biggest_image_name = None
            biggest_image_size = 0

            # Iterate through each image in the subject
            for image_name, image in subject.items():
                # Calculate the size of the current image (number of voxels)
                image_size = image.shape[1] * image.shape[2] * image.shape[3]

                # If the current image is bigger than the biggest found so far, update the variables
                if image_size > biggest_image_size:
                    biggest_image_name = image_name
                    biggest_image_size = image_size
            return biggest_image_name

        self.subject_original_sizes = get_original_sizes(subject)
        # reference_image = subject[select_biggest_image(subject)] #next(iter(subject.values()), None)
        self.transform = tio.Compose([
            tio.ToCanonical(),  # Ensure all images are in RAS+ orientation first
            tio.Resample(1, image_interpolation='bspline'),
            # tio.Resample((subject[select_biggest_image(subject)].shape[1:], subject[select_biggest_image(subject)].affine), image_interpolation='bspline'),
            # tio.ToCanonical(),  # Ensure all images are in RAS+ orientation first
            # tio.CropOrPad(max_dimensions),  # Standardize spatial shape
            tio.Resample(select_biggest_image(subject), image_interpolation='nearest'),
            # tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            # tio.RescaleIntensity((-1, 1)),
        ])
        return self.transform(subject)

    def save_transformed_subject(self, transformed_subject):
        """
        Save the transformed subject images to disk as .nii.gz

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

    # def save_preprocessed_images_to_dcm(self, transformed_subject):
    #     """
    #     Save the transformed subject images to disk as .dcm

    #     Parameters:
    #     - transformed_subject: A transformed TorchIO Subject.
    #     """
    #     # Create a temporary directory
    #     output_directory = tempfile.mkdtemp()
    #     self.to_be_deleted.append(output_directory)

    #     for modality, series in transformed_subject.items():
    #         # Assuming the data is 3D, add code here if 4D (e.g., time series)
    #         data = series.data.numpy().squeeze().transpose((2,0,1))  # Remove channels dim, adjust as necessary
    #         # affine = series.affine
    #         for i in range(data.shape[0]):  # Iterate over the slices
    #             slice_data = data[i, :, :]
    #             # Convert your slice to a pydicom dataset (modify this part based on your original DICOM metadata)
    #             series_dir = str(series.path)
    #             try:
    #                 ds = pydicom.dcmread(os.path.join(series_dir,sorted(os.listdir(series_dir))[i]))  # Assuming 'path' exists and points to a DICOM file used as a template
    #             except:
    #                 ds = pydicom.dcmread(os.path.join(series_dir,sorted(os.listdir(series_dir))[-1]))  # Assuming 'path' exists and points to a DICOM file used as a template
    #             ds.PixelData = slice_data.tobytes()
    #             ds.save_as(os.path.join(output_directory, f"{modality}_{i}.dcm"))
    #     return output_directory

    def download_and_preprocess(self):
        self.subject = self.load_subject()
        max_dimensions = self.find_max_dimensions(self.subject)
        self.subject_preprocessed = self.apply_transformations(self.subject, max_dimensions)
        self.save_transformed_subject(self.subject_preprocessed)
        # output_directory = self.save_preprocessed_images_to_dcm(self.subject)
        # self.delete_patient(seg_only=False)
        # upload_dicom_directory(output_directory)
        return 0

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
            # self.save = False
            return False
        else:
            # self.save = True # set before just making sure
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
        self.segmentation_results = results.copy() # (mode, result, dice_avg) for each result in results
        self.inferred = True
        # vote on the final result using all results, and set final segmentation for each existing mode
        self.combine_results()

    def combine_results(self):
        # implement this where you vote on the final result using all results, but for now just use mode to get each mode's result
        # the result of this function is the final segmentation result for each of the main inference modes (single existing modes)
        # and should be the result of the merging policies to be added after models training
        self.final_segmentation_results = {}
        # mode_list, seg_prob_list, dice_avg_list = map(list, zip(*self.segmentation_results))
        # seg_prob_list_tensors = [torch.tensor(item) for item in seg_prob_list if item is not None]
        # seg_probs = torch.stack(seg_prob_list_tensors) if seg_prob_list_tensors else None  # Stack them to create a single tensor of shape [N, *shape_of_each_tensor]

        # cleaned_dice_avg_list = [item if item is not None else 0 for item in dice_avg_list]
        # confidence_scores = torch.nn.functional.softmax(torch.tensor(cleaned_dice_avg_list, dtype=torch.float), dim=0)

        # # Reshape confidence_scores to be able to multiply it with seg_probs
        # confidence_scores = confidence_scores.view(-1, 1, 1, 1, 1)  # Adjust the shape according to the dimension of your seg_probs

        # # Compute the weighted sum of segmentation probabilities
        # final_seg_prob = torch.sum(seg_probs * confidence_scores, dim=0).detach().numpy()
        # final_seg_prob = (final_seg_prob > 0.5).astype(np.int8)
        # # Initialize the output segmentation
        # seg_out = np.zeros((final_seg_prob.shape[1], final_seg_prob.shape[2], final_seg_prob.shape[3]))
        # # Assign labels to the segmentation
        # seg_out[final_seg_prob[1] == 1] = 2  # Whole tumor
        # seg_out[final_seg_prob[2] == 1] = 3  # Enhancing tumor
        # seg_out[final_seg_prob[0] == 1] = 1  # Tumor core

        # for mode in self.inference_modes:
        #     self.final_segmentation_results[mode] = seg_out.astype(np.uint8)

        for current_mode in self.inference_modes:
            # Filter lists based on the current mode
            filtered_seg_prob_list = [seg_prob for mode, seg_prob, _ in self.segmentation_results if current_mode in mode]
            filtered_dice_avg_list = [dice_avg for mode, _, dice_avg in self.segmentation_results if current_mode in mode]

            # Convert to tensors and compute as before
            seg_prob_list_tensors = [torch.tensor(item) for item in filtered_seg_prob_list if item is not None]
            seg_probs = torch.stack(seg_prob_list_tensors) if seg_prob_list_tensors else None

            cleaned_dice_avg_list = [item if item is not None else 0 for item in filtered_dice_avg_list]
            confidence_scores = torch.nn.functional.softmax(torch.tensor(cleaned_dice_avg_list, dtype=torch.float), dim=0)

            if seg_probs is not None:
                # Reshape confidence_scores to be able to multiply it with seg_probs
                confidence_scores = confidence_scores.view(-1, 1, 1, 1, 1)

                final_seg_prob = torch.sum(seg_probs * confidence_scores, dim=0).detach().numpy()
                final_seg_prob = (final_seg_prob > 0.5).astype(np.int8)

                # Initialize the output segmentation
                seg_out = np.zeros((final_seg_prob.shape[1], final_seg_prob.shape[2], final_seg_prob.shape[3]))

                # Assign labels to the segmentation
                seg_out[final_seg_prob[1] == 1] = 2  # Whole tumor
                seg_out[final_seg_prob[2] == 1] = 3  # Enhancing tumor
                seg_out[final_seg_prob[0] == 1] = 1  # Tumor core
                self.final_segmentation_results[current_mode] = seg_out.copy().astype(np.uint8)
    def get_segmentation_result_per_mode(self, mode):
        if self.crop_inverted==False:
            subject_seg_dict  = {}
            for mode in list(self.final_segmentation_results.keys()):
                subject_seg_dict[mode] = tio.LabelMap(tensor=np.expand_dims(self.final_segmentation_results[mode].copy(),0))
            self.subject_seg = tio.Subject(**subject_seg_dict)
            # self.subject_seg.applied_transforms = self.subject_preprocessed.applied_transforms
            # self.subject_seg_original_space = self.subject_seg.apply_inverse_transform()
            self.subject_seg = self.invert_crop_or_pad(self.subject_seg, self.subject_original_sizes)
            # subject_seg_original_space = self.invert_resample_and_to_canonical(self.subject_seg, self.subject)
            # self.subject_seg_original_space = self.invert_crop_or_pad(self.subject_seg_original_space.copy(), self.subject_original_sizes)
            for key in list(self.final_segmentation_results):
                self.final_segmentation_results[key] = np.transpose(self.subject_seg[key].data[0], (2, 1, 0))
                # self.final_segmentation_results[key] = np.transpose(self.subject_seg[key].data[0], (2, 1, 0))
        return self.final_segmentation_results[mode]

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
            print(f'An error occurred: {e}')
            traceback.print_exc()
            print(f'segmentation_image sum: {self.get_segmentation_result_per_mode(mode).sum()} for mode: {mode}')
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