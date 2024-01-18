import requests
import numpy as np
import pydicom
import os, sys

import pydicom_seg
import SimpleITK as sitk
import json
import time
import io
import tempfile
import uuid
import nibabel as nib
import pydicom as dicom

from pydicom.uid import generate_uid
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

sys.path.append(os.path.abspath(os.path.join(__file__,'../../../')))
from CancerDetection.scripts.infer import infer as perform_segmentation

import pickle

# Configuration
orthanc_url = "http://localhost:8042"
previous_patients_file = "/mnt/sda/freelance_project_girgis/Visualization/sync_inference_code/previous_patients.json"
poll_interval_seconds = 5  # Check every 5 seconds

def load_previous_patients():
    if os.path.exists(previous_patients_file):
        with open(previous_patients_file, "r") as file:
            return json.load(file)
    return set()

def save_previous_patients(patients):
    with open(previous_patients_file, "w") as file:
        json.dump(list(patients), file, indent=4)

def get_all_patients():
    return set(requests.get(f"{orthanc_url}/patients").json())

def get_patient_info(patient_id):
    return requests.get(f"{orthanc_url}/patients/{patient_id}").json()

def get_study_info(study):
    return requests.get(f"{orthanc_url}/studies/{study}").json()

def get_series_info(series):
    return requests.get(f"{orthanc_url}/series/{series}").json()

def get_series_data(series):
    response = requests.get(f"{orthanc_url}/series/{series}/numpy")
    response.raise_for_status()
    series_data = np.load(io.BytesIO(response.content))
    return series_data

def get_patients_series(patient_id):
    patient_studies = get_patient_info(patient_id)['Studies']
    series = []
    for study in patient_studies:
        series.extend(list(filter(lambda x : get_series_info(x)['MainDicomTags']['Modality'] != 'SEG' ,study_info['Series'])))
        
def check_for_previous_patients_updates(previous_patients):
    # check previous patients for updates,
    # return previous patients that are not updated,
    # treating the updated ones as new
    not_updated_patients = []
    for patient in previous_patients:
        patient_info = get_patient_info(patient['patient_id'])
        patient_series = []
        for study in patient_info['Studies']:
            study_info = get_study_info(study)
            patient_series.extend(list(filter(lambda x : get_series_info(x)['MainDicomTags']['Modality'] != 'SEG' ,study_info['Series'])))
        if sorted(set(patient_series)) == sorted(set(patient['patient_series'])):
            not_updated_patients.append(patient)
    return not_updated_patients

def get_patients_dict(current_patients_ids):
    patients = []
    for patient_id in current_patients_ids:
        patient_info = get_patient_info(patient_id)
        patient_series = []
        for study in patient_info['Studies']:
            study_info = get_study_info(study)
            patient_series.extend(list(filter(lambda x : get_series_info(x)['MainDicomTags']['Modality'] != 'SEG' ,study_info['Series'])))
        patients.append({"patient_id":patient_id,"patient_series":sorted(set(patient_series))})
    return patients

def subtract_lists(large_list, small_list):
    # Create a copy of the large list to avoid modifying the original
    result_list = large_list.copy()

    # Remove elements that are present in the smaller list
    for element in small_list:
        while element in result_list:
            result_list.remove(element)

    return result_list

def list_new_patients():
    previous_patients = load_previous_patients()
    previous_patients = check_for_previous_patients_updates(previous_patients)
    current_patients_ids = get_all_patients()
    current_patients = get_patients_dict(current_patients_ids)
    new_patients = subtract_lists(current_patients, previous_patients)
    save_previous_patients(current_patients)
    return new_patients

def get_familiar_mode(input_mode):
    out_mode = input_mode.lower().replace('w', '')
    return out_mode

def get_dicom_refrence(series_id):
    
    output_dir_path = tempfile.mkdtemp()
    # Ensure the output directory exists
    os.makedirs(output_dir_path, exist_ok=True)

    # Get the list of instances in the series
    response = requests.get(f"{orthanc_url}/series/{series_id}/instances")
    if response.status_code != 200:
        print("Error fetching series instances")
        return

    instances = response.json()
    # Download a refrence instance
    for i, instance in enumerate(instances):
        instance_id = instance['ID']
        dicom_response = requests.get(f"{orthanc_url}/instances/{instance_id}/file", stream=True)

        if dicom_response.status_code == 200:
            # output_file_path = os.path.join(output_dir_path, f"{instance_id}.dcm")
            output_file_path = os.path.join(output_dir_path, f"{i}.dcm")
            with open(output_file_path, 'wb') as f:
                f.write(dicom_response.content)
        else:
            print(f"Error downloading instance {instance_id}")
    return output_dir_path

def download_patient(patient):
    # Generate a unique ID
    unique_id = uuid.uuid4()
    # Create a temporary directory
    patient_data_dir_path = tempfile.mkdtemp()
    # Create a subdirectory named "BraTS2021_<unique_id>"
    file_name = f"BraTS2021_{unique_id}"
    patient_data_path = os.path.join(patient_data_dir_path, file_name)
    os.makedirs(patient_data_path)
    refrence_dicom_dir_paths = []
    series_infos = []
    
    for series in patient['patient_series']:
        series_info = get_series_info(series)
        series_data = np.transpose(get_series_data(series),(3,1,2,0)).squeeze(0)
        series_mode = get_familiar_mode(series_info['MainDicomTags']['SeriesDescription'])
        series_path = os.path.join(patient_data_path, f"{file_name}_{series_mode}.nii.gz")
        nifti_img = nib.Nifti1Image(series_data,affine=np.eye(4))
        # Save the NIfTI image as a .nii.gz file
        nifti_img.to_filename(series_path)
        refrence_dicom_dir_path = get_dicom_refrence(series)
        
        refrence_dicom_dir_paths.append(refrence_dicom_dir_path)
        series_infos.append(series_info)

    return patient_data_dir_path, refrence_dicom_dir_paths, series_infos


def create_dicom_seg(segmentation_result, reference_dicom_dir_path, series_info, series_num):
    temp_dir = tempfile.mkdtemp()
    segmentation_result_formatted = np.transpose(segmentation_result[0], (2, 0, 1)).astype(np.uint8)
    # segmentation_result_formatted = segmentation_result[0].astype(np.uint8)

    # Create a DICOM SEG template
    template = pydicom_seg.template.from_dcmqi_metainfo('/mnt/sda/freelance_project_girgis/Visualization/sync_inference_code/metainfo.json')

    # Create a DICOM SEG writer
    writer = pydicom_seg.MultiClassWriter(template=template)

    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(reference_dicom_dir_path)
    source_images = [pydicom.dcmread(x, stop_before_pixels=True) for x in dcm_files]
    if source_images[0].Modality == 'SEG':
        return None
    reader.SetFileNames(dcm_files)
    image = reader.Execute()

    # Create a SimpleITK image from the numpy array
    segmentation_image = sitk.GetImageFromArray(segmentation_result_formatted)
    segmentation_image.CopyInformation(image)

    # Write the DICOM SEG file
    dcm_seg = writer.write(segmentation_image, source_images)
    
    # Save the DICOM SEG file
    dcm_seg_path = os.path.join(temp_dir, 'output_seg.dcm')
    dcm_seg.save_as(dcm_seg_path)

    return dcm_seg_path
    
def upload_dicom_to_orthanc(dcm_seg_path):
    with open(dcm_seg_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(f"{orthanc_url}/instances", files=files)
    return response.status_code == 200

def get_model_path():
    model_path_config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "model_path.txt"))
    with open(model_path_config_file, 'r') as f:
        model_path = f.read().strip()
    return model_path

def process_new_patients():
    for patient in list_new_patients():
        patient_data_dir_path, refrence_dicom_dir_paths, series_infos = download_patient(patient)
        model_path = get_model_path()#"/mnt/sda/freelance_project_girgis/runs_best/fold4_f48_ep300_4gpu_dice0_9035/model.pt"        
        segmentation_result = perform_segmentation(model_path, patient_data_dir_path)
        
        dcm_seg_paths = []
        for i, refrence_dicom_dir_path in enumerate(refrence_dicom_dir_paths):
            dcm_seg_path = create_dicom_seg(segmentation_result, refrence_dicom_dir_path, series_infos[i], i)
            if dcm_seg_path is not None:
                dcm_seg_paths.append(dcm_seg_path)
                
        # Upload segmentation result back to Orthanc
        for i, dcm_seg_path in enumerate(dcm_seg_paths):
            if upload_dicom_to_orthanc(dcm_seg_path):
                print(f'Sucessfully added segmentation series {i}')
            else:
                print("Couldn't add segmentation series {i}")

if __name__ == "__main__":
    # This script could be scheduled to run at regular intervals
    while True:
        process_new_patients()
        print('Waiting...')
        time.sleep(poll_interval_seconds)
