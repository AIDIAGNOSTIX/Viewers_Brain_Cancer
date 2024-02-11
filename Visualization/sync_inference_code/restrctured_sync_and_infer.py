import requests
import numpy as np
from torch import inf
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
previous_patients_file = os.path.join(os.path.dirname(__file__),"previous_patients_restrctured.json")
previous_patients_folder = os.path.join(os.path.dirname(__file__),"previous_patients")
poll_interval_seconds = 5  # Check every 5 seconds



class Patient():
    def __init__(self, patient_id):
        self.id = patient_id
        self.info = self.get_patient_info()
        self.studies = [Study(s) for s in self.info['Studies']]
        self.data_dir_path = None
        self.data_path = None
        self.refrence_dicom_dir_paths = []
        self.save_if_ok = False
        self.series = []
        self.get_all_series()
        self.infered = self.contains_seg()
        self.inference_modes = []
        self.segmentation_results = []
        self.dcm_seg_paths = []
        # Create a DICOM SEG template
        self.template = pydicom_seg.template.from_dcmqi_metainfo(os.path.join(os.path.dirname(__file__),"metainfo.json"))
        
        # Create a DICOM SEG writer
        self.writer = pydicom_seg.MultiClassWriter(template=self.template)
        
        self.reader = sitk.ImageSeriesReader()
    
    def contains_seg(self):
        for series in self.series:
            if series.modality == 'SEG':
                return True
        return False
    
    def get_all_series(self):
        for study in self.studies:
            self.series.extend(study.series)
    
    def download(self):
        # Generate a unique ID
        unique_id = uuid.uuid4()
        # Create a temporary directory
        self.data_dir_path = tempfile.mkdtemp()
        # Create a subdirectory named "BraTS2021_<unique_id>"
        file_name = f"BraTS2021_{unique_id}"
        self.data_path = os.path.join(self.data_dir_path, file_name)
        os.makedirs(self.data_path)
        for series in self.series:
            series_path = os.path.join(self.data_path, f"{file_name}_{series.mode}.nii.gz")
            nifti_img = nib.Nifti1Image(series.data, affine=np.eye(4))
            # Save the NIfTI image as a .nii.gz file
            nifti_img.to_filename(series_path)
            refrence_dicom_dir_path = get_dicom_refrence(series.id)
            self.refrence_dicom_dir_paths.append(refrence_dicom_dir_path) # TODO: IN THIS LINE ADD INFO ABOUT REFERENCE MODE (TO BE USED LATER WHEN CREATING SEGEMNTATION FOR SOURCE IMAGES)
    
    def validate(self): # do not run if seg already exists for patient (for now)
        if self.infered:
            return False
        else:
            self.save_if_ok = False
            return True
    
    def get_patient_info(self):
        return requests.get(f"{orthanc_url}/patients/{self.id}").json()
    
    def get_inference_modes(self):
        for series in self.series:
            self.inference_modes.append(series.mode)
        self.inference_modes = list(set(self.inference_modes))
        return self.inference_modes
            
    def set_segmentation_result(self, results):
        for result in results:
            self.segmentation_results.append((result[0], np.transpose(result[1].copy()[0], (2, 0, 1)).astype(np.uint8)))
        self.infered = True
        # self.combine_results() # implement this where you vote on the final result using all results
    
    def create_dicom_seg(self, reference_dicom_dir_path): # TODO : MAKE THIS SEPERATE FOR EACH MODE'S OUTPUT
        temp_dir = tempfile.mkdtemp()
        
        dcm_files = self.reader.GetGDCMSeriesFileNames(reference_dicom_dir_path)
        source_images = [pydicom.dcmread(x, stop_before_pixels=True) for x in dcm_files]
        if source_images[0].Modality == 'SEG':
            return None
        self.reader.SetFileNames(dcm_files)
        image = self.reader.Execute()

        # Create a SimpleITK image from the numpy array
        segmentation_image = sitk.GetImageFromArray(self.segmentation_result)
        segmentation_image.CopyInformation(image)

        # Write the DICOM SEG file
        dcm_seg = self.writer.write(segmentation_image, source_images)
        
        # Save the DICOM SEG file
        dcm_seg_path = os.path.join(temp_dir, 'output_seg.dcm')
        dcm_seg.save_as(dcm_seg_path)
        return dcm_seg_path
        
    def prepare_dicom_seg(self):
        for i, refrence_dicom_dir_path in enumerate(self.refrence_dicom_dir_paths):
            dcm_seg_path = self.create_dicom_seg(refrence_dicom_dir_path)
            if dcm_seg_path is not None:
                self.dcm_seg_paths.append(dcm_seg_path)
    
    def upload_dcm_seg_worker(self, dcm_seg_path):
        with open(dcm_seg_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(f"{orthanc_url}/instances", files=files)
        return response.status_code == 200
    
    def upload_dicom_seg(self):
        # Upload segmentation result back to Orthanc
        for i, dcm_seg_path in enumerate(self.dcm_seg_paths):
            if self.upload_dcm_seg_worker(dcm_seg_path):
                print(f'Sucessfully added segmentation series {i}')
            else:
                print("Couldn't add segmentation series {i}")
                
    def get_series(self):
        return self.series
    
class Study():
    def __init__(self, study_id):
        self.id = study_id
        self.info = self.get_study_info()
        self.series = [Series(s) for s in self.info['Series']]
    def get_study_info(self):
        return requests.get(f"{orthanc_url}/studies/{self.id}").json()


class Series():
    def __init__(self, series_id):
        self.id = series_id
        self.info = self.get_series_info()
        self.instances = None
        self.data = self.get_series_data() 
        self.contributes = True
        self.used = False
        self.modality = self.info['MainDicomTags']['Modality']
        self.mode = self.get_familiar_mode(self.info['MainDicomTags']['SeriesDescription'])
        
    def get_series_info(self):
        return requests.get(f"{orthanc_url}/series/{self.id}").json()
    
    def get_familiar_mode(self, input_mode):
        out_mode = input_mode.lower().replace('w', '')
        return out_mode
    
    def get_series_data(self):
        response = requests.get(f"{orthanc_url}/series/{self.id}/numpy")
        response.raise_for_status()
        series_data = np.load(io.BytesIO(response.content))
        series_data = np.transpose(series_data,(3,1,2,0)).squeeze(0)
        return series_data

class Instance():
    def __init__(self):
        self.id = None
        self.info = None
        self.instances = None
        self.data = None

def get_models_dict():
    models_dict_path = os.path.abspath(os.path.join(__file__,'../','models_paths.json'))
    with open(models_dict_path, 'r') as json_file:
        models_dict = json.load(json_file)
    return models_dict

def get_models_paths(inference_modes, models_dict): # in the future use inference modes to select most suitable model for inference
    models_paths = []
    for mode in inference_modes:
        models_paths.append((mode, models_dict[mode].strip()))
    if 'flair' in inference_modes and 't2' in inference_modes:
        models_paths.append(("flair_t2", models_dict['flair_t2'].strip()))
        if 't1' in inference_modes:
            models_paths.append(("flair_t1_t2", models_dict['flair_t1_t2'].strip()))
            if 't1ce' in inference_modes:
                models_paths.append(("flair_t1ce_t1_t2", models_dict['flair_t1ce_t1_t2'].strip()))
    return models_paths

def save_new_patients(patients):
    global previous_patients_file, previous_patients_folder
    
    # Ensure the folder for patient objects exists
    if not os.path.exists(previous_patients_folder):
        os.makedirs(previous_patients_folder)

    # Load existing data if the JSON file exists
    if os.path.exists(previous_patients_file):
        with open(previous_patients_file, 'r', encoding='utf-8') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error reading existing patients file: {e}")
                existing_data = []
    else:
        existing_data = []

    # Update the data with new patients
    for patient in patients:
        if not patient.save_if_ok:
            continue
        # Serialize and save each patient object
        patient_obj_name = f"patient_{patient.id}.pkl"
        patient_obj_full_path = os.path.join(previous_patients_folder, patient_obj_name)
        with open(patient_obj_full_path, 'wb') as output_file:
            pickle.dump(patient, output_file)

        # Add patient info to the JSON data
        patient_info = {
            'obj_file': patient_obj_name,
            # Add any other necessary patient attributes here
        }
        existing_data.append(patient_info)

    # Save the updated JSON data
    with open(previous_patients_file, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, indent=4)


def load_previous_patients():
    global previous_patients_file, previous_patients_folder
    file_path = previous_patients_file
    if os.path.exists(previous_patients_file):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Check if the file is empty
            if not content:
                print("There was no past patients detected in previous patients file.")
                return None
            # Check if the content is valid JSON
            try:
                data = json.loads(content)
                patients = []
                for p in data:
                    patient_obj_name = p['obj_file']
                    patient_obj_full_path = os.path.join(previous_patients_folder, patient_obj_name)
                    with open(patient_obj_full_path, 'rb') as input_file:
                        patients.append(pickle.load(input_file))
                return patients
            except json.JSONDecodeError as e:
                print(f"An error occurred while decoding JSON: {e}")
                print(f"Content: {content}")
                return None
    else:
        return None


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

def check_for_previous_patients_updates(previous_patients):
    # check previous patients for updates,
    # return previous patients that are not updated,
    # treating the updated ones as new
    not_updated_patients = []
    for patient in previous_patients:
        patient_series = patient.get_series()
        if sorted(set(patient.series)) == sorted(set(patient_series)):
            not_updated_patients.append(patient)
    return not_updated_patients

def get_all_patients_ids():
    return set(requests.get(f"{orthanc_url}/patients").json())

def get_all_patients(current_patients_ids):
    current_patients = []
    for patient_id in current_patients_ids:
        patient = Patient(patient_id)
        current_patients.append(patient)
    return current_patients

def subtract_patients(current_patients, previous_patients):
    # Use a list comprehension to keep elements not in the small_set
    if previous_patients is not None:
        result_list = [element for element in current_patients if element not in previous_patients]
        return result_list
    else:
        return current_patients

def mark_for_saving(new_patients):
    for patient in new_patients:
        patient.save_if_ok = True

def list_new_patients():
    previous_patients = load_previous_patients()
    if previous_patients is not None:
        previous_patients = check_for_previous_patients_updates(previous_patients)
    current_patients_ids = get_all_patients_ids()
    current_patients = get_all_patients(current_patients_ids)
    new_patients = subtract_patients(current_patients, previous_patients)
    mark_for_saving(new_patients)
    return new_patients

def process_new_patients():
    to_save = []
    for patient in list_new_patients():
        # 1. check on patients series, if seg series exists removes it from inference series,
        # check on previously contributing inference series if seg exists if same as now
        # don't infer
        # 2. check if all remaining series have same number of instances
        # if not remove once with low number of instances from inference series,
        # this is a bit tricky as many cases could happen:
        # a. all have really small number of instances (below 50 for example) -> cannot infer
        # b. all except one have small number of instances (all below 50 exepct one) -> infer
        # using just this one and copy segmentation mask that intersects with others
        # c. two have small number and two exceeds threshold (example 50), but the big ones
        # have same number of instances -> infer using both and copy segmentation mask that
        # intersects with smaller ones
        # d. two have small number and two exceeds threshold (example 50), but the big
        # ones have different number of instances -> if difference is more than some
        # other threshold (example 50 also) then use bigger one only and copy segmentation
        # mask intersection for rest else ignore bigger one's excess instances and use
        # intersection between those two only for inference and copy ....
        # e. three exceeds threhold and similar -> infer using them then copy ..
        # f. three exceeds threhold and not similar -> get closest pair , see
        # if thier diff is more than threhold if yes use bigger series,
        # if not use thier intersection then compare that to third series 
        # as if they were just two unmatched series
        # g. four exceeds, similar -> ...
        # h. four exceeds, not similar -> find closest pair,
        # exceeds thres -> use biggest one, within thres get intersection between
        # them then compare with closest third series if exceeds use the first pair,
        # if not then get intersection , compare with fourth if exceeds use three series pair,
        # if not intersect and use all four 
        
        infer = patient.validate()
        if not infer:
            continue
        patient.download()
        
        # which modes to infer with (check internal variables calculated 
        # during validate and see which modes are gonna be used for inference)
        inference_modes = patient.get_inference_modes() 
        models_dict = get_models_dict()
        models_paths = get_models_paths(inference_modes, models_dict)
        segmentation_results = []
        for model_path in models_paths:
            segmentation_result = perform_segmentation(model_path[1], patient.data_dir_path, model_path[0])
            segmentation_results.append((model_path[0], segmentation_results))
        patient.set_segmentation_result(segmentation_results)
        patient.prepare_dicom_seg()
        patient.upload_dicom_seg()
        to_save.append(patient)
    save_new_patients(to_save)

def main():
    # This script could be scheduled to run at regular intervals
    while True:
        try:
            process_new_patients()
            print('Waiting...')
            time.sleep(poll_interval_seconds)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
    