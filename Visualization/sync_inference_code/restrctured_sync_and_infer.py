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
previous_patients_file = os.path.join(os.path.dirname(__file__),"previous_patients.json")
previous_patients_folder = os.path.join(os.path.dirname(__file__),"previous_patients")
poll_interval_seconds = 5  # Check every 5 seconds



Class Patient():
    def __init__(self, patient_id):
        self.id = patient_id
        self.info = self.get_patient_info()
        self.studies = [Study(s) for s in self.info['Studies']]
        self.infered = False
        self.data_dir_path = None
        self.refrence_dicom_dir_paths = None
    
    def get_patient_info(self):
        return requests.get(f"{orthanc_url}/patients/{self.id}").json()
    
    def get_inference_modes(self):
        pass
    def set_segmentation_result(self):
        pass
    def prepare_dicom_seg(self):
        pass
    def upload_dicom_seg(self):
        pass
    def get_series(self):
        pass
    
Class Study():
    def __init__(self, study_id):
        self.id = study_id
        self.info = self.get_study_info()
        self.series = [Series(s) for s in self.info['Series']]
    def get_study_info(self):
        return requests.get(f"{orthanc_url}/studies/{self.id}").json()


Class Series():
    def __init__(self, series_id):
        self.id = series_id
        self.info = self.get_series_info()
        self.instances = None
        self.data = self.get_series_data()
        self.contributes = True
        self.used = False
        self.mode = self.info['MainDicomTags']['Modality']
    def get_series_info(self):
        return requests.get(f"{orthanc_url}/series/{self.id}").json()
    def get_series_data(self):
        response = requests.get(f"{orthanc_url}/series/{self.id}/numpy")
        response.raise_for_status()
        series_data = np.load(io.BytesIO(response.content))
        return series_data

Class Instance():
    def __init__(self):
        self.id = None
        self.info = None
        self.instances = None
        self.data = None

def get_model_path(inference_modes): # in the future use inference modes to select most suitable model for inference
    model_path_config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "model_path.txt"))
    with open(model_path_config_file, 'r') as f:
        model_path = f.read().strip()
    return model_path

def load_previous_patients():
    global previous_patients_file
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

def list_new_patients():
    previous_patients = load_previous_patients()
    if previous_patients is not None:
        previous_patients = check_for_previous_patients_updates(previous_patients)
    current_patients_ids = get_all_patients_ids()
    current_patients = get_all_patients(current_patients_ids)

def process_new_patients():
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
        
        use = patient.validate() 
        if not use:
            continue
        patient.download()
        inference_modes = patient.get_inference_modes()
        model_path = get_model_path(inference_modes)
        segmentation_result = perform_segmentation(model_path, patient_data_dir_path)
        patient.set_segmentation_result(segmentation_result)
        patient.prepare_dicom_seg()
        patient.upload_dicom_seg()

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
    