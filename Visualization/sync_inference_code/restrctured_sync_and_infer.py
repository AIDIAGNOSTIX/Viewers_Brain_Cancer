import pickle
import requests
import numpy as np
import pydicom
import os
import sys
import shutil

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

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))
from CancerDetection.scripts.infer import infer as perform_segmentation
from params import orthanc_url, previous_patients_file, previous_patients_folder, poll_interval_seconds
from classes import Patient, Study, Series, Instance

model = None

def get_models_dict():
    models_dict_path = os.path.abspath(
        os.path.join(__file__, "../", "models_paths.json")
    )
    with open(models_dict_path, "r") as json_file:
        models_dict = json.load(json_file)
    return models_dict


# in the future use inference modes to select most suitable model for inference
def get_models_paths(inference_modes, models_dict):
    models_paths = []
    for mode in inference_modes:
        models_paths.append(([mode], models_dict[mode].strip()))
    if "flair" in inference_modes and "t2" in inference_modes:
        models_paths.append((["flair", "t2"], models_dict["flair_t2"].strip()))
        if "t1" in inference_modes:
            models_paths.append((["flair", "t1", "t2"], models_dict["flair_t1_t2"].strip()))
            if "t1ce" in inference_modes:
                models_paths.append(
                    (["flair", "t1ce", "t1", "t2"], models_dict["flair_t1ce_t1_t2"].strip())
                )
    return models_paths


def save_new_patients(patients):
    global previous_patients_file, previous_patients_folder

    # Ensure the folder for patient objects exists
    if not os.path.exists(previous_patients_folder):
        os.makedirs(previous_patients_folder)

    # Load existing data if the JSON file exists
    if os.path.exists(previous_patients_file):
        with open(previous_patients_file, "r", encoding="utf-8") as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error reading existing patients file: {e}")
                existing_data = []
    else:
        existing_data = []

    # Update the data with new patients
    for patient in patients:
        if not patient.save:
            continue
        # Serialize and save each patient object
        patient_obj_name = f"patient_{patient.id}.pkl"
        patient_obj_full_path = os.path.join(previous_patients_folder, patient_obj_name)
        with open(patient_obj_full_path, "wb") as output_file:
            pickle.dump(patient, output_file)

        # Add patient info to the JSON data
        patient_info = {
            "obj_file": patient_obj_name,
            # Add any other necessary patient attributes here
        }
        existing_data.append(patient_info)
        print('saved patient info')

    # Save the updated JSON data
    with open(previous_patients_file, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, indent=4)
    print('updated saved patient json file')


def load_previous_patients():
    global previous_patients_file, previous_patients_folder
    file_path = previous_patients_file
    if os.path.exists(previous_patients_file):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            # Check if the file is empty
            if not content:
                print("There was no past patients detected in previous patients file.")
                return []
            # Check if the content is valid JSON
            try:
                data = json.loads(content)
                patients = []
                for p in data:
                    patient_obj_name = p["obj_file"]
                    patient_obj_full_path = os.path.join(
                        previous_patients_folder, patient_obj_name
                    )
                    with open(patient_obj_full_path, "rb") as input_file:
                        patients.append(pickle.load(input_file))
                return patients
            except json.JSONDecodeError as e:
                print(f"An error occurred while decoding JSON: {e}")
                print(f"Content: {content}")
                return []
    else:
        return []


def get_dicom_reference(series_id):

    output_dir_path = tempfile.mkdtemp()
    # Ensure the output directory exists
    os.makedirs(output_dir_path, exist_ok=True)

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
    return sorted(set(requests.get(f"{orthanc_url}/patients").json()))


def get_all_patients(current_patients_ids):
    current_patients = []
    for patient_id in current_patients_ids:
        patient = Patient(patient_id)
        current_patients.append(patient)
    return current_patients


def subtract_patients(current_patients, previous_patients):
    # Use a list comprehension to keep elements not in the small_set
    if previous_patients is not None:
        result_list = [
            element for element in current_patients if element not in previous_patients
        ]
        return result_list
    else:
        return current_patients


def mark_for_saving(new_patients):
    for patient in new_patients:
        patient.save = True


def list_new_patients():
    previous_patients = load_previous_patients()
    if len(previous_patients)>0:
        previous_patients = check_for_previous_patients_updates(previous_patients)
    current_patients_ids = get_all_patients_ids()
    current_patients = get_all_patients(current_patients_ids)
    new_patients = subtract_patients(current_patients, previous_patients)
    mark_for_saving(new_patients)
    return new_patients




def process_new_patients():
    global model
    # to_save = []
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
            patient.delete_temp()
            continue
        patient.download()

        # which modes to infer with (check internal variables calculated
        # during validate and see which modes are gonna be used for inference)
        inference_modes = patient.get_inference_modes()
        models_dict = get_models_dict()
        models_paths = get_models_paths(inference_modes, models_dict)
        segmentation_results = []
        for model_path in models_paths:
            segmentation_result, model = perform_segmentation(
                model_path[1], patient.data_dir_path, model_path[0], model=model
            ) # model_path, infer_example_dir, inference_modes (used for this model)
            segmentation_results.append((model_path[0], segmentation_result))
        patient.set_segmentation_result(segmentation_results)
        patient.prepare_dicom_seg()
        patient.upload_dicom_seg()
        save_new_patients([patient])
        patient.delete_temp()
    #     to_save.append(patient)
    # save_new_patients(to_save)


def main():
    # This script could be scheduled to run at regular intervals
    while True:
        try:
            process_new_patients()
            print("Waiting...")
            time.sleep(poll_interval_seconds)
        except Exception as e:
            print(e, f'traceback: ',*sys.exc_info())

if __name__ == "__main__":
    main()
