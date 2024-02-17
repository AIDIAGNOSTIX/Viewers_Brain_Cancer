import requests
import os
import json

from params import orthanc_url, previous_patients_file, models_dict_path
from classes import Patient

model = None


def get_models_dict():
    global models_dict_path
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
            models_paths.append(
                (["flair", "t1", "t2"], models_dict["flair_t1_t2"].strip())
            )
            if "t1ce" in inference_modes:
                models_paths.append(
                    (
                        ["flair", "t1ce", "t1", "t2"],
                        models_dict["flair_t1ce_t1_t2"].strip(),
                    )
                )
    return models_paths


def save_new_patients(patients):
    global previous_patients_file

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

        # Add patient info to the JSON data
        patient_info = {
            "patient_id": patient.id,
            "patient_series": [series.id for series in patient.series]
            # Add any other necessary patient attributes here
        }
        existing_data.append(patient_info)
        print("saved patient info")

    # Make sure nothing is saved twice
    seen = set()
    unique_data = [p_info for p_info in existing_data if p_info['patient_id'] not in seen and not seen.add(p_info['patient_id'])]
    # Save the updated JSON data
    with open(previous_patients_file, "w", encoding="utf-8") as file:
        json.dump(unique_data, file, indent=4)
    print("updated saved patient json file")
    # Save the updated JSON data
    with open(previous_patients_file, "w", encoding="utf-8") as file:
        json.dump(unique_data, file, indent=4)
    print("updated saved patient json file")


def load_previous_patients_ids():
    global previous_patients_file
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
                patients_ids = []
                for p in data:
                    patient_id = p["patient_id"]
                    patients_ids.append(patient_id)
                return patients_ids
            except json.JSONDecodeError as e:
                print(f"An error occurred while decoding JSON: {e}")
                print(f"Content: {content}")
                return []
    else:
        return []


def get_all_patients_ids():
    return sorted(set(requests.get(f"{orthanc_url}/patients").json()))


def get_patients(patients_ids):
    patients = []
    for patient_id in patients_ids:
        patient = Patient(patient_id)
        patients.append(patient)
    return patients


def subtract_patients_ids(current_patients_ids, previous_patients_ids):
    # Use a list comprehension to keep elements not in the small_set
    if len(previous_patients_ids)>0:
        result_list = [
            pid for pid in current_patients_ids if pid not in previous_patients_ids
        ]
        return result_list
    else:
        return current_patients_ids


def list_new_patients():
    previous_patients_ids = load_previous_patients_ids()
    current_patients_ids = get_all_patients_ids()
    new_patients_ids = subtract_patients_ids(current_patients_ids, previous_patients_ids)
    new_patients = get_patients(new_patients_ids)
    return new_patients