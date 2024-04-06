
import requests
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))
from classes import Patient
# Orthanc server settings
ORTHANC_URL = "http://localhost:8042"  # Replace with your Orthanc server URL
AUTH = ('username', 'password')  # Replace with your credentials if needed


def get_all_patients_ids():
    return sorted(set(requests.get(f"{ORTHANC_URL}/patients").json()))

def get_patient_info(patient_id):
    return requests.get(f"{ORTHANC_URL}/patients/{patient_id}").json()

def delete_series(series_id):
    series_url = f"{ORTHANC_URL}/series/{series_id}"
    print(f"Deleting series {series_id} with modality 'SEG'")
    delete_response = requests.delete(series_url, auth=AUTH)
    if delete_response.status_code == 200:
        print(f"Series {series_id} deleted successfully")
    else:
        print(f"Failed to delete series {series_id}")


def delete_patient(patient, seg_only=True):
    for series in patient.series:
        modality = series.info.get('MainDicomTags', {}).get('Modality')
        if seg_only:
            if modality == 'SEG':
                delete_series(series.id)
        else:
            patient_short_id = patient.info.get('MainDicomTags', {}).get('PatientID')
            if patient_short_id == '00492':
                delete_series(series.id)
def main():
    # Get all patients
    patients_delete_list = ['00492']
    patients_ids = get_all_patients_ids()
    for patient_id in patients_ids:
        patient = Patient(patient_id)
        patient_short_id = patient.info.get('MainDicomTags', {}).get('PatientID')
        if patient_short_id in patients_delete_list:
            print(patient.id)
            delete_patient(patient)


if __name__ == "__main__":
    main()