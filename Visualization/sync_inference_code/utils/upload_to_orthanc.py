import os
import requests

# Set the Orthanc server address and port
orthanc_server = 'http://localhost:8042'

# The directory containing the DICOM files organized by patient
dicom_directory = '/mnt/sda/downloads/rsna-miccai-brain-tumor-radiogenomic-classification/test'
# Function to upload a single DICOM file to the Orthanc server
def upload_dicom_file(dicom_file_path):
    with open(dicom_file_path, 'rb') as f:
        files = {'file': (dicom_file_path, f)}
        try:
            response = requests.post(f'{orthanc_server}/instances', files=files)
            response.raise_for_status()
            print(f"Successfully uploaded: {dicom_file_path}")
        except requests.exceptions.HTTPError as err:
            print(f"Failed to upload {dicom_file_path}: {err}")

def upload_dicom_directory(dicom_directory):
    for dirpath, _, filenames in os.walk(dicom_directory):
        for filename in filenames:
            if filename.lower().endswith('.dcm'):
                dicom_file_path = os.path.join(dirpath, filename)
                upload_dicom_file(dicom_file_path)

def main():
    upload_dicom_directory(dicom_directory)

if __name__ == "__main__":
    main()