from pydicom import dcmread

# Path to the DICOM file
file_path = '/mnt/sda/freelance_project_girgis/Visualization/sync_inference_code/seg_test'

# Read the DICOM file
ds = dcmread(file_path)

# Print out some basic information about the file
print("Patient ID:", ds.get("PatientID", "Not available"))
print("Study ID:", ds.get("StudyID", "Not available"))
print("Series Number:", ds.get("SeriesNumber", "Not available"))
print("Modality:", ds.get("Modality", "Not available"))
print("Number of Frames:", ds.get("NumberOfFrames", "Not available"))

# If the file contains pixel data, print its shape
if "pixel_array" in dir(ds):
    print("Dimensions:", ds.pixel_array.shape)
else:
    print("Pixel data not available")
