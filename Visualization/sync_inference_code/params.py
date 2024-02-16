import os

# Configurations
orthanc_url = "http://localhost:8042"
previous_patients_file = os.path.join(
    os.path.dirname(__file__), "previous_patients_restrctured.json"
)
previous_patients_folder = os.path.join(os.path.dirname(__file__), "previous_patients")
poll_interval_seconds = 5  # Check every 5 seconds