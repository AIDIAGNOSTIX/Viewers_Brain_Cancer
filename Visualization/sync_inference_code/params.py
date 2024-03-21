import os

# Configurations
orthanc_url = "http://localhost:8042"
previous_patients_file = os.path.join(
    os.path.dirname(__file__), "previous_patients_restrctured.json"
)
models_dict_path = os.path.abspath(os.path.join(__file__, "../", "models_paths.json"))

poll_interval_seconds = 5  # Check every 5 seconds
auth = ('username', 'password')  # Replace with your credentials if needed
