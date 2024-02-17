import requests

# Orthanc server settings
ORTHANC_URL = "http://localhost:8042"  # Replace with your Orthanc server URL
AUTH = ('username', 'password')  # Replace with your credentials if needed

def delete_series_if_modality_is_SEG(series_id):
    """Delete the series if its modality is 'SEG'."""
    series_url = f"{ORTHANC_URL}/series/{series_id}"
    response = requests.get(series_url, auth=AUTH)

    if response.status_code == 200:
        series_info = response.json()
        modality = series_info.get('MainDicomTags', {}).get('Modality')

        if modality == 'SEG':
            print(f"Deleting series {series_id} with modality 'SEG'")
            delete_response = requests.delete(series_url, auth=AUTH)
            if delete_response.status_code == 200:
                print(f"Series {series_id} deleted successfully")
            else:
                print(f"Failed to delete series {series_id}")
    else:
        print(f"Failed to get info for series {series_id}")

def main():
    # Get all series
    response = requests.get(f"{ORTHANC_URL}/series", auth=AUTH)
    if response.status_code == 200:
        series_list = response.json()

        for series_id in series_list:
            delete_series_if_modality_is_SEG(series_id)
    else:
        print("Failed to retrieve series list from Orthanc")

if __name__ == "__main__":
    main()
