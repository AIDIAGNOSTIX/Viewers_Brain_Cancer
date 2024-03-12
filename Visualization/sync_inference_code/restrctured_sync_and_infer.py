import os
import sys
import time
import traceback

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))
from CancerDetection.scripts.infer_restructure import CancerSegmentation
from params import poll_interval_seconds
from utils.utils import get_models_dict, get_models_paths, save_new_patients, list_new_patients
from time import time as ctime

def process_new_patients(cancer_segmentation, save_history=True):
    st_tot = ctime()
    for patient in list_new_patients():
        try:
            st = ctime()
            infer = patient.validate()
            if not infer:
                if save_history==True:
                    save_new_patients([patient])
                patient.delete_temp()
                continue
            patient.download_and_preprocess()

            # which modes to infer with (check internal variables calculated
            # during validate and see which modes are gonna be used for inference)
            inference_modes = patient.get_inference_modes()
            models_dict = get_models_dict()
            models_paths = get_models_paths(inference_modes, models_dict)
            segmentation_results = []
            for model_path in models_paths:
                segmentation_result = cancer_segmentation.infer(
                    model_path[1], patient.data_dir_path, model_path[0]
                ) # model_path, infer_example_dir, inference_modes (used for this model)
                segmentation_results.append((model_path[0], segmentation_result, model_path[2]))
            patient.set_segmentation_result(segmentation_results)
            patient.prepare_dicom_seg()
            patient.upload_dicom_seg()
            if save_history==True:
                save_new_patients([patient])
            patient.delete_temp()
            print(f"Time took to process patient: {patient.id} is: {ctime()-st}")
        except Exception as e:
            print(f'An error occurred: {e}')
            traceback.print_exc()
            continue
    print(f"Time took to process all patients: is: {ctime()-st_tot}")


def main():
    # This script could be scheduled to run at regular intervals
    st = ctime()
    cancer_segmentation = CancerSegmentation()
    print(f"Time to initialize model is {ctime()-st}")
    while True:
        try:
            process_new_patients(cancer_segmentation, save_history=True)
            print("Waiting...")
            time.sleep(poll_interval_seconds)
        except Exception as e:
            print(f'An error occurred: {e}')
            traceback.print_exc()

if __name__ == "__main__":
    main()
