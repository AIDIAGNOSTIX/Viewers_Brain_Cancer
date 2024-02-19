import os
import sys
import time
import traceback

sys.path.append(os.path.abspath(os.path.join(__file__, "../../../")))
from CancerDetection.scripts.infer_restructure import CancerSegmentation
from params import poll_interval_seconds
from utils.utils import get_models_dict, get_models_paths, save_new_patients, list_new_patients
from time import time as ctime

def process_new_patients(cancer_segmentation):
    st_tot = ctime()
    for patient in list_new_patients():
        try:
            """
                1. check on patients series, if seg series exists removes it from inference series,
                    check on previously contributing inference series if seg exists if same as now
                    don't infer

                2. check if all remaining series have same number of instances
                    if not remove once with low number of instances from inference series,
                    this is a bit tricky as many cases could happen:
                        a. all have really small number of instances (below 50 for example) -> cannot infer

                        b. all except one have small number of instances (all below 50 exepct one) -> infer
                            using just this one and copy segmentation mask that intersects with others

                        c. two have small number and two exceeds threshold (example 50), but the big ones
                            have same number of instances -> infer using both and copy segmentation mask that
                            intersects with smaller ones

                        d. two have small number and two exceeds threshold (example 50), but the big
                            ones have different number of instances -> if difference is more than some
                            other threshold (example 50 also) then use bigger one only and copy segmentation
                            mask intersection for rest else ignore bigger one's excess instances and use
                            intersection between those two only for inference and copy ....

                        e. three exceeds threshold and similar -> infer using them then copy ..

                        f. three exceeds threshold and not similar -> get closest pair , see
                            if their diff is more than threshold if yes use bigger series,
                            if not use their intersection then compare that to third series
                            as if they were just two unmatched series

                        g. four exceeds, similar -> ...

                        h. four exceeds, not similar -> find closest pair,
                            exceeds threshold -> use biggest one, within threshold get intersection between
                            them then compare with closest third series if exceeds use the first pair,
                            if not then get intersection , compare with fourth if exceeds use three series pair,
                            if not intersect and use all four
            """
            st = ctime()
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
                segmentation_result = cancer_segmentation.infer(
                    model_path[1], patient.data_dir_path, model_path[0]
                ) # model_path, infer_example_dir, inference_modes (used for this model)
                segmentation_results.append((model_path[0], segmentation_result))
            patient.set_segmentation_result(segmentation_results)
            patient.prepare_dicom_seg()
            patient.upload_dicom_seg()
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
    print(f"Time to initialize model with GPU is {ctime()-st}")
    while True:
        try:
            process_new_patients(cancer_segmentation)
            print("Waiting...")
            time.sleep(poll_interval_seconds)
        except Exception as e:
            print(f'An error occurred: {e}')
            traceback.print_exc()

if __name__ == "__main__":
    main()
