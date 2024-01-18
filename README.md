
# Guide

## Steps:

1- Getting Orthanc and OHIF up and working
    - Go to `../Visualization/Viewers_Brain_Cancer`
    - (Optional) If you want to make changes to docker compose, dockerfile, ngnix config, ohif config files:
      - Go to recipe dir `../Visualization/Viewers_Brain_Cancer/platform/app/.recipes/OpenResty-Orthanc`
      - You can find intended files on the recipe dir or in the config dir within: `../Visualization/Viewers_Brain_Cancer/platform/app/.recipes/OpenResty-Orthanc/config`
    - `yarn orthanc:up ` to start orthanc
    - `yarn run dev:orthanc` start viewer after launching orthanc server
    - You can now access the Orthanc database on `localhost:8042` and OHIF on `localhost:3000`
2- Getting `sync_and_infer.py` script up and running:
    - Download model weights from the server located at: `/mnt/sda/freelance_project_girgis/runs_best/fold4_f48_ep300_4gpu_dice0_9035/model.pt`
    - After saving the weights on local machine put the complete file path in model path config file: `../Visualization/sync_inference_code/model_path.txt`
    - Run the script `python sync_and_infer.py`
3- Upload examples on Orthanc portal (located on `localhost:8042`)