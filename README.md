# code

## preprocessing
download code to Lian_project_code/
 
## train U-GAT-IT
python main_pelvic.py --gpu {GPU_ID} --data_dir {DATA_DIR} --checkpoint_dir {CHECKPOINT_DIR}

## test U-GAT-IT
python main_pelvic.py --phase test --gpu {GPU_ID} --data_dir {DATA_DIR} --checkpoint_dir {CHECKPOINT_DIR} --result_dir {OUTPUT_DIR}
