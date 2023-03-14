python train_multi_hrv_binary.py --gpu_id 0 --aug_mode adasyn --max_epoch 200 --logdir hrv --class_type binary_hrv --dataset rri_hrv_data_osw_75
python train_multi_rri_binary.py --gpu_id 0 --aug_mode adasyn --max_epoch 200 --logdir rri --class_type binary_rri --dataset rri_hrv_data_osw_75
python train_multi_rri_hrv_binary.py --gpu_id 0 --aug_mode adasyn --max_epoch 200 --logdir combined --class_type binary_combined --dataset rri_hrv_data_osw_75
