python train_multi_hrv.py --gpu_id 1 --aug_mode adasyn --max_epoch 500 --logdir quad_multi_hrv_no --class_type quad_multi_hrv_no
python train_multi_rri.py --gpu_id 1 --aug_mode naive --max_epoch 500 --logdir quad_multi_rri_no --class_type quad_multi_rri_no
python train_multi_rri_hrv.py --gpu_id 1 --aug_mode adasyn --max_epoch 500 --logdir quad_multi_rri_hrv_no --class_type quad_multi_rri_hrv_no