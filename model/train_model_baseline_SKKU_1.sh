python train_multi_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_multi_hrv_osw_75 --class_type quad_multi_hrv_osw_75 --dataset rri_hrv_data_osw_75
python train_multi_rri.py --gpu_id 0 --aug_mode adasyn --max_epoch 500 --logdir quad_multi_rri_osw_75 --class_type quad_multi_rri_osw_75 --dataset rri_hrv_data_osw_75
python train_multi_rri_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_multi_rri_hrv_osw_75 --class_type quad_multi_rri_hrv_osw_75 --dataset rri_hrv_data_osw_75

python train_multi_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_multi_hrv_osw_150 --class_type quad_multi_hrv_osw_150 --dataset rri_hrv_data_osw_150
python train_multi_rri.py --gpu_id 0 --aug_mode adasyn --max_epoch 500 --logdir quad_multi_rri_osw_150 --class_type quad_multi_rri_osw_150 --dataset rri_hrv_data_osw_150
python train_multi_rri_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_multi_rri_hrv_osw_150 --class_type quad_multi_rri_hrv_osw_150 --dataset rri_hrv_data_osw_150

python train_multi_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_multi_hrv_osw_225 --class_type quad_multi_hrv_osw_225 --dataset rri_hrv_data_osw_225
python train_multi_rri.py --gpu_id 0 --aug_mode adasyn --max_epoch 500 --logdir quad_multi_rri_osw_225 --class_type quad_multi_rri_osw_225 --dataset rri_hrv_data_osw_225
python train_multi_rri_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_multi_rri_hrv_osw_225 --class_type quad_multi_rri_hrv_osw_225 --dataset rri_hrv_data_osw_225

python train_multi_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_multi_hrv_osw_300 --class_type quad_multi_hrv_osw_300 --dataset rri_hrv_data_osw_300
python train_multi_rri.py --gpu_id 0 --aug_mode adasyn --max_epoch 500 --logdir quad_multi_rri_osw_300 --class_type quad_multi_rri_osw_300 --dataset rri_hrv_data_osw_300
python train_multi_rri_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_multi_rri_hrv_osw_300 --class_type quad_multi_rri_hrv_osw_300 --dataset rri_hrv_data_osw_300