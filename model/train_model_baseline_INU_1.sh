python train_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_hrv_osw_75 --class_type quad_hrv_osw_75 --dataset rri_hrv_data_osw_75
python train_rri.py --gpu_id 0 --aug_mode adasyn --max_epoch 500 --logdir quad_rri_osw_75 --class_type quad_rri_osw_75 --dataset rri_hrv_data_osw_75
python train_rri_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_rri_hrv_osw_75 --class_type quad_rri_hrv_osw_75 --dataset rri_hrv_data_osw_75

python train_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_hrv_osw_150 --class_type quad_hrv_osw_150 --dataset rri_hrv_data_osw_150
python train_rri.py --gpu_id 0 --aug_mode adasyn --max_epoch 500 --logdir quad_rri_osw_150 --class_type quad_rri_osw_150 --dataset rri_hrv_data_osw_150
python train_rri_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_rri_hrv_osw_150 --class_type quad_rri_hrv_osw_150 --dataset rri_hrv_data_osw_150

python train_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_hrv_osw_225 --class_type quad_hrv_osw_225 --dataset rri_hrv_data_osw_225
python train_rri.py --gpu_id 0 --aug_mode adasyn --max_epoch 500 --logdir quad_rri_osw_225 --class_type quad_rri_osw_225 --dataset rri_hrv_data_osw_225
python train_rri_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_rri_hrv_osw_225 --class_type quad_rri_hrv_osw_225 --dataset rri_hrv_data_osw_225

python train_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_hrv_300 --class_type quad_hrv_300 --dataset rri_hrv_data_osw_300
python train_rri.py --gpu_id 0 --aug_mode adasyn --max_epoch 500 --logdir quad_rri_300 --class_type quad_rri_300 --dataset rri_hrv_data_osw_300
python train_rri_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_rri_hrv_300 --class_type quad_rri_hrv_300 --dataset rri_hrv_data_osw_300