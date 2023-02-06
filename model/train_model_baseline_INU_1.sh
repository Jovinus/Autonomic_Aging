python train_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_hrv_no --class_type quad_hrv_no --dataset rri_hrv_data_osw_75
python train_rri.py --gpu_id 0 --aug_mode adasyn --max_epoch 500 --logdir quad_rri_no --class_type quad_rri_no --dataset rri_hrv_data_osw_75
python train_rri_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_rri_hrv_no --class_type quad_rri_hrv_no --dataset rri_hrv_data_osw_75

python train_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_hrv_no --class_type quad_hrv_no --dataset rri_hrv_data_osw_150
python train_rri.py --gpu_id 0 --aug_mode adasyn --max_epoch 500 --logdir quad_rri_no --class_type quad_rri_no --dataset rri_hrv_data_osw_150
python train_rri_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_rri_hrv_no --class_type quad_rri_hrv_no --dataset rri_hrv_data_osw_150

python train_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_hrv_no --class_type quad_hrv_no --dataset rri_hrv_data_osw_225
python train_rri.py --gpu_id 0 --aug_mode adasyn --max_epoch 500 --logdir quad_rri_no --class_type quad_rri_no --dataset rri_hrv_data_osw_225
python train_rri_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_rri_hrv_no --class_type quad_rri_hrv_no --dataset rri_hrv_data_osw_225

python train_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_hrv_no --class_type quad_hrv_no --dataset rri_hrv_data_osw_300
python train_rri.py --gpu_id 0 --aug_mode adasyn --max_epoch 500 --logdir quad_rri_no --class_type quad_rri_no --dataset rri_hrv_data_osw_300
python train_rri_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_rri_hrv_no --class_type quad_rri_hrv_no --dataset rri_hrv_data_osw_300