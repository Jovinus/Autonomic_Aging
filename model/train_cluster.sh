python train_rri.py --gpu_id 0 --aug_mode adasyn --max_epoch 2 --logdir quad_aging_rri_cosine_loss --class_type quad_aging_rri_cosine_loss
python train_rri.py --gpu_id 0 --aug_mode randomover --max_epoch 500 --logdir quad_aging_rri_cosine_loss --class_type quad_aging_rri_cosine_loss
python train_rri.py --gpu_id 0 --aug_mode hybrid --max_epoch 500 --logdir quad_aging_rri_cosine_loss --class_type quad_aging_rri_cosine_loss
python train_rri.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_aging_rri_cosine_loss --class_type quad_aging_rri_cosine_loss

python train_rri_hrv.py --gpu_id 0 --aug_mode adasyn --max_epoch 500 --logdir quad_aging_rri_hrv_cosine_loss --class_type quad_aging_rri_hrv_cosine_loss
python train_rri_hrv.py --gpu_id 0 --aug_mode randomover --max_epoch 500 --logdir quad_aging_rri_hrv_cosine_loss --class_type quad_aging_rri_hrv_cosine_loss
python train_rri_hrv.py --gpu_id 0 --aug_mode hybrid --max_epoch 500 --logdir quad_aging_rri_hrv_cosine_loss --class_type quad_aging_rri_hrv_cosine_loss
python train_rri_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_aging_rri_hrv_cosine_loss --class_type quad_aging_rri_hrv_cosine_loss

python train_hrv.py --gpu_id 0 --aug_mode adasyn --max_epoch 500 --logdir quad_aging_hrv_cosine_loss --class_type quad_aging_hrv_cosine_loss
python train_hrv.py --gpu_id 0 --aug_mode randomover --max_epoch 500 --logdir quad_aging_hrv_cosine_loss --class_type quad_aging_hrv_cosine_loss
python train_hrv.py --gpu_id 0 --aug_mode hybrid --max_epoch 500 --logdir quad_aging_hrv_cosine_loss --class_type quad_aging_hrv_cosine_loss
python train_hrv.py --gpu_id 0 --aug_mode naive --max_epoch 500 --logdir quad_aging_hrv_cosine_loss --class_type quad_aging_hrv_cosine_loss