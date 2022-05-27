python train.py --gpu_id 0 --aug_mode adasyn --max_epoch 400 --logdir tri_aging_10fold --class_type tri_aging_10fold
python train.py --gpu_id 0 --aug_mode randomover --max_epoch 400 --logdir tri_aging_10fold --class_type tri_aging_10fold
python train.py --gpu_id 0 --aug_mode randomunder --max_epoch 400 --logdir tri_aging_10fold --class_type tri_aging_10fold
python train.py --gpu_id 0 --aug_mode hybrid --max_epoch 400 --logdir tri_aging_10fold --class_type tri_aging_10fold