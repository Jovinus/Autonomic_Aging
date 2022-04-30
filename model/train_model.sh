python train.py --gpu_id 0 --aug_mode adasyn --max_epoch 400 --logdir binary_aging --class_type binary
python train.py --gpu_id 0 --aug_mode randomover --max_epoch 400 --logdir binary_aging --class_type binary
python train.py --gpu_id 0 --aug_mode randomunder --max_epoch 400 --logdir binary_aging --class_type binary
python train.py --gpu_id 0 --aug_mode hybrid --max_epoch 400 --logdir binary_aging --class_type binary