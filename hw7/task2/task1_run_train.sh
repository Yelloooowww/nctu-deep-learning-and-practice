LOG=$(date +%m%d_%H%M_logs_task1)
echo $LOG
mkdir $LOG
python3 train.py --y_condition --output_dir $LOG \
                  --batch_size 16 \
                  --epochs 1500 \
                  --dataroot "/home/yellow/deep-learning-and-practice/hw7/dataset/task_1/" \
                  --dataset "task1" \
                  --classifier_weight "/home/yellow/deep-learning-and-practice/hw7/classifier_weight.pth" \
                  --flow_coupling "additive" \
                  --K 8 \
                  --L 3 \
                  # --saved_model "/home/yellow/Glow-PyTorch/0619_0211_logs_task1/glow_checkpoint_144064.pt" \
                  # --lr 5e-7\
                  # --y_weight 0.5
