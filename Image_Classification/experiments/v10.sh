python train.py --n_fold 4 --s_fold 1 --t_epoch 7 --batch_size 32 --age_filter 59 \
                --lr 2e-3 --eta_min 7e-5 --T_max 70 --decay 0 \
                --gridshuffle 1 --weighted_sampler 0 --mixed_precision 0 --uda 0 --cls_weight 1 \
                --pseudo_label 1 --pseudo_label_path "tf_efficientnet_b3_ns_v7_fold1_tta.csv" \
                --crit "arcface" --arcface_crit "focal" --focal_type "bce" --sched_type =="plateau" \
                --model_type "tf_efficientnet_b3_ns" --optim "SGD" --split "gender_ages"  --postfix "v7_fold1"


python train.py --n_fold 4 --s_fold 2 --t_epoch 7 --batch_size 32 --age_filter 59 \
                --lr 2e-3 --eta_min 7e-5 --T_max 70 --decay 0 \
                --gridshuffle 1 --weighted_sampler 0 --mixed_precision 0 --uda 0 --cls_weight 1 \
                --pseudo_label 1 --pseudo_label_path "tf_efficientnet_b3_ns_v7_fold1_tta.csv" \
                --crit "arcface" --arcface_crit "focal" --focal_type "bce" --sched_type =="plateau" \
                --model_type "tf_efficientnet_b3_ns" --optim "SGD" --split "gender_ages"  --postfix "v7_fold2"


python train.py --n_fold 4 --s_fold 3 --t_epoch 7 --batch_size 32 --age_filter 59 \
                --lr 2e-3 --eta_min 7e-5 --T_max 70 --decay 0 \
                --gridshuffle 1 --weighted_sampler 0 --mixed_precision 0 --uda 0 --cls_weight 1 \
                --pseudo_label 1 --pseudo_label_path "tf_efficientnet_b3_ns_v7_fold1_tta.csv" \
                --crit "arcface" --arcface_crit "focal" --focal_type "bce" --sched_type =="plateau" \
                --model_type "tf_efficientnet_b3_ns" --optim "SGD" --split "gender_ages"  --postfix "v7_fold3"


python train.py --n_fold 4 --s_fold 4 --t_epoch 7 --batch_size 32 --age_filter 59 \
                --lr 2e-3 --eta_min 7e-5 --T_max 70 --decay 0 \
                --gridshuffle 1 --weighted_sampler 0 --mixed_precision 0 --uda 0 --cls_weight 1 \
                --pseudo_label 1 --pseudo_label_path "tf_efficientnet_b3_ns_v7_fold1_tta.csv" \
                --crit "arcface" --arcface_crit "focal" --focal_type "bce" --sched_type =="plateau" \
                --model_type "tf_efficientnet_b3_ns" --optim "SGD" --split "gender_ages"  --postfix "v7_fold4"