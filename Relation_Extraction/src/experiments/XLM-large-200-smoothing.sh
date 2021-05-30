python3 enriching_kfold.py --epochs 10 --n_splits 5 --data_type 'original' --batch_size 32 \
                               --scheduler_type 'cosine' --version 'Enriching_kfold_XLM-large-200_v3' \
                               --warmup_steps 300 --smoothing 0.5 --cls_weight 0 --weight_decay 0.0001 --hidden_dropout_prob 0.4 \
                               --model_name 'xlm-roberta-large' --fold_s 0 --fold_e 1 --max_len 200


python3 enriching_kfold.py --epochs 10 --n_splits 5 --data_type 'original' --batch_size 32 \
                               --scheduler_type 'cosine' --version 'Enriching_kfold_XLM-large-200_v3' \
                               --smoothing 0.5 --cls_weight 0 --weight_decay 0.0001 --hidden_dropout_prob 0.4 --warmup_steps 300 \
                               --model_name 'xlm-roberta-large' --fold_s 2 --fold_e 3 --max_len 200


python3 enriching_kfold.py --epochs 10 --n_splits 5 --data_type 'original' --batch_size 32 \
                               --scheduler_type 'cosine' --version 'Enriching_kfold_XLM-large-200_v3' \
                               --smoothing 0.5 --cls_weight 0 --weight_decay 0.0001 --hidden_dropout_prob 0.4 --warmup_steps 300 \
                               --model_name 'xlm-roberta-large' --fold_s 4 --fold_e 4 --max_len 200


python3 enriching_kfold.py --epochs 10 --n_splits 5 --data_type 'original' --batch_size 32 \
                               --scheduler_type 'cosine' --version 'Enriching_kfold_XLM-large-150_v2' \
                               --warmup_steps 300 --smoothing 0.5 --cls_weight 0 --weight_decay 0.0001 --hidden_dropout_prob 0.4 \
                               --model_name 'xlm-roberta-large' --fold_s 0 --fold_e 1 --max_len 150


python3 enriching_kfold.py --epochs 10 --n_splits 5 --data_type 'original' --batch_size 32 \
                               --scheduler_type 'cosine' --version 'Enriching_kfold_XLM-large-150_v2' \
                               --smoothing 0.5 --cls_weight 0 --weight_decay 0.0001 --hidden_dropout_prob 0.4 --warmup_steps 300 \
                               --model_name 'xlm-roberta-large' --fold_s 2 --fold_e 3 --max_len 150


python3 enriching_kfold.py --epochs 10 --n_splits 5 --data_type 'original' --batch_size 32 \
                               --scheduler_type 'cosine' --version 'Enriching_kfold_XLM-large-150_v2' \
                               --smoothing 0.5 --cls_weight 0 --weight_decay 0.0001 --hidden_dropout_prob 0.4 --warmup_steps 300 \
                               --model_name 'xlm-roberta-large' --fold_s 4 --fold_e 4 --max_len 150