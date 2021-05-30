python3 enriching_kfold.py --epochs 10 --n_splits 5 --data_type 'original' --batch_size 32 \
                               --scheduler_type 'cosine' --version 'Enriching_kfold_XLM-large-150' \
                               --cls_weight 0 --weight_decay 0.0001 --model_name 'xlm-roberta-large' --fold_s 0 --fold_e 2 --max_len 150


python3 enriching_kfold.py --epochs 10 --n_splits 5 --data_type 'original' --batch_size 32 \
                               --scheduler_type 'cosine' --version 'Enriching_kfold_XLM-large-150' \
                               --cls_weight 0 --weight_decay 0.0001 --model_name 'xlm-roberta-large' --fold_s 3 --fold_e 4 --max_len 150