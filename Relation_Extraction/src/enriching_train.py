import os
import wandb
import logging
import argparse
from tqdm import tqdm, trange
import torch.utils.data
from torch.utils.data import TensorDataset
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import transformers
from transformers import AutoTokenizer, AutoConfig
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from load_data import *
from enriching_model import BertForSequenceClassification, XLMRobertaForSequenceClassification
import random
import numpy as np
import torch


# 평가를 위한 metrics function.
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    seed_everything()
    transformers.logging.set_verbosity_info()

    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', default='bert-base-multilingual-cased')
    parser.add_argument('--model_name', default='xlm-roberta-large')
    parser.add_argument('--version', default='v6', type=str)
    parser.add_argument('--valid_ratio', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--l2_reg_lambda', type=float, default=5e-3)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2)
    parser.add_argument('--max_len', type=int, default=150)
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--data_type', type=str, default='original')

    args = parser.parse_args()

    if not os.path.exists(f'../results/{args.version}'):
        os.makedirs(f'../results/{args.version}', exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logging.basicConfig(filename=f'../results/{args.version}.log', filemode='w', format='%(asctime)s ==> %(message)s')
    wandb.init(config=args, project="[Pstage-NLP]", name=args.version, save_code=True)

    # load model and tokenizer
    MODEL_NAME = args.model_name
    if MODEL_NAME == "monologg/kobert":
        from tokenization_kobert import KoBertTokenizer
        tokenizer = KoBertTokenizer.from_pretrained(MODEL_NAME)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if args.data_type == "original":
        train_data_path = "../data/train/train.tsv"
    elif args.data_type == "extra_v1":
        train_data_path = "../data/train/train_and_extra_v1.tsv"
    elif args.data_type == "extra_v2":
        train_data_path = "../data/train/train_and_extra_v2.tsv"
    elif args.data_type == "aug":
        train_data_path = "../data/train/aug_extra_train.tsv"


    # load dataset
    total_dataset = load_data2(train_data_path, "../data/label_type.pkl")
    if args.valid_ratio > 0.0:
        train_dataset, valid_dataset = train_test_split(total_dataset,
                                                        test_size=args.valid_ratio, random_state=42,
                                                        shuffle=True, stratify=total_dataset.label)
        valid_label = valid_dataset['label'].values
        valid_features = tokenized_dataset2(valid_dataset, tokenizer, max_length=args.max_len)
        all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        all_e1_mask = torch.tensor([f.e1_mask for f in valid_features], dtype=torch.long)  # add e1 mask
        all_e2_mask = torch.tensor([f.e2_mask for f in valid_features], dtype=torch.long)  # add e2 mask
        all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_ds = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_e1_mask, all_e2_mask)
        valid_dl = torch.utils.data.DataLoader(all_input_ids, all_input_mask,all_segment_ids, all_label_ids, all_e1_mask, all_e2_mask)
        evaluation_startegy = 'steps'
    else:
        train_dataset = total_dataset




    xlm = True if MODEL_NAME.startswith("xlm") else False
    train_features = tokenized_dataset2(train_dataset, tokenizer, xlm=xlm, max_length=args.max_len)

    # make dataset for pytorch.
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_e1_mask = torch.tensor([f.e1_mask for f in train_features], dtype=torch.long)  # add e1 mask
    all_e2_mask = torch.tensor([f.e2_mask for f in train_features], dtype=torch.long)  # add e2 mask
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_ds = TensorDataset(all_input_ids, all_input_mask,all_segment_ids, all_label_ids, all_e1_mask, all_e2_mask)
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=3)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    val_counts = train_dataset.label.value_counts().sort_index().values
    cls_weight = 1 / np.log1p(val_counts)
    cls_weight = (cls_weight / cls_weight.sum()) * 42
    cls_weight = torch.tensor(cls_weight, dtype=torch.float32).to(device)


    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 42
    model_config.l2_reg_lambda = args.l2_reg_lambda
    model_config.latent_entity_typing = False
    if MODEL_NAME.startswith("bert"):
        model = BertForSequenceClassification(model_config, MODEL_NAME)
    elif MODEL_NAME.startswith("xlm"):
        model = XLMRobertaForSequenceClassification(model_config, MODEL_NAME)
    model.parameters
    model.to(device)



    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters()
                                                if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                                    {'params': [p for n, p in model.named_parameters()
                                                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                                    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_eps)

    num_training_steps = int(len(train_dl)//args.accumulation_steps * args.epochs)
    if args.scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)

    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.epochs)
    logging.info("  Total optimization steps = %d", num_training_steps)

    wandb.watch(model)
    global_step = 0
    tr_loss = 0.0
    model.zero_grad()

    min_loss = float("INF")
    train_iterator = trange(int(args.epochs), desc="Epoch")
    for _ in train_iterator:
        corrects = 0
        total_sample = 0
        epoch_iterator = tqdm(train_dl, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3],
                      'e1_mask': batch[4],
                      'e2_mask': batch[5],
                      'cls_weight': cls_weight
                      }

            outputs = model(**inputs)
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]
            pred = outputs[1]
            _, pred = torch.max(pred, dim=-1)
            corrects += np.sum((pred == batch[3]).detach().cpu().numpy())
            total_sample += batch[0].size(0)
            tr_acc = corrects / total_sample * 100

            if args.accumulation_steps > 1:
                loss = loss / args.accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            if (step + 1) % args.accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                wandb.log({"loss": tr_loss / global_step, "acc": tr_acc})

                if global_step % 100 == 0:
                    logging.info("global_step = %s, average loss = %s", global_step, tr_loss/global_step)

                    if min_loss > tr_loss/global_step:
                        logging.info(f"Loss: {min_loss:.6f} -> {tr_loss/global_step:.6f}")
                        logging.info("save.")
                        min_loss = tr_loss/global_step

                        save_path = os.path.join(f"../results/{args.version}/checkpoint-best")
                        model.save_pretrained(save_path)


if __name__ == '__main__':
    train()