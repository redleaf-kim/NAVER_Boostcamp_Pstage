from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from load_data import *
import pandas as pd
import torch
import random
import numpy as np
import argparse
from tqdm import tqdm
from enriching_model import BertForSequenceClassification, XLMRobertaForSequenceClassification


def pred_answrs(models_info, test_dl, device, TOK_NAME):
	output_pred = []

	for info in models_info:
		print(f"{info} Model Inference.")
		m_pred = []
		if TOK_NAME.startswith("bert"):
			m = BertForSequenceClassification.from_pretrained(info)
		elif TOK_NAME.startswith("xlm"):
			m = XLMRobertaForSequenceClassification.from_pretrained(info)
		m.to(device)

		for i, data in tqdm(enumerate(test_dl)):
			with torch.no_grad():
				data = tuple(t.to(device) for t in data)
				data = {'input_ids': data[0],
				        'attention_mask': data[1],
				        'token_type_ids': data[2],
				        'e1_mask': data[4],
				        'e2_mask': data[5],
				        }

				m.eval()
				logits = m(**data)[0].detach().cpu().numpy()
				m_pred.extend(logits)
		del m
		m_pred = np.array(m_pred)
		output_pred.append(m_pred)

	print("Collect all results...")
	output_pred = np.array(output_pred)
	output_pred = np.sum(output_pred, axis=0)
	output_pred = np.argmax(output_pred, axis=1)
	return output_pred.reshape(-1, 1)


def load_test_dataset(dataset_dir, tokenizer, xlm=False, max_len=None):
	test_dataset = load_data2(dataset_dir, '../data/label_type.pkl')
	test_label = test_dataset['label'].values
	# tokenizing dataset
	features = tokenized_dataset2(test_dataset, tokenizer, xlm=xlm, max_length=max_len)
	return features, test_label


def seed_everything(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def main(args):
	seed_everything()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	model_info = [
		'../results/Enriching_kfold_XLM-large-200_v3/0_checkpoint-best_acc',
		'../results/Enriching_kfold_XLM-large-200_v3/1_checkpoint-best_acc',
		'../results/Enriching_kfold_XLM-large-200_v3/2_checkpoint-best_acc',
		'../results/Enriching_kfold_XLM-large-200_v3/3_checkpoint-best_acc',
		'../results/Enriching_kfold_XLM-large-200_v3/4_checkpoint-best_acc',

		'../results/Enriching_kfold_XLM-large-150_v2/0_checkpoint-best_acc',
		'../results/Enriching_kfold_XLM-large-150_v2/1_checkpoint-best_acc',
		'../results/Enriching_kfold_XLM-large-150_v2/2_checkpoint-best_acc',
		'../results/Enriching_kfold_XLM-large-150_v2/3_checkpoint-best_acc',
	]

	TOK_NAME = 'xlm-roberta-large'
	tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

	# load test datset
	test_dataset_dir = "../data/test/test.tsv"
	xlm = True if TOK_NAME.startswith("xlm") else False
	test_features, test_label = load_test_dataset(test_dataset_dir, tokenizer, xlm=xlm, max_len=args.max_len)
	all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
	all_e1_mask = torch.tensor([f.e1_mask for f in test_features], dtype=torch.long)  # add e1 mask
	all_e2_mask = torch.tensor([f.e2_mask for f in test_features], dtype=torch.long)  # add e2 mask
	all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
	test_ds = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_e1_mask, all_e2_mask)
	test_dl = torch.utils.data.DataLoader(
		test_ds,
		batch_size=4,
		shuffle=False,
		num_workers=3)

	# predict answer
	print("Start inference.")
	pred_answer = pred_answrs(model_info, test_dl, device, TOK_NAME)

	# make csv file with predicted answer
	# 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
	output = pd.DataFrame(pred_answer, columns=['pred'])
	output.to_csv(f'../prediction/[{args.postfix}]pred_answer.csv', index=False)
	print("Done.")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--postfix', type=str, default='v1')
	parser.add_argument('--max_len', type=int, default=100)
	args = parser.parse_args()
	print(args)

	main(args)