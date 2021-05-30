from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from load_data import *
import pandas as pd
import torch
import random
import numpy as np
import argparse
from enriching_model import BertForSequenceClassification, XLMRobertaForSequenceClassification


def pred_answrs(model, test_dl, device):
	model.eval()
	output_pred = []

	for i, data in enumerate(test_dl):
		with torch.no_grad():
			data = tuple(t.to(device) for t in data)
			data = {'input_ids': data[0],
			          'attention_mask': data[1],
			          'token_type_ids': data[2],
			          'e1_mask': data[4],
			          'e2_mask': data[5],
			          }

			outputs = model(**data)
		logits = outputs[0]
		logits = logits.detach().cpu().numpy()
		result = np.argmax(logits, axis=-1)
		output_pred.append(result)

	return list(np.array(output_pred).reshape(-1))


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


def main(args, version):
	seed_everything()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# load tokenizer
	TOK_NAME = args.model_type
	tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

	# load my model
	if TOK_NAME.startswith("bert"):
		model = BertForSequenceClassification.from_pretrained(args.model_dir)
	elif TOK_NAME.startswith("xlm"):
		model = XLMRobertaForSequenceClassification.from_pretrained(args.model_dir)
	print(model)
	model.parameters
	model.to(device)

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
		batch_size=40,
		shuffle=False,
		num_workers=3)

	# predict answer
	pred_answer = pred_answrs(model, test_dl, device)

	# make csv file with predicted answer
	# 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
	output = pd.DataFrame(pred_answer, columns=['pred'])
	output.to_csv(f'../prediction/[{version}-{args.postfix}]pred_answer.csv', index=False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# model dir
	parser.add_argument('--model_type', default="bert-base-multilingual-cased", type=str)
	parser.add_argument('--model_dir', type=str, default="../results/Enriching-250/checkpoint-best")
	parser.add_argument('--postfix', type=str, default='v1')
	parser.add_argument('--max_len', type=int, default=100)
	args = parser.parse_args()
	print(args)

	version = args.model_dir.split("/")[2]
	main(args, version)