import pickle as pickle
import os
from functools import partial
import pandas as pd
import torch

# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
  label = []
  for i in dataset[8]:
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
  return out_dataset


# tsv 파일을 불러옵니다.
def load_data(dataset_dir, label_dir):
  # load label_type, classes
  with open(label_dir, 'rb') as f:
    label_type = pickle.load(f)

  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)

  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type)
  
  return dataset


def load_aug_data(dataset_dir):
    return pd.read_csv(dataset_dir, delimiter='\t')


# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer, max_length=None):
  if max_length is None: max_length = 100

  concat_entity = []
  for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=max_length,
      add_special_tokens=True,
      )
  return tokenized_sentences


def preprocessing_dataset2(dataset, label_type):
    label = []
    for i in dataset[8]:
        if i == 'blind':
            label.append(100)
        else:
            label.append(label_type[i])

    def entity(sent, s1, e1, s2, e2):
        if s1 < s2:
            return sent[:s1] + "$" + sent[s1:e1+1] + "$" + sent[e1+1:s2] + "#" + sent[s2:e2+1] + "#" + sent[e2+1:]
        else:
            return sent[:s2] + "#" + sent[s2:e2+1] + "#" + sent[e2+1:s1] + "$" + sent[s1:e1+1] + "$" + sent[e1+1:]


    out_dataset = pd.DataFrame(
        {'sentence': dataset[1],
         'entity_01': dataset[2], 'entity_01_s': dataset[3],  'entity_01_e': dataset[4],
         'entity_02': dataset[5], 'entity_02_s': dataset[6],  'entity_02_e': dataset[7],
         'label': label, })

    out_dataset['sentence'] = out_dataset.apply(lambda x: entity(x['sentence'],
                                                                 x['entity_01_s'], x['entity_01_e'],
                                                                 x['entity_02_s'], x['entity_02_e']), axis=1)
    return out_dataset


# tsv 파일을 불러옵니다.
def load_data2(dataset_dir, label_dir):
    # load label_type, classes
    with open(label_dir, 'rb') as f:
        label_type = pickle.load(f)

    # load dataset
    if isinstance(dataset_dir, list):
        dataset = None
        for path in dataset_dir:
            tmp_ds = pd.read_csv(path, delimiter='\t', header=None)
            if dataset is None:
                dataset = tmp_ds
            else:
                dataset = pd.concat([dataset, tmp_ds], axis=0)
    else:
        dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)

    # preprecessing dataset
    dataset = preprocessing_dataset2(dataset, label_type)
    return dataset


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 e1_mask, e2_mask,
                 segment_ids,
                 label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask



def tokenized_dataset2(dataset, tokenizer, xlm=False, max_length=None, mask_padding_with_zero=True):
    if max_length is None: max_length = 150

    features = []
    max_pos = 0
    for idx, sample in dataset.iterrows():
        tokens = tokenizer.tokenize(sample['sentence'])
        l = len(tokens)

        if not xlm:
            e1s = tokens.index("#")+2
            e1e = l-tokens[::-1].index("#")

            e2s = tokens.index("$")+2
            e2e = l-tokens[::-1].index("$")
        else:
            try:
                e1s = tokens.index("▁#") + 2
            except Exception as e:
                e1s = tokens.index("#") + 2
            e1e = l - tokens[::-1].index("#")

            try:
                e2s = tokens.index("▁$") + 2
            except Exception as e:
                e2s = tokens.index("$") + 2
            e2e = l - tokens[::-1].index("$")
        max_pos = max(max_pos, e1s, e2s)

        if not xlm:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
        else:
            tokens = ["<s>"] + tokens + ["</s>"]
        segment_ids = [0] * len(tokens)
        segment_ids[0] = 1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([0] * padding_length)


        e1_mask = [0 for _ in range(len(input_mask))]
        e2_mask = [0 for _ in range(len(input_mask))]
        for i in range(e1s, e1e):
            e1_mask[i] = 1
        for i in range(e2s, e2e):
            e2_mask[i] = 1

        if padding_length < 0:
            if xlm:
                input_ids = input_ids[:max_length-1] + tokenizer.convert_tokens_to_ids(["</s>"])
            else:
                input_ids = input_ids[:max_length-1] + tokenizer.convert_tokens_to_ids(["[SEP]"])
            input_mask = input_mask[:max_length-1] + [1 if mask_padding_with_zero else 0]
            segment_ids = segment_ids[:max_length-1] + [0]
            e1_mask = e1_mask[:max_length-1] + [0]
            e2_mask = e2_mask[:max_length-1] + [0]


        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(segment_ids) == max_length


        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          e1_mask=e1_mask,
                          e2_mask=e2_mask,
                          segment_ids=segment_ids,
                          label_id=sample['label']))

    return features