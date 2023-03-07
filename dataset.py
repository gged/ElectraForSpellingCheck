# coding=utf-8
# email: wangzejunscut@126.com

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def search(offsets, index):
    if 2 * index < len(offsets) and offsets[2 * index] == index:
        return index
    left = 0
    right = len(offsets) / 2
    while left < right:
        mid = int((left + right) / 2)
        if offsets[2 * mid] == index:
            return mid
        elif offsets[2 * mid] < index:
            left = mid + 1
        else:
            right = mid
    return -1

def load_dataset(data_path, tokenizer, sep="\t", max_seq_length=512):
    data = []
    with open(data_path, mode="r", encoding="utf-8") as handle:
        for line in tqdm(handle):
            line = line.rstrip().split(sep)
            if len(line) < 1:
                continue
            sentence = line[0]
            encoding = tokenizer.encode(sentence, truncation=True, max_length=max_seq_length)
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            label = [0] * len(input_ids)
            offsets = encoding["offsets"]
            for i in range(1, len(line), 2):
                index = int(line[i])
                pos = search(offsets, index)
                if pos == -1:
                    continue
                label[pos + 1] = 1
            data.append((input_ids, attention_mask, label))
    return data

class SpellCheckDataset(Dataset):
    def __init__(self, data_path, tokenizer, sep="\t", max_seq_length=512):
        super(SpellCheckDataset, self).__init__()
        self.data = load_dataset(data_path, tokenizer, sep, max_seq_length)
    
    def __len__(self):
        """
        Returns the number of samples in dataset.
        """
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Basic function of `SpellCheckDataset` to get sample from dataset with a given 
        index.
        """
        return self.data[index]


class Collate:
    def __init__(self, pad_token_id=0, pad_label_id=0):
        self.pad_token_id = pad_token_id
        self.pad_label_id = pad_label_id
    
    def __call__(self, batch):
        input_ids      = [item[0] for item in batch]
        attention_mask = [item[1] for item in batch]
        labels         = [item[2] for item in batch]
        
        # Padding
        max_seq_len = max(len(_) for _ in input_ids)
        for i in range(len(batch)):
            pad_num = max_seq_len - len(input_ids[i])
            input_ids[i].extend([self.pad_token_id] * pad_num)
            attention_mask[i].extend([0] * pad_num)
            labels[i].extend([self.pad_label_id] * pad_num)
                   
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)
        return input_ids, attention_mask, labels

