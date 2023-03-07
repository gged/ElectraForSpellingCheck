# coding=utf-8
# email: wangzejunscut@126.com

import argparse
import random
import torch
from easytokenizer import AutoTokenizer
from transformers import ElectraForMaskedLM

def load_common_characters(path):
    chars = set()
    with open(path, mode="r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip().split()
            chars.update(line)
    return chars

def load_electra_vocab(path):
    vocab = []
    with open(path, mode="r", encoding="utf-8") as handle:
        for line in handle:
            vocab.append(line.strip())
    return vocab

def isChinese(c):
    cp = ord(c[0])
    if cp >= 0x4E00 and cp <= 0x9FA5:
        return True
    return False

def load_phonetic_set(path, chars):
    phonetics = {}
    with open(path, mode="r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip().split()
            if len(line) < 2:
                continue
            key = line[0]
            value = []
            for c in line[1: ]:
                if c in chars and c not in value and c != key:
                    value.append(c)
            if value:
                phonetics[key] = value
    return phonetics

def load_similar_set(path, chars):
    similars = {}
    with open(path, mode="r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip().split(",")
            if len(line) < 2:
                continue
            key = line[0]
            value = []
            for c in line[1: ]:
                if c in chars and c not in value and c != key:
                    value.append(c)
            if value:
                similars[key] = value
    return similars

def parse_args():
    parser = argparse.ArgumentParser(description="Chinese spelling check preprocess.")
    parser.add_argument("--sentence_file",
                        required=True,
                        type=str,
                        help="The full path of sentences to be processed.")
    parser.add_argument("--common_characters_file",
                        required=True,
                        type=str,
                        help="The full path of 3500 Chinese characters.")
    parser.add_argument("--homophone_file",
                        required=True,
                        type=str,
                        help="The full path of homophone set.")
    parser.add_argument("--near_phonetic_file",
                        required=True,
                        type=str,
                        help="The full path of near-phonetic character set.")
    parser.add_argument("--similar_file_1",
                        required=True,
                        type=str,
                        help="The full path of similar character set from PLOME.")
    parser.add_argument("--similar_file_2",
                        required=True,
                        type=str,
                        help="The full path of similar character set from other sources.")
    parser.add_argument("--vocab_file",
                        required=True,
                        type=str,
                        help="The vocabulary file to be used.")

    parser.add_argument("--mask_ratio",
                        default=0.05,
                        type=float,
                        help="The ratio of characters to be replaced. Default 0.05")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="Random seed. Default 42")
    parser.add_argument("--generator",
                        default="hfl/chinese-electra-180g-base-generator",
                        type=str,
                        help="The chinese electra generator to be used. Default hfl/chinese-electra-180g-base-generator")
    parser.add_argument("--topk",
                        default=30,
                        type=int,
                        help="The number of candidate characters. Default 30")
    parser.add_argument("--do_lower_case",
                        action="store_true",
                        help="Set this flag if you are using an uncased model.")
    args = parser.parse_args()
    return args

def binary_search(offsets, index):
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

def electra_generator_mask(sent_list, index, args, model):
    c = sent_list[index]
    sent_list[index] = "[MASK]"
    sentence = "".join(sent_list)
    sent_list[index] = c
    encoding = args.tokenizer.encode(sentence)
    input_ids = torch.tensor([encoding["input_ids"]], device=args.device)

    # inference
    with torch.no_grad():
        output = model(input_ids)
    
    # binary search
    tensor_index = binary_search(encoding["offsets"], index) + 1
        
    topk_indexes = output[0][0][tensor_index].topk(args.topk, largest=True, sorted=True)[1].tolist()
    start = min(args.topk // 2, 5)
    random_index = random.choice(topk_indexes[start : ])
    repeats = 10
    n = 0
    while args.vocab[random_index] == c or not isChinese(args.vocab[random_index]):
        random_index = random.choice(topk_indexes)
        n += 1
        if n >= repeats:
            return ""
    return args.vocab[random_index]
    
def do_mask(sentence, args, model):
    total_num = len(sentence)
    mask_num = int(total_num * args.mask_ratio)
    sent_list = list(sentence)
    indexes = []
    chars = []
    for _ in range(mask_num):
        index = random.randint(0, total_num - 1)
        c = sentence[index]
        if not isChinese(c) or index in indexes:
            continue
        
        p = random.random()
        # electra generator mask
        if p <= 0.3:
            mask = electra_generator_mask(sent_list, index, args, model)
            if not mask:
                continue
            indexes.append(index)
            chars.append(mask)
        # homophone mask
        elif p > 0.3 and p <= 0.6:
            if c in args.phonetics:
                mask = random.choice(args.phonetics[c])
            else:
                mask = electra_generator_mask(sent_list, index, args, model)
                if not mask:
                    continue
            indexes.append(index)
            chars.append(mask)
        # near-phonetic mask
        elif p > 0.6 and p <= 0.7:
            if c in args.near_phonetics:
                mask = random.choice(args.near_phonetics[c])
            elif c in args.phonetics:
                mask = random.choice(args.phonetics[c])
            else:
                mask = electra_generator_mask(sent_list, index, args, model)
                if not mask:
                    continue
            indexes.append(index)
            chars.append(mask)
        # similar mask 1
        elif p > 0.7 and p <= 0.75:
            if c in args.similars1:
                mask = random.choice(args.similars1[c])
            else:
                mask = electra_generator_mask(sent_list, index, args, model)
                if not mask:
                    continue
            indexes.append(index)
            chars.append(mask)
        # similar mask 2
        elif p > 0.75 and p <= 0.8:
            if c in args.similars2:
                mask = random.choice(args.similars2[c])
            elif c in args.similars1:
                mask = random.choice(args.similars1[c])
            else:
                mask = electra_generator_mask(sent_list, index, args, model)
                if not mask:
                    continue
            indexes.append(index)
            chars.append(mask)
        # random mask
        elif p > 0.8 and p <= 0.9:
            mask = random.choice(args.chars)
            if mask != c:
                indexes.append(index)
                chars.append(mask)
        # keep same
        else: 
            continue
    for index, mask in zip(indexes, chars):
        sent_list[index] = mask
    new_sent = "".join(sent_list)
    for index, mask in zip(indexes, chars):
        new_sent += "\t"
        new_sent += str(index)
        new_sent += "\t"
        new_sent += mask
    print(new_sent)

def process(args, model):
    with open(args.sentence_file, mode="r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip().split("\t", maxsplit=1)
            do_mask(line[0], args, model)
            
if __name__ == "__main__":
    args = parse_args()
    chars = load_common_characters(args.common_characters_file)
    args.chars = list(chars)
    args.phonetics = load_phonetic_set(args.homophone_file, chars)
    args.near_phonetics = load_phonetic_set(args.near_phonetic_file, chars)
    args.similars1 = load_similar_set(args.similar_file_1, chars)
    args.similars2 = load_phonetic_set(args.similar_file_2, chars)
    args.vocab = load_electra_vocab(args.vocab_file)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer(args.vocab_file, do_lower_case=args.do_lower_case)
    model = ElectraForMaskedLM.from_pretrained(args.generator)
    model.to(device)
    model.eval()
    
    args.device = device
    args.tokenizer = tokenizer
    random.seed(args.seed)
    process(args, model)
