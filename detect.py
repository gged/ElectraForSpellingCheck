# coding=utf-8
# email: wangzejunscut@126.com

import argparse
import torch

from easytokenizer import AutoTokenizer
from model import ElectraForSpellingCheck
from transformers import ElectraModel

def detect(args, model, tokenizer, text):
    encoding = tokenizer.encode(text, truncation=True, max_length=args.max_seq_length)
    input_ids = torch.tensor([encoding["input_ids"]], device=args.device)
    offsets = encoding["offsets"]
    with torch.no_grad():
        logits = model(input_ids)
    predictions = logits.squeeze(dim=0).sigmoid().round()
    index = torch.nonzero(predictions).squeeze(dim=-1).tolist()
    output = []
    for idx in index:
        idx = offsets[2 * (idx - 1)]
        output.append((idx, text[idx]))
    print("检查结果: ", output)
    
def main():
    parser = argparse.ArgumentParser(description="Chinese Spelling Check Demo")
    parser.add_argument("--model_file",
                        type=str,
                        required=True,
                        help="The full path of model file.")
    parser.add_argument("--vocab_file",
                        type=str,
                        required=True,
                        help="The full path of vocabulary file.")
    parser.add_argument("--do_lower_case",
                        action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length",
                        type=int,
                        default=256,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated.")
    parser.add_argument("--pretrained_model_name_or_path",
                        type=str,
                        default="hfl/chinese-electra-180g-base-discriminator",
                        help="The pretrained base model to be used.")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer(args.vocab_file, do_lower_case=args.do_lower_case)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    pretrained_model = ElectraModel.from_pretrained(args.pretrained_model_name_or_path)
    model = ElectraForSpellingCheck(pretrained_model)
    state_dict = torch.load(args.model_file)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    while True:
        text = input("请输入待检查的文本(最长256个字符, quit/q 退出): ")
        if text in ["quit", "q"]:
            break
        detect(args, model, tokenizer, text)

if __name__ == "__main__":
    main()
