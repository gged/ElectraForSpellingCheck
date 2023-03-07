# coding=utf-8
# email: wangzejunscut@126.com

import math
import random
import streamlit as st
import time
import torch

from easytokenizer import AutoTokenizer
from model import ElectraForSpellingCheck
from transformers import ElectraModel

@st.cache_data
def load_samples(path):
    samples = []
    with open(path, mode="r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip().split("\t", maxsplit=1)
            sentence = line[0]
            if sentence:
                samples.append(sentence)
    return samples

@st.cache_resource
def load_tokenizer(path):
    tokenizer = AutoTokenizer(path)
    return tokenizer

@st.cache_resource
def load_small_model(path, device):
    pretrained_model = ElectraModel.from_pretrained("hfl/chinese-electra-180g-small-discriminator")
    model = ElectraForSpellingCheck(pretrained_model)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_base_model(path, device):
    pretrained_model = ElectraModel.from_pretrained("hfl/chinese-electra-180g-base-discriminator")
    model = ElectraForSpellingCheck(pretrained_model)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def isChinese(c):
    cp = ord(c[0])
    if cp >= 0x4E00 and cp <= 0x9FA5:
        return True
    return False

def hanzi_ratio(sentence):
    if not sentence:
        return 0
    count = 0
    for c in sentence:
        if isChinese(c):
            count += 1
    return count / len(sentence)

def split_sentence(text, max_length):
    i = 0
    n = len(text)
    count = 0
    start = 0
    pos_list = []
    sent_list = []
    while i < n:
        i += 1
        count += 1
        if count == max_length:
            move = 1
            while move < max_length / 2:
                c = text[i - move]
                if c in ["。", "！", "？", "；", "\n", "!", "?", ";"]:
                    i = i - move + 1
                    break
                move += 1
            if move == max_length / 2:
                move = 1
                while move < max_length / 2:
                    c = text[i - move]
                    if c in [",", "，"]:
                        i = i - move + 1
                        break
                    move += 1
            sent = text[start : i].rstrip().replace("\n", "#")
            if hanzi_ratio(sent) > 0.5:
                pos_list.append(start)
                sent_list.append(sent)
            start = i
            count = 0
    if count:
        sent = text[start : ].rstrip().replace("\n", "#")
        if hanzi_ratio(sent) > 0.5:
            pos_list.append(start)
            sent_list.append(sent)
    return (pos_list, sent_list)

def inference(text, model, tokenizer, device, max_length, batch_size, threshold):
    result = []
    pos_list, sent_list = split_sentence(text, max_length)
    n = len(sent_list)
    num_batches = math.ceil(n / batch_size)
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        batch_sent_list = sent_list[start : end]
        batch_encodings = tokenizer.encode(batch_sent_list, max_length=256)
        batch_offsets = batch_encodings["offsets"]
        batch_input_ids = torch.tensor(batch_encodings["input_ids"], device=device)
        batch_attention_mask = torch.tensor(batch_encodings["attention_mask"], device=device)
        with torch.no_grad():
            logits = model(batch_input_ids, batch_attention_mask)
        predictions = logits.sigmoid()
        indexes = torch.nonzero(predictions > threshold)
        length = batch_attention_mask.sum(dim=-1)
        for index in indexes:
            row = index[0]
            col = index[1]
            if col > 0 and col < length[row] - 1:
                pos = batch_offsets[row][2 * (col - 1)]
                c = batch_sent_list[row][pos]
                if isChinese(c):
                    result.append(pos_list[start + row] + pos)
    return result 

def html_for_show(text, pos, color):
    color = color.lower()
    char_list = list(text)
    for p in pos:
        char_list[p] = "<font " + "color=" + color + ">" + char_list[p] + "</font>"
    char_list = ["<br>" if c == "\n" else c for c in char_list]
    return "".join(char_list)
    
def main():
    st.set_page_config(
        page_title="中文拼写检查系统演示",
        page_icon=" ", layout="centered")
    
    samples = load_samples("data/sighan_all.txt")
    tokenizer = load_tokenizer("data/vocab.txt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    small_model = load_small_model("checkpoint/small_warmup_5000_lr_1e-4_epochs_3/pytorch_model.bin", device)
    base_model = load_base_model("checkpoint/base_warmup_5000_lr_1e-4_epochs_3/pytorch_model.bin", device)
    
    st.sidebar.markdown("## 中文拼写检查系统")
    st.sidebar.caption("联系人: 王泽军")
    st.sidebar.caption("email: wangzejunscut@126.com")
    st.sidebar.markdown("<div style='border:1px solid white'></div>", unsafe_allow_html=True)

    model_radio = st.sidebar.radio(label="请选择模型:", options=("small", "base"))
    model = base_model if model_radio == "base" else small_model
    threshold_slider = st.sidebar.slider("请选择阈值:", 0., 1., 0.5, 0.01)
    color_slider = st.sidebar.select_slider("错误字符颜色:", 
        ("Red","Orange","Yellow","Green","Blue","Purple"), "Red")
    max_length = st.sidebar.number_input("请输入最大句子长度:", value=128)
    batch_size = st.sidebar.number_input("请输入批处理大小:", value=20)
    
    button1 = st.button("随机示例")
    if button1:
        text = st.text_area("请输入文本:", value=random.choice(samples), height=200)
        button2 = st.button("开始检查")
        start_time = time.time()
        output = inference(text, model, tokenizer, device, max_length, batch_size, threshold_slider)
        end_time = time.time()
        html = html_for_show(text, output, color_slider)
        time_usage = "**检查耗时:** {:.4f}s".format(end_time - start_time)
        st.markdown(time_usage)
        st.markdown("**检查结果:**")
        st.markdown(html, unsafe_allow_html=True)
        return
    
    text = st.text_area("请输入文本:", height=200)
    button2 = st.button("开始检查")
    if button2:
        start_time = time.time()
        output = inference(text, model, tokenizer, device, max_length, batch_size, threshold_slider)
        end_time = time.time()
        html = html_for_show(text, output, color_slider)
        time_usage = "**检查耗时:** {:.4f}s".format(end_time - start_time)
        st.markdown(time_usage)
        st.markdown("**检查结果:**")
        st.markdown(html, unsafe_allow_html=True)
            
if __name__ == "__main__":
    main()
