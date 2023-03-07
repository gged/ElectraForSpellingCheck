# coding=utf-8
# email: wangzejunscut@126.com

import argparse
import math
import os
import time

import torch
import torchmetrics
from torch.optim import AdamW
from torch.utils.data import DataLoader

from accelerate import Accelerator
from dataset import SpellCheckDataset, Collate
from easytokenizer import AutoTokenizer
from model import ElectraForSpellingCheck
from transformers import set_seed, get_linear_schedule_with_warmup, AutoModel

def main():
    parser = argparse.ArgumentParser(description="Sequence labeling model for spelling error check.")
    parser.add_argument("--train_file",
                        type=str,
                        required=True,
                        help="The full path of training set.")
    parser.add_argument("--eval_file",
                        type=str,
                        required=True,
                        help="The full path of evaluation set.")
    parser.add_argument("--vocab_file",
                        type=str,
                        required=True,
                        help="The full path of vocabulary file.")
    
    parser.add_argument("--seed", 
                        type=int, 
                        default=42,
                        help="Random seed for initialization. Default 42")
    parser.add_argument("--output_dir",
                        type=str,
                        default="./checkpoint",
                        help="Optional directory where model checkpoint will be stored. Default checkpoint/")
    parser.add_argument("--pretrained_model_name_or_path",
                        type=str,
                        default="hfl/chinese-electra-180g-base-discriminator",
                        help="The pretrained base model to be used. Default hfl/chinese-electra-180g-base-discriminator")
    parser.add_argument("--do_lower_case",
                        action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="If passed, will train on the CPU.")
    parser.add_argument("--mixed_precision",
                        type=str,
                        default=None,
                        choices=["no", "fp16", "bf16"],
                        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). "
                             "Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU. Default None")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=64,
                        help="Batch size per GPU/CPU for training. Default 64")
    parser.add_argument("--eval_batch_size",
                        type=int,
                        default=64,
                        help="Batch size per GPU/CPU for evaluation. Default 64")
    parser.add_argument("--max_seq_length",
                        type=int,
                        default=192,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated. Default 192")
    parser.add_argument("--max_steps",
                        type=int,
                        default=-1,
                        help="Set total number of training steps to perform. If > 0: Override num_train_epochs. Default -1")
    parser.add_argument("--epochs",
                        type=int,
                        default=3,
                        help="Total number of training epochs to perform. Default 3")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=5e-5,
                        help="The initial learning rate for Adam. Default 5e-5")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.01,
                        help="Weight decay if we apply some. Default 0.01")
    parser.add_argument("--warmup_proportion", 
                        type=float,
                        default=0.1,    
                        help="Linear warmup proportion over the training process. Default 0.1")
    parser.add_argument("--warmup_steps",
                        type=int,
                        default=None,
                        help="Linear warmup steps over the training process. Default None")
    parser.add_argument("--max_grad_norm", 
                        type=float,
                        default=1.0,
                        help="Max gradient norm. Default 1.0")
    parser.add_argument("--magnification",
                        type=float,
                        default=1.0,
                        help="Loss magnification. Default 1.0")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass. Default 1")
    parser.add_argument("--loss_type",
                        type=str,
                        default="bce",
                        choices=["lsr", "bce", "focal"],
                        help="Loss function type. Default bce")
    parser.add_argument("--reduction",
                        type=str,
                        default="mean",
                        choices=["mean", "sum"],
                        help="Specify the reduction to apply to loss tensor. Default mean")
    parser.add_argument("--alpha",
                        type=float,
                        default=None,
                        help="Hyper parameter alpha in Focal loss. Default None")
    parser.add_argument("--gamma",
                        type=float,
                        default=2.0,
                        help="Hyper parameter gamma in Focal loss. Default 2.0")
    parser.add_argument("--label_smoothing",
                        type=float,
                        default=0.1,
                        help="The smoothing factor in LSR loss. Default 0.1")
    parser.add_argument("--pos_weight",
                        type=float,
                        default=1.0,
                        help="The weight of positive examples. Default 1.0")
    parser.add_argument("--save_best_model",
                        action="store_true",
                        help="Whether to save checkpoint on best evaluation performance.")
    parser.add_argument("--logging_steps",
                        type=int,
                        default=20,
                        help="The interval steps to log. Default 20")
    parser.add_argument("--save_steps",
                        type=int,
                        default=200,
                        help="The interval steps to save checkpoint. Default 200")
    args = parser.parse_args()
    train(args)
    
def train(args):
    # Initialize accelerator
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)
    
    # Set random seed
    set_seed(args.seed)
    
    # Fast tokenizer
    tokenizer = AutoTokenizer(args.vocab_file, do_lower_case=args.do_lower_case)
    
    # Load dataset
    with accelerator.main_process_first():
        train_dataset = SpellCheckDataset(args.train_file, tokenizer, max_seq_length=args.max_seq_length)
        eval_dataset  = SpellCheckDataset(args.eval_file, tokenizer, max_seq_length=args.max_seq_length)
    
    # Instantiate dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        collate_fn=Collate(pad_token_id=tokenizer.pad_id()),
        batch_size=args.train_batch_size
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        shuffle=False,
        collate_fn=Collate(pad_token_id=tokenizer.pad_id()),
        batch_size=args.eval_batch_size
    )
    
    # Instantiate the model
    pretrained_model = AutoModel.from_pretrained(args.pretrained_model_name_or_path)
    model = ElectraForSpellingCheck(pretrained_model)
    
    # Instantiate optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Instantiate scheduler
    if args.max_steps > 0:
        num_training_steps = args.max_steps
        args.epochs = math.ceil(args.max_steps / (len(train_dataloader // args.gradient_accumulation_steps)))
    else:
        num_training_steps = (len(train_dataloader) * args.epochs) // args.gradient_accumulation_steps
    
    if args.warmup_steps is not None:
        num_warmup_steps = args.warmup_steps
    else:
        num_warmup_steps = int(num_training_steps * args.warmup_proportion)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Prepare everything
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # Train the model
    bestF1 = 0
    global_step = 0
    os.makedirs(args.output_dir, exist_ok=True)
    saved_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    num_training_steps = num_training_steps // accelerator.num_processes
    tic_train = time.time()
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            input_ids, attention_mask, labels = batch
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                labels=labels,
                loss_type=args.loss_type,
                reduction=args.reduction,
                alpha=args.alpha,
                gamma=args.gamma,
                label_smoothing=args.label_smoothing,
                pos_weight=args.pos_weight
            )
            
            loss = outputs[0]
            loss = loss * args.magnification / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                if global_step % args.logging_steps == 0:
                    gathered_loss = accelerator.gather(loss)
                    # Use accelerator.print to print only on the main process.
                    accelerator.print("global step %d/%d, epoch: %d, batch: %d, lr: %.10f, loss: %.4f, speed: %.2f step/s" % 
                                        (global_step, num_training_steps, epoch + 1, step + 1, optimizer.param_groups[0]['lr'], 
                                        gathered_loss.mean(), args.logging_steps / (time.time() - tic_train)))
                    tic_train = time.time()
                
                if global_step % args.save_steps == 0 or global_step == num_training_steps:
                    Precision, Recall, F1Score = evaluate(accelerator, model, eval_dataloader)
                    accelerator.print("Precision: %.4f, Recall: %.4f, F1Score: %.4f" % (Precision, Recall, F1Score))
                    if args.save_best_model:
                        if bestF1 < F1Score:
                            bestF1 = F1Score
                            accelerator.print("save model checkpoint!")
                            #accelerator.save_state(args.output_dir)
                            accelerator.wait_for_everyone()
                            state_dict = accelerator.get_state_dict(model)
                            accelerator.save(state_dict, saved_model_file)
                    else:
                        #accelerator.save_state(args.output_dir)
                        accelerator.wait_for_everyone()
                        state_dict = accelerator.get_state_dict(model)
                        accelerator.save(state_dict, saved_model_file)
                    model.train()
                    tic_train = time.time()
                
                if global_step >= num_training_steps:
                    return


def evaluate(accelerator, model, dataloader):
    model.eval()
    metric = torchmetrics.StatScores("binary").to(accelerator.device)
    for step, batch in enumerate(dataloader):
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = logits.sigmoid().round()
        # Recursively pad the tensors from all devices to the same size.
        predictions, labels, attention_mask = accelerator.pad_across_processes(
            (predictions, labels, attention_mask), dim=1)
        # Gather tensors for metric calculation.
        predictions, labels, attention_mask = accelerator.gather_for_metrics(
            (predictions, labels, attention_mask))

        active_index = attention_mask == 1
        active_predictions = predictions[active_index]
        active_labels = labels[active_index]
        metric.update(active_predictions, active_labels)
    
    # Compute Precision / Recall / F1Score
    scores = metric.compute().cpu().numpy()
    tp, fp, tn, fn, _ = scores
    Precision = tp / (tp + fp) if tp + fp else 0.
    Recall = tp / (tp + fn) if tp + fn else 0.
    F1Score = (2 * Precision * Recall) / (Precision + Recall) if Precision else 0.
    return Precision, Recall, F1Score
    
if __name__ == "__main__":
    main()
