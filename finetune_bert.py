import torch

from torch.utils.data import TensorDataset, RandomSampler, DataLoader,SequentialSampler
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm, trange
import os
import numpy as np
import argparse
import random
import math
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import matplotlib
import matplotlib.pyplot as plt

def finetune_bert():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_path",
                        required=True,
                        type=str)

    parser.add_argument("--eval_data_path",
                        required=True,
                        type=str)
    parser.add_argument("--output_dir",
                        required=True,
                        type=str)
    parser.add_argument("--data_column",
                        required=True,
                        type=str)
    parser.add_argument("--label_column",
                        required=True,
                        type=str)

    # parser.add_argument("--tokenizer_file",
    #                     type=str)
    # parser.add_argument("--config_file",
    #                     type=str)
    parser.add_argument("--model",
                        type=str)

    parser.add_argument("--max_len",
                        default=256,
                        type=int)
    parser.add_argument("--batch_size",
                        default=16,
                        type=int)
    parser.add_argument("--num_epochs",
                        default=4,
                        type=int)
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float)
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float)
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float)
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float)
    parser.add_argument("--random_seed",
                        default=42,
                        type=int)


    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.listdir(args.output_dir):
        print("Error: the specified directory is not empty.")
        print("Please specify an empty directory in order to avoid rewriting "
              "important data.")
        sys.exit()

    print("Setting the random seed...")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_path = os.path.join(args.output_dir, "log")

    results_f1 = []
    iteration = []


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    tokenizer = BertTokenizer.from_pretrained(model_name , do_lower_case=False)



    print("Reading data...")
    df_data = pd.read_csv(args.train_data_path, sep="\t")

    df_data = df_data.sample(frac=1, random_state=args.random_seed)
    train_data = df_data[args.data_column].tolist()
    train_labels = df_data[args.label_column].tolist()

    label_set = sorted(list(set(df_data[args.label_column].values)))

    train_labels = encode_labels(train_labels, label_set)



    df_eval_data = pd.read_csv(args.eval_data_path, sep="\t")
    eval_data = df_eval_data[args.data_column].tolist()
    eval_labels = df_eval_data[args.label_column].tolist()
    eval_labels = encode_labels(eval_labels, label_set)


    #Load Model
    model = BertForSequenceClassification.from_pretrained(model_name,
                                                          num_labels=len(label_set))

    output_subdir = args.output_dir
    print(output_subdir)
    if not os.path.exists(output_subdir):
        os.mkdir(output_subdir)

    train_dataloader = prepare_labeled_data(train_data, train_labels, tokenizer, args.max_len, args.batch_size, split='train')
    eval_dataloader = prepare_labeled_data(eval_data, eval_labels, tokenizer, args.max_len, args.batch_size, split='eval')

    _, eval_metrics = bert_train(model, device, train_dataloader, eval_dataloader, train_data, eval_data, label_set, output_subdir, args.num_epochs,
                       args.warmup_proportion, args.weight_decay, args.learning_rate, args.adam_epsilon,
                       save_best=True)

    #plotting metrics for eval data
    output_eval_plot_file = os.path.join(output_subdir, "eval_f1_plot.png")
    x = np.arange(1, args.num_epochs+1, 1)
    eval_f1_scores = []
    for dict in eval_metrics:
        eval_f1_scores.append(dict['f1'])
    x_label = "Epoch"
    y_label = "macro F1"
    plot_f1(x, eval_f1_scores, x_label, y_label, output_eval_plot_file)

    print("Done.")


def plot_f1(iterations, results, x_label, y_label, output_file):
    fig, ax = plt.subplots()
    ax.plot(iterations, results)

    ax.set(xlabel=x_label, ylabel=y_label)
    ax.grid()

    fig.savefig(output_file)


def bert_train(model, device, train_dataloader, eval_dataloader,  train_data, eval_data,
               labels_set, output_dir, num_epochs, warmup_proportion, weight_decay,
               learning_rate, adam_epsilon, save_best=False):
    """Training loop for bert fine-tuning. Save best works with F1 only currently."""

    t_total = len(train_dataloader) * num_epochs
    warmup_steps = len(train_dataloader) * warmup_proportion
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    train_iterator = trange(int(num_epochs), desc="Epoch")
    model.to(device)
    tr_loss_track = []
    eval_metric_track = []
    output_filename = os.path.join(output_dir, 'pytorch_model.bin')
    f1 = float('-inf')
    predictions_file_name_counter = 1

    for _ in train_iterator:

        train_predictions_filename = os.path.join(output_dir,
                                                 "train_predictions_epoch_" + str(predictions_file_name_counter))

        eval_predictions_filename = os.path.join(output_dir,
                                                 "eval_predictions_epoch_" + str(predictions_file_name_counter))

        model.train()
        model.zero_grad()
        tr_loss = 0
        nr_batches = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            tr_loss = 0
            input_ids, input_mask, labels = batch
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=input_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
            nr_batches += 1
            model.zero_grad()

        print("Evaluating the model on the train split...")
        metrics, train_predictions, prob_0, prob_1 = bert_evaluate(model, train_dataloader, device)
        # train_metric_track.append(metrics)
        write_model_outputs(train_predictions, train_predictions_filename, train_data, labels_set, prob_0, prob_1)

        print("Evaluating the model on the evaluation split...")
        metrics, eval_predictions,prob_0, prob_1 = bert_evaluate(model, eval_dataloader, device)
        eval_metric_track.append(metrics)
        write_model_outputs(eval_predictions, eval_predictions_filename, eval_data, labels_set,prob_0, prob_1)
        if save_best:
            if f1 < metrics['f1']:
                model.save_pretrained(output_dir)
                torch.save(model.state_dict(), output_filename)
                print("The new value of f1 score of " + str(metrics['f1']) + " is higher then the old value of " +
                      str(f1) + ".")
                print("Saving the new model...")
                f1 = metrics['f1']
            else:
                print("The new value of f1 score of " + str(metrics['f1']) + " is not higher then the old value of " +
                      str(f1) + ".")

        predictions_file_name_counter += 1

        tr_loss = tr_loss / nr_batches
        tr_loss_track.append(tr_loss)

    if not save_best:
        model.save_pretrained(output_dir)
        # tokenizer.save_pretrained(output_dir)
        torch.save(model.state_dict(), output_filename)

    return tr_loss_track, eval_metric_track


def bert_evaluate(model, eval_dataloader, device):
    """Evaluation of trained checkpoint."""
    model.to(device)
    model.eval()
    predictions = []
    prob_0 = []
    prob_1 = []
    true_labels = []
    data_iterator = tqdm(eval_dataloader, desc="Iteration")
    softmax = torch.nn.Softmax(dim=-1)
    for step, batch in enumerate(data_iterator):
        input_ids, input_mask, labels = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        # loss is only output when labels are provided as input to the model ... real smooth
        logits = outputs[0]
        probs = softmax(logits)
        # print(type(logits))
        logits = logits.to('cpu').numpy()
        label_ids = labels.to('cpu').numpy()

        for label,l, prob in zip(label_ids,logits, probs):
            true_labels.append(label)
            predictions.append(np.argmax(l))
            prob_0.append(prob[0].to('cpu').numpy())
            prob_1.append(prob[1].to('cpu').numpy())
    metrics = get_metrics(true_labels, predictions)
    return metrics, predictions, prob_0, prob_1


def bert_predict_during_train(model, dataloader, device):
    """Outputing predictions from a trained model."""
    model.eval()
    predictions = []
    prob_0 = []
    prob_1 = []
    data_iterator = tqdm(dataloader, desc="Iteration")
    softmax = torch.nn.Softmax(dim=-1)
    for step, batch in enumerate(data_iterator):
        input_ids, input_mask, _ = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        # loss is only output when labels are provided as input to the model ... real smooth
        logits = outputs[0]
        probs = softmax(logits)
        # print(type(logits))
        logits = logits.to('cpu').numpy()
        for l, prob in zip(logits, probs):
            predictions.append(np.argmax(l))
            prob_0.append(prob[0].to('cpu').numpy())
            prob_1.append(prob[1].to('cpu').numpy())

    return predictions, prob_0, prob_1



def prepare_labeled_data(data, labels, tokenizer, max_len, batch_size, split='train'):
    for i, sentence in enumerate(data):
        if isinstance(sentence, float):
            data[i] = " "

    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in data]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized_sentences]

    # dtype must be long because BERT apparently expects it
    input_ids = pad_sequences(input_ids, dtype='long', maxlen=max_len, padding="post", truncating="post")

    # attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    attention_masks = torch.tensor(attention_masks)

    transformed_data = TensorDataset(input_ids, attention_masks, labels)
    if split == 'train':
        sampler = RandomSampler(transformed_data)
    else:
        sampler = SequentialSampler(transformed_data)
    dataloader = DataLoader(transformed_data, sampler=sampler, batch_size=batch_size)

    return dataloader


def prepare_data(data, tokenizer, max_len, batch_size, split='train'):
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in data]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized_sentences]

    # dtype must be long because BERT apparently expects it
    input_ids = pad_sequences(input_ids, dtype='long', maxlen=max_len, padding="post", truncating="post")

    # attention masks
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    transformed_data = TensorDataset(input_ids, attention_masks)
    if split == 'train':
        sampler = RandomSampler(transformed_data)
    else:
        sampler = SequentialSampler(transformed_data)
    dataloader = DataLoader(transformed_data, sampler=sampler, batch_size=batch_size)

    return dataloader


def encode_labels(labels, labels_set):
    """Maps each label to a unique index.
    :param labels: (list of strings) labels of every instance in the dataset
    :param labels_set: (list of strings) set of labels that appear in the dataset
    :return (list of int) encoded labels
    """
    encoded_labels = []
    for label in labels:
        encoded_labels.append(labels_set.index(label))
    return encoded_labels


def get_metrics(actual, predicted):
    metrics = {'accuracy': accuracy_score(actual, predicted),
               'recall': recall_score(actual, predicted, average="macro"),
               'precision': precision_score(actual, predicted, average="macro"),
               'f1': f1_score(actual, predicted, average="macro")}

    return metrics


def write_model_outputs(predictions, output_file, original_data, labels_set,prob_0, prob_1):
    f = open(output_file, 'a')
    # print(predictions)
    for p, data, p0,p1 in zip(predictions, original_data,prob_0, prob_1):
        f.write(str(labels_set[int(p)]) + "\t")
        f.write(str(p0) + "\t")
        f.write(str(p1) + "\t")
        f.write(data + "\n")
    f.close()


if __name__ == "__main__":
    finetune_bert()
