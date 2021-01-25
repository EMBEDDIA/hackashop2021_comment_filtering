import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm
import numpy as np

from keras.preprocessing.sequence import pad_sequences
import argparse
import pandas as pd
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification


def inference():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_data_path",
                        required=True,
                        type=str)
    parser.add_argument("--output_file_name",
                        required=True,
                        type=str)
    parser.add_argument("--data_column",
                        required=True,
                        type=str)
    parser.add_argument("--label_column",
                        required=True,
                        type=str)

    parser.add_argument("--saved_model_dir",
                        type=str)

    args = parser.parse_args()

    df_test_data = pd.read_csv(args.test_data_path, sep="\t")
    test_data = df_test_data[args.data_column].tolist()
    test_labels = df_test_data[args.label_column].tolist()
    print(test_labels)

    label_set = sorted(list(set(df_test_data[args.label_column].values)))
    test_labels = encode_labels(test_labels, label_set)
    print(test_labels)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model
    saved_model_dir = args.saved_model_dir
    model_name = saved_model_dir+'pytorch_model.bin'
    config_file = saved_model_dir +'config.json'
    config = BertConfig.from_pretrained(config_file)
    tokenizer = BertTokenizer.from_pretrained(config._name_or_path, do_lower_case=False)

    model = BertForSequenceClassification.from_pretrained(model_name,
                                                          num_labels=config.num_labels)
    max_len = 256 #because all models were trained with this window
    batch_size = 8 #arbitrary
    test_dataloader = prepare_data(test_data, tokenizer, max_len, batch_size)
    predictions, prob_0, prob_1 = bert_predict(model, test_dataloader, device)

    output_dict = {'label': predictions, 'data': test_data, 'prob_0': prob_0, 'prob_1': prob_1}
    df_output = pd.DataFrame(output_dict)
    df_output.to_csv(args.output_file_name, sep="\t", index=False)


def bert_predict(model, dataloader, device):
    """Outputing predictions from a trained model."""
    model.to(device)
    model.eval()
    predictions = []
    prob_0 = []
    prob_1 = []
    data_iterator = tqdm(dataloader, desc="Iteration")
    softmax = torch.nn.Softmax(dim=-1)
    for step, batch in enumerate(data_iterator):
        input_ids, input_mask = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)

        # loss is only output when labels are provided as input to the model ... real smooth
        logits = outputs[0]
        probs = softmax(logits)
        # print(type(logits))
        logits = logits.to('cpu').numpy()

        for l,prob in zip(logits, probs):
            predictions.append(np.argmax(l))
            prob_0.append(prob[0].to('cpu').numpy())
            prob_1.append(prob[1].to('cpu').numpy())

    return predictions,prob_0,prob_1


def prepare_data(data, tokenizer, max_len, batch_size):
    for i, sentence in enumerate(data):
        if isinstance(sentence, float):
            data[i] = " "

    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in data]

    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    # print("Example of tokenized sentence:")
    # print(tokenized_sentences[0])

    input_ids = [tokenizer.convert_tokens_to_ids(sentence) for sentence in tokenized_sentences]
    # print("Printing encoded sentences:")
    # print(input_ids[0])
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


if __name__ == "__main__":
    inference()