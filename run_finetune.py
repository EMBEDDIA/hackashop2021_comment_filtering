import argparse
import random
import math
import sys

import subprocess

base_out_dirs ={
                "all":"./models/all/",
               "cse":"./models/cse/",
               "engee":"./models/engee/"
                }

language_data = {
                "all": {"data": "text", "label": "label", "train_data_path": "./data/all_train.csv",
                           "eval_data_path": "./data/all_val.csv",
                           "offensive_label": "OFF"},
                "cse": {"data": "text", "label": "label", "train_data_path": "./data/cse_train.csv",
                                           "eval_data_path": "./data/cse_val.csv",
                                           "offensive_label": "OFF"},

                "engee": {"data": "text", "label": "label", "train_data_path": "./data/engee_train.csv",
                                           "eval_data_path": "../data/engee_val.csv",

                                           "offensive_label": "OFF"}

                }


bert = ['bert-base-multilingual-cased','EMBEDDIA/crosloengual-bert','EMBEDDIA/finest-bert']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--lang",
                        required=True,
                        type=str)
    parser.add_argument("--model",
                        required=True,
                        type=str)
    parser.add_argument("--python_path",
                        # required=True,
                        type=str,
                        default='python')
    #parser.add_argument("--gpu_id",
                        # required=True,
                        # type=int,
                        # default=-1)
    args = parser.parse_args()


    train_lang = args.lang
    model = args.model
    python_path = args.python_path


    data_details = language_data[train_lang]
    base_out_dir = base_out_dirs[train_lang]
    out_dir = base_out_dir + model + '_' + train_lang  + '/'

    list_arg = [python_path, "finetune_bert.py",
                "--train_data_path", data_details['train_data_path'],

                "--eval_data_path", data_details['eval_data_path'],
                "--output_dir", out_dir,
                "--data_column", data_details['data'],
                "--label_column", data_details['label'],
                "--model", model
                ]
    subprocess.call(list_arg)

