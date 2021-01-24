import argparse
import random
import math
import sys

import subprocess

base_out_dirs ={"all":"./models/all/",
               "cse":"./models/cse/",
               "engee":"./models/engee/"
                }

language_data = {
                "all": {"data": "text", "label": "label", "train_data_path": "./data/all_train.csv",
                           "eval_data_path": "./data/all_val.csv", "test_data_path": "./data/all_test.csv",
                           "offensive_label": "OFF"},
                "cse": {"data": "text", "label": "label", "train_data_path": "./data/cse_train.csv",
                                           "eval_data_path": "./data/cse_val.csv", "test_data_path": "./data/cse_test.csv",
                                           "offensive_label": "OFF"},

                "engee": {"data": "text", "label": "label", "train_data_path": "./data/engee_train.csv",
                                           # "eval_data_path": "../data/engee_val.csv", "test_data_path": "egnee_test.csv",
                                            "eval_data_path": "./data/engee_train.csv", "test_data_path": "./data/engee_train.csv",
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
                "--test_data_path", data_details['test_data_path'],
                "--eval_data_path", data_details['eval_data_path'],
                "--output_dir", out_dir,
                "--data_column", data_details['data'],
                "--label_column", data_details['label'],
                "--model", model
                ]
    subprocess.call(list_arg)

    # for pre_train_lang in pre_train_langs:
    #     model_details = int_model_files[pre_train_lang][model]
    #     base_out_dir = base_out_dirs[pre_train_lang]
    #
    #     langs = ['arabic', 'german','slovenian', 'croatian','english']
    #     seeds =  ['42', '84', '126']
    #     runs = ['1', '2', '3']
    #
    #     for lang in langs:
    #         #No task traning on same as pre-training language
    #         if lang == pre_train_lang:
    #             continue
    #         data_details = language_data[lang]
    #
    #         for run, seed in zip(runs, seeds):
    #             out_dir = base_out_dir + model + '_'+lang+run+'/'
    #
    #             if len(model_details['model_file']) == 0:
    #                 list_arg = [python_path, "incremental_learning.py",
    #                             "--train_data_path", data_details['train_data_path'],
    #                             "--test_data_path", data_details['test_data_path'],
    #                             "--eval_data_path", data_details['eval_data_path'],
    #                             "--output_dir", out_dir,
    #                             "--data_column", data_details['data'],
    #                             "--label_column", data_details['label'],
    #                             "--random_seed", seed]
    #
    #             elif len(model_details['tokenizer_file']) == 0:
    #
    #                 list_arg = [python_path, "incremental_learning.py",
    #                                  "--train_data_path", data_details['train_data_path'],
    #                                  "--test_data_path", data_details['test_data_path'],
    #                                  "--eval_data_path", data_details['eval_data_path'],
    #                                  "--output_dir", out_dir,
    #                                  "--data_column", data_details['data'],
    #                                  "--label_column", data_details['label'],
    #                                  "--config_file", model_details['config_file'],
    #                                  "--model_file", model_details['model_file'],
    #                                  "--random_seed", seed]
    #             else:
    #                 list_arg = [python_path, "incremental_learning.py",
    #                             "--train_data_path", data_details['train_data_path'],
    #                             "--test_data_path", data_details['test_data_path'],
    #                             "--eval_data_path", data_details['eval_data_path'],
    #                             "--output_dir", out_dir,
    #                             "--data_column", data_details['data'],
    #                             "--label_column", data_details['label'],
    #                             "--tokenizer_file",model_details['tokenizer_file'],
    #                             "--config_file", model_details['config_file'],
    #                             "--model_file", model_details['model_file'],
    #                             "--random_seed", seed]
    #             print(list_arg)
    #
    #             subprocess.call(list_arg)
