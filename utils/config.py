import os
import time
import torch
import argparse

from utils.basic_utils import mkdirp, load_json, save_json, make_zipfile, dict_to_markdown
import shutil

class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = None
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        parser = argparse.ArgumentParser()
        parser.add_argument("--dset_name", type=str, choices=["ADNI", 'ADNI_90_120_fMRI', 'FTD_90_200_fMRI', 'OCD_90_200_fMRI', 'PPMI'])
        parser.add_argument("--model", type=str, choices=["MLP", 'TCN', 'KNN', 'SVM'])
        parser.add_argument("--results_root", type=str, default="results")
        parser.add_argument("--exp_id", type=str, default=None, help="id of this run, required at training")
        parser.add_argument("--seed", type=int, default=2018, help="random seed")
        parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        parser.add_argument("--num_workers", type=int, default=0,
                            help="num subprocesses used to load the data, 0: use main process")
        parser.add_argument("--no_pin_memory", action="store_true",
                            help="Don't use pin_memory=True for dataloader. "
                                 "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")

        # training config
        parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        parser.add_argument("--lr_drop", type=int, default=500, help="drop learning rate to 1/10 every lr_drop epochs")
        parser.add_argument("--n_epoch", type=int, default=300, help="number of epochs to run")
        parser.add_argument("--bsz", type=int, default=16, help="mini-batch size")
        parser.add_argument("--eval_bsz", type=int, default=100,
                            help="mini-batch size at inference, for query")
        parser.add_argument("--hdim", type=int, default=256, help="learning rate")
        parser.add_argument("--num_layers", type=int, default=2, help="learning rate")
        parser.add_argument("--dropout", type=float, default=0.1, help="learning rate")
        parser.add_argument("--n_neighbors", type=int, default=10, help="learning rate")
        parser.add_argument("--kernel_size", type=int, default=3, help="learning rate")
        parser.add_argument("--grad_clip", type=float, default=0.2, help="perform gradient clip, -1: disable")
        parser.add_argument("--resume", type=str, default=None,
                            help="checkpoint path to resume or evaluate, without --resume_all this only load weights")
        parser.add_argument("--resume_all", action="store_true",
                            help="if --resume_all, load optimizer/scheduler/epoch as well")
        self.parser = parser

    def display_save(self, opt):
        args = vars(opt)
        # Display settings
        print(dict_to_markdown(vars(opt), max_str_len=120))
        # Save settings
        option_file_path = os.path.join(opt.results_dir, self.saved_option_filename)  # not yaml file indeed
        save_json(args, option_file_path, save_pretty=True)

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        opt.results_dir = os.path.join(opt.results_root,
                                        "-".join([opt.dset_name, opt.exp_id, opt.model,
                                                    time.strftime("%Y_%m_%d_%H_%M_%S")]))
        mkdirp(opt.results_dir)

        self.display_save(opt)

        opt.ckpt_filepath = os.path.join(opt.results_dir, self.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, self.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, self.eval_log_filename)
        opt.tensorboard_log_dir = os.path.join(opt.results_dir, self.tensorboard_log_dir)
        opt.device = torch.device("cuda" if opt.device >= 0 else "cpu")
        opt.pin_memory = not opt.no_pin_memory

        self.opt = opt
        return opt
