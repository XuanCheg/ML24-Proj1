import time
import json
import pprint
import random
import numpy as np

import torch
import torch.nn as nn

from utils.config import BaseOptions
from utils.eval import eval_epoch
from utils.model_utils import count_parameters

from dataset.dataset import CustomDataset as dataset
from utils.setup import setup_model
from torch.nn import functional as F
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)

    train_dataset = dataset(opt.dset_name, 'train')
    eval_dataset = dataset(opt.dset_name, 'val')
    test_dataset = dataset(opt.dset_name, 'test')
    if opt.dset_name == 'ADNI':
        input_dim = 186
        output_dim = 5
    elif opt.dset_name == 'ADNI_90_120_fMRI':
        input_dim = (90, 120)
        if opt.model == 'MLP':
            input_dim = 90 * 120
        output_dim = 4
    elif opt.dset_name == 'FTD_90_200_fMRI':
        input_dim = (90, 200)
        if opt.model == 'MLP':
            input_dim = 90 * 200
        output_dim = 2
    elif opt.dset_name == 'OCD_90_200_fMRI':
        input_dim = (90, 200)
        if opt.model == 'MLP':
            input_dim = 90 * 200
        output_dim = 2
    elif opt.dset_name == 'PPMI':
        input_dim = 294
        output_dim = 2
    else:
        raise ValueError("Unknown dataset")

    model = setup_model(opt, input_dim, output_dim, device=torch.device(opt.device))

    if opt.model in ['KNN', 'SVM']:
        opt.bsz = 10000

    criterion = nn.CrossEntropyLoss()
    # import pdb; pdb.set_trace()
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    except Exception as e:
        m = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(m.parameters(), lr=opt.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop, gamma=0.5)
    logger.info(f"Model {model}")
    count_parameters(model)
    logger.info("Start Training...")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bsz, shuffle=True, num_workers=opt.num_workers)
    best = 0
    
    for epoch in range(opt.n_epoch):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data

            if opt.dset_name in ['ADNI_90_120_fMRI', 'FTD_90_200_fMRI', 'OCD_90_200_fMRI'] and opt.model in ['MLP', 'KNN', 'SVM']:
                inputs = inputs.flatten(start_dim=1)

            if opt.model in ['KNN', 'SVM']:
                inputs, labels = inputs.numpy(), labels.numpy()
                model(inputs, labels)
                continue
            inputs, labels = inputs.to(opt.device), labels.to(opt.device)     

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            # outputs = F.softmax(outputs, dim=-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if opt.model in ['KNN', 'SVM']:
            break
        lr_scheduler.step()

        misc = eval_epoch(model, eval_dataset, opt, criterion)
        running_loss /= len(train_loader)
        with open(opt.train_log_filepath, 'a') as f:
            f.write(f"Epoch {epoch+1}/{opt.n_epoch}, Loss: {running_loss:.4f}, Eval: {misc}\n")
        print(f"\033[1;32mEpoch {epoch+1}/{opt.n_epoch}, Loss: {running_loss:.4f}, Eval: {misc} / {best}\033[0m")
        if misc['Accuracy'] > best:
            best = misc['Accuracy']
            save = misc
            torch.save(model.state_dict(), opt.ckpt_filepath)
    if opt.model in ['KNN', 'SVM']:
        misc = eval_epoch(model, eval_dataset, opt, criterion)
        with open(opt.train_log_filepath, 'a') as f:
            f.write(f"Val: {misc} \n")
        final_misc = eval_epoch(model, test_dataset, opt, criterion)
        with open(opt.train_log_filepath, 'a') as f:
            f.write(f"Test: {final_misc}")
        return
    
    with open(opt.train_log_filepath, 'a') as f:
        f.write('Val: ' + str(save) + '\n')

    model.load_state_dict(torch.load(opt.ckpt_filepath))
    final_misc = eval_epoch(model, test_dataset, opt, criterion)
    with open(opt.train_log_filepath, 'a') as f:
        f.write(f"Test: {final_misc}")
    logger.info("Training Finished.")

if __name__ == "__main__":
    start_training()