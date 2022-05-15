import json
import os
import shutil

import torch


def yes_no_input():
    while True:
        choice = input("Please respond with 'yes' or 'no' [y/N]: ").lower()
        if choice in ['y', 'ye', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False


def check_if_done(filename):
    if os.path.exists(filename):
        print("%s already exists. Is it O.K. to overwrite it and start this program?" % filename)
        if not yes_no_input():
            raise Exception("Please restart training after you set args.savename differently!")


def mkdir_if_not_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_dic_to_json(dic, fn, verbose=True):
    with open(fn, "w") as f:
        json_str = json.dumps(dic, sort_keys=True, indent=4)
        if verbose:
            print(json_str)
        f.write(json_str)
    print("param file '%s' was saved!" % fn)


def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs, decay_epoch=15):
    """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
    lr = lr_init
    if epoch == num_epochs // 2:
        lr *= 0.1
    elif epoch == round(num_epochs * 0.75):
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
