from __future__ import division

import os

import torch
from torch.utils import data

import wandb
from args import get_train_args
from datasets import ConcatDataset, get_dataset, check_src_tgt_ok, get_n_class
from epochs import TrainEpoch, ValidEpoch
from loss import CrossEntropyLoss, IdentificationLoss, VertebraeCharacteristicsLoss
from metrics import IoU, Recall, Fscore, DetectionMetricWrapper, IdentificationRate
from models.model_util import get_models, get_optimizer, get_n_parameters
from utility_functions.util import mkdir_if_not_exist, save_dic_to_json, check_if_done, save_checkpoint, adjust_learning_rate

torch.cuda.empty_cache()

parser = get_train_args()
args = parser.parse_args()
args.n_class = get_n_class(args.mode)
args.machine = os.uname()[1]

check_src_tgt_ok(args.src_dataset, args.tgt_dataset)

assert (
                   args.use_vertebrae_loss and not args.without_detections and args.mode == "identification") or not args.use_vertebrae_loss
assert (args.use_vertebrae_loss and not args.no_da) or not args.use_vertebrae_loss

args.start_epoch = 0
resume_flg = True if args.resume else False
if args.resume:
    print("=> loading checkpoint '{}'".format(args.resume))
    if not os.path.exists(args.resume):
        raise OSError("%s does not exist!" % args.resume)

    indir, infn = os.path.split(args.resume)

    old_savename = args.savename
    args.savename = infn.split("-")[0]
    print("savename is %s (original savename %s was overwritten)" % (args.savename, old_savename))

    checkpoint = torch.load(args.resume, map_location='cuda')
    args.start_epoch = checkpoint["epoch"]
    # ---------- Replace Args!!! ----------- #
    args2 = checkpoint['args']
    # -------------------------------------- #
    model_g, model_head = get_models(mode=args2.mode,
                                     n_class=args2.n_class,
                                     is_data_parallel=args2.is_data_parallel)

    optimizer = get_optimizer(list(model_head.parameters()) + list(model_g.parameters()), lr=args.lr, opt=args.opt,
                              momentum=args.momentum, weight_decay=args.weight_decay)

    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_head.load_state_dict(checkpoint['f1_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}'".format(args.resume))

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

else:
    model_g, model_head = get_models(mode=args.mode,
                                     n_class=args.n_class,
                                     is_data_parallel=args.is_data_parallel)
    optimizer = get_optimizer(list(model_head.parameters()) + list(model_g.parameters()), opt=args.opt,
                              lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

print(f"Parameters: Encoder={get_n_parameters(model_g)}, Head={get_n_parameters(model_head)}")

mode = "%s2%s" % (args.src_dataset, args.tgt_dataset)
model_name = "%s-%s" % (args.savename, args.mode)
outdir = os.path.join(args.base_outdir, mode)

# Create Model Dir
pth_dir = os.path.join(outdir, "pth")
mkdir_if_not_exist(pth_dir)

# Log with WandB
if args.use_wandb:
    if args.mode == "identification":
        wandb.init(project=f"uda-vi-identification", job_type='train', config=args)
    else:
        wandb.init(project=f"uda-vi-detection", job_type='train', config=args)

# Save param dic
if resume_flg:
    json_fn = os.path.join(outdir, "param-%s_resume.json" % model_name)
else:
    json_fn = os.path.join(outdir, "param-%s.json" % model_name)
check_if_done(json_fn)
save_dic_to_json(args.__dict__, json_fn)

dataset_train = get_dataset(args.src_dataset, "train", type="source", mode=args.mode,
                            use_data_augmentation=args.use_data_augmentation, with_detection=False)
dataset_valid = get_dataset(args.src_dataset, "test", type="source", mode=args.mode,
                            use_data_augmentation=args.use_data_augmentation, with_detection=False)

if not args.no_da or args.use_vertebrae_loss:
    tgt_dataset_train = get_dataset(args.tgt_dataset, "train", type="target", mode=args.mode,
                                    use_data_augmentation=args.use_data_augmentation,
                                    with_detection=not args.without_detections and args.mode != "detection",
                                    use_train_labels_target=args.train_some_tgt_labels,
                                    with_weak_mask=args.use_region_proposal_loss and args.mode != "detection")
    dataset_train = ConcatDataset(
        dataset_train,
        tgt_dataset_train
    )

    tgt_dataset_valid = get_dataset(args.tgt_dataset, "test", type="target", mode=args.mode,
                                    use_data_augmentation=args.use_data_augmentation,
                                    with_detection=not args.without_detections and args.mode != "detection",
                                    with_weak_mask=not args.use_region_proposal_loss and args.mode != "detection")
    dataset_valid = ConcatDataset(
        dataset_valid,
        tgt_dataset_valid
    )

train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    num_workers=0,
)

valid_loader = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    num_workers=0,
)

if args.use_labeled_tgt:
    tgt_dataset_labeled = get_dataset(args.tgt_dataset, "test", type="target-labeled", mode=args.mode,
                                      use_data_augmentation=args.use_data_augmentation,
                                      with_detection=not args.without_detections and args.mode != "detection",
                                      with_weak_mask=not args.use_region_proposal_loss and args.mode != "detection")
    target_labeled_loader = torch.utils.data.DataLoader(
        tgt_dataset_labeled,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=0,
    )

if torch.cuda.is_available():
    model_g.cuda()
    model_head.cuda()

if args.mode == "detection":
    weight = torch.FloatTensor([.1, 1.])
    if not args.ignore_bg_loss:
        weight[0] = 0  # Ignore background loss

    if torch.cuda.is_available():
        weight = weight.cuda()

    criterion = CrossEntropyLoss(weight)
    criterion_vert = None

    metrics = {"IoU": DetectionMetricWrapper(IoU(), args.n_class),
               "IoU ignored 0": DetectionMetricWrapper(IoU(ignore_channels=[0]), args.n_class),
               "IoU ignored 1": DetectionMetricWrapper(IoU(ignore_channels=[1]), args.n_class),
               "Recall": DetectionMetricWrapper(Recall(), args.n_class),
               "Recall ignored 0": DetectionMetricWrapper(Recall(ignore_channels=[0]), args.n_class),
               "Recall ignored 1": DetectionMetricWrapper(Recall(ignore_channels=[1]), args.n_class),
               "F-Score": DetectionMetricWrapper(Fscore(), args.n_class)
               }

elif args.mode == "identification":
    criterion = IdentificationLoss()

    if args.use_vertebrae_loss:
        criterion_vert = VertebraeCharacteristicsLoss(args.use_descending_loss, args.use_vertical_equal_loss,
                                                      args.use_center_dist_loss, args.use_region_proposal_loss)
    else:
        criterion_vert = None

    metrics = {"Classification Rate": IdentificationRate(),
               }

else:
    raise AttributeError()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_epoch = TrainEpoch(
    model_g=model_g,
    model_head=model_head,
    criterion=criterion,
    criterion_vert=criterion_vert,
    optimizer=optimizer,
    metrics=metrics,
    n_classes=args.n_class,
    device=device,
    verbose=not args.no_verbose,
    use_domain_adaptation=not args.no_da,
    with_detections=not args.without_detections and args.mode != "detection",
)
valid_epoch = ValidEpoch(
    model_g=model_g,
    model_head=model_head,
    criterion=criterion,
    criterion_vert=criterion_vert,
    metrics=metrics,
    n_classes=args.n_class,
    device=device,
    verbose=not args.no_verbose,
    use_domain_adaptation=not args.no_da,
    with_detections=not args.without_detections and args.mode != "detection",
)

if args.use_labeled_tgt:
    valid_epoch_labeled_tgt = ValidEpoch(
        model_g=model_g,
        model_head=model_head,
        criterion=criterion,
        criterion_vert=criterion_vert,
        metrics=metrics,
        n_classes=args.n_class,
        device=device,
        verbose=not args.no_verbose,
        use_domain_adaptation=False,
        stage_name='valid labeled tgt',
        with_detections=not args.without_detections and args.mode != "detection",
    )

for epoch in range(args.start_epoch, args.epochs):

    train_logs = train_epoch.run(train_loader, epoch)
    valid_logs = valid_epoch.run(valid_loader, epoch)
    if args.use_labeled_tgt:
        valid_labeled_tgt_logs = valid_epoch_labeled_tgt.run(target_labeled_loader, epoch)

    if args.use_wandb:
        logs = {
            'epoch': epoch,
            'learning rate': args.lr,
        }
        logs = {**logs, **train_logs}
        logs = {**logs, **valid_logs}

        if args.use_labeled_tgt:
            logs = {**logs, **valid_labeled_tgt_logs}

        wandb.log(logs)

    if args.adjust_lr:
        args.lr = adjust_learning_rate(optimizer, args.lr, args.weight_decay, epoch, args.epochs)

    checkpoint_fn = os.path.join(pth_dir, "%s-%s.pth.tar" % (model_name, epoch + 1))
    args.start_epoch = epoch + 1
    save_dic = {
        'epoch': epoch + 1,
        'args': args,
        'g_state_dict': model_g.state_dict(),
        'f1_state_dict': model_head.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    save_checkpoint(save_dic, is_best=False, filename=checkpoint_fn)
