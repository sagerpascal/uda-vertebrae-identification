import argparse


def get_train_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='PyTorch Vertebrae Detection and Identification')
    parser.add_argument('--src_dataset', type=str, default="biomedia",  choices=["biomedia"])
    parser.add_argument('--tgt_dataset', type=str, default="covid19-ct", choices=["covid19-ct", "verse19"])
    parser.add_argument('--mode', type=str, default="detection", choices=["detection", "identification"])
    parser.add_argument('--use_labeled_tgt', action="store_true", help="Use labeled target data for evaluation during training")
    parser.add_argument("--train_some_tgt_labels", action="store_true", help='use some target labels during training to compare performance')
    parser.add_argument('--use_data_augmentation', action="store_true", help="Whether to apply data augmentation")
    parser.add_argument('--without_detections', action="store_true", help="Whether not to use the detection output during identification")
    parser.add_argument("--is_data_parallel", action="store_true", help='whether you use torch.nn.DataParallel')

    # ---------- How to Save ---------- #
    parser.add_argument('--savename', type=str, default="normal", help="save name(Do NOT use '-')")
    parser.add_argument('--base_outdir', type=str, default='train_output', help="base output dir")
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')

    # ---------- Define Loss Detection ---------- #
    parser.add_argument("--ignore_bg_loss", action="store_false", help='whether you add background loss or not')

    # ---------- Define Loss Identification ---------- #
    parser.add_argument("--use_vertebrae_loss", action="store_true",
                        help='whether to use an additional loss for identification on target labels')
    parser.add_argument("--use_descending_loss", action="store_false", help='whether you use Descending Target Loss')
    parser.add_argument("--use_vertical_equal_loss", action="store_false",
                        help='whether you use Vertical Equal Target Loss')
    parser.add_argument("--use_center_dist_loss", action="store_false",
                        help='whether you use Center Distance Target Loss')
    parser.add_argument("--use_region_proposal_loss", action="store_false",
                        help='whether you use Region Proposal Target Loss')

    # ---------- Hyperparameters ---------- #
    parser.add_argument('--opt', type=str, default="adam", choices=['sgd', 'adam'], help="network optimizer")
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 0.0005)')
    parser.add_argument("--adjust_lr", action="store_true", help='whether you change lr')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum sgd (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=2e-5, help='weight_decay (default: 2e-5)')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help="batch_size")


    # ---------- Mic. ---------- #
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers of the data loader")
    parser.add_argument("--resume", type=str, default=None, metavar="PTH.TAR", help="model(pth) path")
    parser.add_argument('--use_wandb', action="store_true", help="Whether to log results with weight & bias")
    parser.add_argument('--no_verbose', action="store_true", help="Whether to use console output")
    parser.add_argument('--no_da', action="store_true", help="Whether to use Domain Adaptation")

    return parser
