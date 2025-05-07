import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hybridmqa.data.dataset import DatasetBuilder, custom_collate_fn
from hybridmqa.model.archit import GNN, BaseEncoder, HybridMQA, QualityEncoder
from hybridmqa.utils import (RankLoss, accumulate_step_outputs,
                             count_model_params, create_pred_dataframe, plcc,
                             send_dict_tensors_to_device, srcc)


def main():
    # Cuda Setup
    if torch.cuda.is_available():
        logging.info("Cuda is available!")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        logging.info("Cuda is not available!")
        device = torch.device("cpu")

    # Create the dataset
    split = 'all' if args.test_all else 'test'
    validset = DatasetBuilder().build(
        name=args.dataset,
        root_dir=args.root_dir,
        split=split,
        normalize_score=args.norm,
        seed=args.shuffle_seed,
        kfold_seed=args.kfold_seed
    )
    validloader = DataLoader(
        validset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=args.num_work,
        pin_memory=args.pin_mem
    )

    # Create an instance of the model
    # create Base Encoder
    base_enc = BaseEncoder(
        hidden_channels=args.enc_ch[0],
        out_channels=args.enc_ch[1],
        input_maps=args.enc_map,
    ).to(device)
    # create GNN
    gnn_in_chs = args.enc_ch[1] if args.dataset != 'VCMesh' else 9
    gnn = GNN(
        in_channels=gnn_in_chs,
        hidden_channels=args.gnn_ch[0],
        out_channels=args.gnn_ch[1],
        gnn_arch=args.gnn_arch,
        gnn_act=args.cg_act
    ).to(device)
    # create Quality Encoder
    quality_enc = QualityEncoder(
        in_channels=args.gnn_ch[1],
        hidden_channels=args.reg_ch[0],
        out_channels=args.reg_ch[1],
        mask_attn=args.mask,
        num_proj=6,
        patch_size=args.patch_size,
        nonempty_ratio=args.emp_rat,
        stride_ratio=args.str_rat,
        flip_aug=False
    ).to(device)
    # create HybridMQA model
    fc_input_dim = args.gnn_ch[1] + 5 * args.reg_ch[1]
    if args.dataset in ['YN2023', 'VCMesh']:
        lighting = 'directional'
    elif args.dataset in ['TMQA', 'TSMD']:
        lighting = 'ambient'
    else:
        raise ValueError(f"Unknown dataset: {args.dataset} for the purpose of lighting selection.")
    model = HybridMQA(
        base_enc=base_enc,
        gnn=gnn,
        quality_enc=quality_enc,
        fc_input_dim=fc_input_dim,
        img_size=args.img_size,
        num_proj=6,
        angle_aug=False,
        lighting=lighting,
        vcmesh=args.dataset == 'VCMesh',
    ).to(device)

    # log model summary
    logging.info(model)
    logging.info(count_model_params(model))

    # Define the loss function
    reg_criterion = nn.L1Loss()
    rank_criterion = RankLoss() if args.rloss else None

    # predictions dataframes
    valid_pred_df = pd.DataFrame(columns=['Stimulus', 'Target'])

    # Load Checkpoint
    if args.ckpt is not None:
        logging.info(f"Laoding checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        raise ValueError('Checkpoint direction should be given!')

    valid_epoch_outputs = validation_step(
        validloader=validloader,
        model=model,
        reg_criterion=reg_criterion,
        rank_criterion=rank_criterion,
        device=device,
        rloss=args.rloss,
        rank_wgt=args.rank_wgt,
        plcc_map=not args.no_map
    )
    valid_pred_df = create_pred_dataframe(main_df=valid_pred_df, epoch_outputs=valid_epoch_outputs, epoch=0)
    valid_pred_df.to_csv(os.path.join(args.save, 'Predictions.csv'), index=False)


def validation_step(
    validloader: DataLoader,
    model: nn.Module,
    reg_criterion: nn.Module,
    rank_criterion: Optional[nn.Module],
    device: torch.device,
    rloss: bool,
    rank_wgt: float,
    plcc_map: bool
) -> Dict[str, Union[List[str], torch.Tensor]]:
    """
    Executes a single validation epoch for the HybridMQA model.

    Args:
        validloader (DataLoader): The PyTorch DataLoader providing validation batches.
        model (nn.Module): The HybridMQA model to test.
        reg_criterion (nn.Module): The loss function used for quality regression (e.g., L1).
        rank_criterion (Optional[nn.Module]): The ranking loss function. Used if `rloss=True`.
        device (torch.device): The device on which to run validation (CPU or CUDA).
        rloss (bool): Whether to report ranking loss in validation.
        rank_wgt (float): Weight assigned to the rank loss when combined with regression loss.
        plcc_map (bool): Whether to use PLCC mapping when computing Pearson correlation.

    Returns:
        Dict[str, Union[List[str], torch.Tensor]]: A dictionary of accumulated outputs across the epoch,
            including object names, predictions, targets, and averaged loss components.
    """
    num_samples = len(validloader.dataset)
    valid_step_outputs = {
        'names': [],
        'preds': torch.empty(0, device=device),
        'targets': torch.empty(0, device=device),
        'reg_loss': torch.empty(0, device=device),
        'rank_loss': torch.empty(0, device=device),
        'loss': torch.empty(0, device=device)
    }
    model.eval()
    with torch.no_grad():
        for sample_batched in tqdm(validloader):
            names_batched = sample_batched['names']
            # delete names from the 'sample_batched' dict before sending dict values to device
            sample_batched.pop('names')
            sample_batched = send_dict_tensors_to_device(dict_of_tns=sample_batched, device=device)

            # Forward pass
            pred_quality = model(sample_batched)
            # increase dim if batch size = 1
            pred_quality = pred_quality.unsqueeze(0) if pred_quality.dim() == 0 else pred_quality

            # Compute loss
            reg_loss = reg_criterion(pred_quality, sample_batched['score'])
            loss = reg_loss

            # rank loss
            if rloss:
                rank_loss = rank_criterion(pred_quality, sample_batched['score'])
                loss = loss + rank_wgt * rank_loss
            else:
                rank_loss = torch.tensor(0.0, device=device)

            # accumulate information
            valid_step_outputs = accumulate_step_outputs(
                step_outputs=valid_step_outputs,
                names_batched=names_batched,
                pred_quality=pred_quality,
                sample_batched=sample_batched,
                reg_loss=reg_loss,
                rank_loss=rank_loss,
                loss=loss
            )

    avg_valid_reg_loss = valid_step_outputs['reg_loss'].sum() / num_samples
    avg_valid_rnk_loss = valid_step_outputs['rank_loss'].sum() / num_samples
    avg_valid_loss = valid_step_outputs['loss'].sum() / num_samples
    valid_srcc = srcc(pred=valid_step_outputs['preds'], target=valid_step_outputs['targets'])
    valid_plcc = plcc(pred=valid_step_outputs['preds'], target=valid_step_outputs['targets'], mapping=plcc_map)

    logging.info(
        f"Avg Valid Reg Loss: {avg_valid_reg_loss:.4f}, "
        f"Avg Valid Rnk Loss: {avg_valid_rnk_loss:.4f}, "
        f"Avg Valid Loss: {avg_valid_loss:.4f}, "
        f"Valid SRCC: {valid_srcc:.4f}, Valid PLCC: {valid_plcc:.4f}"
    )

    return valid_step_outputs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='mesh_qa')
    # General Experiment Setup
    parser.add_argument('--dataset', type=str, help='dataset to test on')
    parser.add_argument('--tr_dataset', type=str, help='dataset the model was trained on - for logging purposes')
    parser.add_argument('--root_dir', type=str, help='root dir of the test dataset')
    parser.add_argument('--norm', action='store_true', help='whether to normalize the MOS of the dataset')
    parser.add_argument('--save', type=str, default='exp', help='dir to save experiments')
    parser.add_argument('--exp_cmt', type=str, default='DEFAULT', help='optional comment about the experiment')
    parser.add_argument('--ckpt', type=str, help='path to the model checkpoint')
    # Training Configurations
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    # Dataloader Configurations
    parser.add_argument('--num_work', type=int, default=0, help='num workers in DataLoaders')
    parser.add_argument('--pin_mem', action='store_true', help='pin_memory in DataLoaders')
    parser.add_argument('--shuffle_seed', type=int, default=1, help='seed for kfold split')
    parser.add_argument('--kfold_seed', type=int, default=7, help='random seed for shuffling the dataset')
    parser.add_argument('--test_all', action='store_true', help='flag for enabling testing on the whole dataset')
    # model settings
    parser.add_argument('--gnn_arch', type=str, default='graph_2', help='GNN architecture: type_layers_[heads]')
    parser.add_argument('--mask', action='store_true', help='flag for enabling masked attention and pooling')
    parser.add_argument('--enc_map', type=str, default='all', help='input maps to the encoder')
    parser.add_argument('--cg_act', type=str, default='relu', help='conv/gnn block activation function')
    parser.add_argument('--enc_ch', nargs='+', type=int, default=[32, 16],
                        help='Base Encoder hidden and output channel dimensions')
    parser.add_argument('--gnn_ch', nargs='+', type=int, default=[16, 16],
                        help='GNN hidden and output channel dimensions')
    parser.add_argument('--reg_ch', nargs='+', type=int, default=[32, 64],
                        help='Quality Encoder hidden and output channel dimensions')
    parser.add_argument('--img_size', type=int, default=512, help='image sizes for rendering')
    parser.add_argument('--patch_size', type=int, default=64, help='Patch size used during patchification')
    parser.add_argument('--emp_rat', type=float, default=0.1, help='non_empty ratio for discarding empty patches')
    parser.add_argument('--str_rat', type=float, default=1.0, help='stride ratio for patchifying')
    # Loss Configurations
    parser.add_argument('--rloss', action='store_true', help='whether to use rank loss in addition to regression loss')
    parser.add_argument('--rank_wgt', type=float, default=1.0, help='weight factor for rank loss')
    # plcc mapping
    parser.add_argument('--no_map', action='store_true', help='if set, will disable plcc mapping')
    args = parser.parse_args()

    # Create a directory to save the experiment
    args.save = os.path.join(args.save, time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(args.save, exist_ok=True)

    # Set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # log all input arguments
    for arg in vars(args):
        if arg != 'root_dir':
            logging.info(f"{arg}: {getattr(args, arg)}")

    main()
