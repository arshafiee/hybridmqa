import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

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

    # wandb init
    if not args.no_wandb:
        run = wandb.init(project="mesh_qa_fr")
        wandb.run.name = args.save.split('/')[-1]
        wandb.config.update(args)

    # Create the datasets
    trainset = DatasetBuilder().build(
        name=args.dataset,
        root_dir=args.root_dir,
        split='train',
        normalize_score=args.norm,
        seed=args.shuffle_seed,
        kfold_seed=args.kfold_seed
    )
    validset = DatasetBuilder().build(
        name=args.dataset,
        root_dir=args.root_dir,
        split='test',
        normalize_score=args.norm,
        seed=args.shuffle_seed,
        kfold_seed=args.kfold_seed
    )
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=args.num_work,
        pin_memory=args.pin_mem
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
        num_proj=args.num_proj,
        patch_size=args.patch_size,
        nonempty_ratio=args.emp_rat,
        stride_ratio=args.str_rat,
        flip_aug=args.flip_aug
    ).to(device)
    # create HybridMQA model
    fc_input_dim = args.gnn_ch[1] + 5 * args.reg_ch[1]
    model = HybridMQA(
        base_enc=base_enc,
        gnn=gnn,
        quality_enc=quality_enc,
        fc_input_dim=fc_input_dim,
        img_size=args.img_size,
        num_proj=args.num_proj,
        angle_aug=args.ang_aug,
        dataset=args.dataset
    ).to(device)

    # log model summary
    logging.info(model)
    logging.info(count_model_params(model))

    # Define the loss function
    reg_criterion = nn.L1Loss()
    rank_criterion = RankLoss() if args.rloss else None

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-5, verbose=True)

    # predictions dataframes
    train_pred_df = pd.DataFrame(columns=['Stimulus', 'Target'])
    valid_pred_df = pd.DataFrame(columns=['Stimulus', 'Target'])

    # Resume training from checkpoint
    start_epoch = 0
    if args.ckpt is not None:
        logging.info(f"Resuming training from checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        train_epoch_outputs = train_step(
            trainloader=trainloader,
            model=model,
            reg_criterion=reg_criterion,
            rank_criterion=rank_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            num_epochs=args.num_epochs,
            rloss=args.rloss,
            rank_wgt=args.rank_wgt,
            plcc_map=not args.no_map
        )
        valid_epoch_outputs = validation_step(
            validloader=validloader,
            model=model,
            reg_criterion=reg_criterion,
            rank_criterion=rank_criterion,
            device=device,
            epoch=epoch,
            num_epochs=args.num_epochs,
            rloss=args.rloss,
            rank_wgt=args.rank_wgt,
            plcc_map=not args.no_map
        )
        if (epoch + 1) % 5 == 0:
            train_pred_df = create_pred_dataframe(main_df=train_pred_df, epoch_outputs=train_epoch_outputs, epoch=epoch)
            valid_pred_df = create_pred_dataframe(main_df=valid_pred_df, epoch_outputs=valid_epoch_outputs, epoch=epoch)

        # save checkpoint
        if (epoch + 1) % args.ckpt_step == 0:
            logging.info(f"Saving checkpoint for epoch {epoch + 1}")
            if not os.path.exists(os.path.join(args.save, 'ckpt')):
                os.makedirs(os.path.join(args.save, 'ckpt'))
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(ckpt, os.path.join(args.save, 'ckpt', f'ckpt_{epoch + 1}.pth'))

    # log predictions
    if not args.no_wandb:
        train_pred_table = wandb.Table(dataframe=train_pred_df)
        valid_pred_table = wandb.Table(dataframe=valid_pred_df)
        run.log({'train_preds': train_pred_table, 'valid_preds': valid_pred_table})


def train_step(
    trainloader: DataLoader,
    model: nn.Module,
    reg_criterion: nn.Module,
    rank_criterion: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    rloss: bool,
    rank_wgt: float,
    plcc_map: bool
) -> Dict[str, Union[List[str], torch.Tensor]]:
    """
    Executes a single training epoch for the HybridMQA model.

    Args:
        trainloader (DataLoader): The PyTorch DataLoader providing training batches.
        model (nn.Module): The HybridMQA model to train.
        reg_criterion (nn.Module): The loss function used for quality regression (e.g., L1).
        rank_criterion (Optional[nn.Module]): The ranking loss function. Used if `rloss=True`.
        optimizer (torch.optim.Optimizer): The optimizer for parameter updates.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        device (torch.device): The device on which to run training (CPU or CUDA).
        epoch (int): Current epoch index (zero-based).
        num_epochs (int): Total number of epochs.
        rloss (bool): Whether to include ranking loss in training.
        rank_wgt (float): Weight assigned to the rank loss when combined with regression loss.
        plcc_map (bool): Whether to use PLCC mapping when computing Pearson correlation.

    Returns:
        Dict[str, Union[List[str], torch.Tensor]]: A dictionary of accumulated outputs across the epoch,
            including object names, predictions, targets, and averaged loss components.
    """
    num_samples = len(trainloader.dataset)
    training_step_outputs = {
        'names': [],
        'preds': torch.empty(0, device=device),
        'targets': torch.empty(0, device=device),
        'reg_loss': torch.empty(0, device=device),
        'rank_loss': torch.empty(0, device=device),
        'loss': torch.empty(0, device=device)
    }
    model.train()
    for i_batch, sample_batched in enumerate(trainloader):
        # In the batch dimension of "sample_batched", odd indices correspond to distorted
        # inputs while even indices correspond to their reference counterparts.
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

        # Backward pass
        loss.backward()
        if (i_batch + 1) % args.grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()

        # accumulate information
        training_step_outputs = accumulate_step_outputs(
            step_outputs=training_step_outputs,
            names_batched=names_batched,
            pred_quality=pred_quality,
            sample_batched=sample_batched,
            reg_loss=reg_loss,
            rank_loss=rank_loss,
            loss=loss)

        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Step [{i_batch + 1}/{len(trainloader)}], reg loss: {reg_loss.item():.4f}, "
            f"rnk Loss: {rank_loss.item():.4f}, loss: {loss.item():.4f}"
        )
        if not args.no_wandb:
            wandb.log({'reg-loss': reg_loss.item()})
            wandb.log({'rank-loss': rank_loss.item()})
            wandb.log({'loss': loss.item()})

    avg_train_reg_loss = training_step_outputs['reg_loss'].sum() / num_samples
    avg_train_rnk_loss = training_step_outputs['rank_loss'].sum() / num_samples
    avg_train_loss = training_step_outputs['loss'].sum() / num_samples
    train_srcc = srcc(pred=training_step_outputs['preds'], target=training_step_outputs['targets'])
    train_plcc = plcc(pred=training_step_outputs['preds'], target=training_step_outputs['targets'], mapping=plcc_map)
    scheduler.step()

    logging.info(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Avg Train Reg Loss: {avg_train_reg_loss:.4f}, "
        f"Avg Train Rnk Loss: {avg_train_rnk_loss:.4f}, "
        f"Avg Train Loss: {avg_train_loss:.4f}, "
        f"Train SRCC: {train_srcc:.4f}, Train PLCC: {train_plcc:.4f}"
    )
    if not args.no_wandb:
        wandb.log({'avg_train_reg_loss': avg_train_reg_loss})
        wandb.log({'avg_train_rank_loss': avg_train_rnk_loss})
        wandb.log({'avg_train_loss': avg_train_loss})
        wandb.log({'train_srcc': train_srcc})
        wandb.log({'train_plcc': train_plcc})

    return training_step_outputs


def validation_step(
    validloader: DataLoader,
    model: nn.Module,
    reg_criterion: nn.Module,
    rank_criterion: Optional[nn.Module],
    device: torch.device,
    epoch: int,
    num_epochs: int,
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
        epoch (int): Current epoch index (zero-based).
        num_epochs (int): Total number of epochs.
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
        for i_batch, sample_batched in enumerate(validloader):
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
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Avg Valid Reg Loss: {avg_valid_reg_loss:.4f}, "
        f"Avg Valid Rnk Loss: {avg_valid_rnk_loss:.4f}, "
        f"Avg Valid Loss: {avg_valid_loss:.4f}, "
        f"Valid SRCC: {valid_srcc:.4f}, Valid PLCC: {valid_plcc:.4f}"
    )
    if not args.no_wandb:
        wandb.log({'avg_valid_reg_loss': avg_valid_reg_loss})
        wandb.log({'avg_valid_rank_loss': avg_valid_rnk_loss})
        wandb.log({'avg_valid_loss': avg_valid_loss})
        wandb.log({'valid_srcc': valid_srcc})
        wandb.log({'valid_plcc': valid_plcc})

    return valid_step_outputs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='mesh_qa')
    # General Experiment Setup
    parser.add_argument('--dataset', type=str, default='YN2023', help='dataset to use for training')
    parser.add_argument('--root_dir', type=str, help='root dir of the dataset')
    parser.add_argument('--norm', action='store_true', help='whether to normalize the MOS of the dataset')
    parser.add_argument('--save', type=str, default='exp', help='dir to save experiments')
    parser.add_argument('--exp_cmt', type=str, default='DEFAULT', help='optional comment about the experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='path to the checkpoint to resume training')
    parser.add_argument('--ckpt_step', type=int, default=15, help='epoch intervals to save checkpoints')
    # Training Configurations
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=15, help='number of training epochs')
    parser.add_argument('--grad_acc', type=int, default=1,
                        help='gradient accumulation steps, grad_acc=1 means no accumulation, '
                        'effective batch size = batch_size * grad_acc')
    # Dataloader Configurations
    parser.add_argument('--num_work', type=int, default=0, help='num workers in DataLoaders')
    parser.add_argument('--pin_mem', action='store_true', help='pin_memory in DataLoaders')
    parser.add_argument('--shuffle_seed', type=int, default=1, help='random seed for shuffling the dataset')
    parser.add_argument('--kfold_seed', type=int, default=7, help='random seed for kfold split')
    # model settings
    parser.add_argument('--num_proj', type=int, default=2, help='Number of projections for rendering in training')
    parser.add_argument('--gnn_arch', type=str, default='graph_2', help='GNN architecture: type_layers_[heads]')
    parser.add_argument('--mask', action='store_true', help='flag for enabling masked attention and pooling')
    parser.add_argument('--enc_map', type=str, default='all', help='input maps to the encoder')
    parser.add_argument('--cg_act', type=str, default='relu', help='gnn block activation function')
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
    # Data Augmentation
    parser.add_argument('--ang_aug', action='store_true', help='whether to apply angle augmentation in training')
    parser.add_argument('--flip_aug', action='store_true', help='whether to apply flip augmentation in training')
    # Loss Configurations
    parser.add_argument('--rloss', action='store_true', help='whether to use rank loss in addition to regression loss')
    parser.add_argument('--rank_wgt', type=float, default=1.0, help='weight factor for rank loss')
    # wandb support
    parser.add_argument('--no_wandb', action='store_true', help='if set, will disable logging to wandb')
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
