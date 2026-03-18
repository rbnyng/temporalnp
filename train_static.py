"""
Train Neural Process for static (single-year) biomass estimation.

Uses BufferedSpatialSplitter and GEDINeuralProcessDataset for pure spatial
evaluation, matching the original static training pipeline.

Usage:
    python train_static.py \
        --region_bbox -73 2 -72 3 \
        --start_time 2022-01-01 \
        --end_time 2022-12-31 \
        --embedding_year 2022 \
        --output_dir ./results/guaviare_2022
"""

import argparse
import json
from pathlib import Path
import pickle
from time import time

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.gedi import GEDIQuerier
from data.embeddings import create_embedding_extractor, EMBEDDING_SOURCES
from data.dataset import GEDINeuralProcessDataset, collate_neural_process
from data.spatial_cv import BufferedSpatialSplitter
from models.neural_process import (
    GEDINeuralProcess,
    neural_process_loss,
)
from utils.config import save_config, _make_serializable
from utils.evaluation import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train Static GEDI Neural Process')

    # Data arguments
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--start_time', type=str, default='2022-01-01',
                        help='Start date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--end_time', type=str, default='2022-12-31',
                        help='End date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--embedding_year', type=int, default=2022,
                        help='Year of embeddings')
    parser.add_argument('--embedding_source', type=str, default='geotessera',
                        choices=['geotessera', 'alphaearth'],
                        help='Embedding source: geotessera (128D) or alphaearth (64D)')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Directory for caching GEDI query results')
    parser.add_argument('--embeddings_dir', type=str, default='./embeddings',
                        help='Directory where embeddings are stored')
    parser.add_argument('--ee_project', type=str, default=None,
                        help='Google Cloud project ID for Earth Engine (alphaearth only)')

    # Model arguments
    parser.add_argument('--patch_size', type=int, default=3,
                        help='Embedding patch size (default: 3x3)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer dimension')
    parser.add_argument('--embedding_feature_dim', type=int, default=1024,
                        help='Embedding feature dimension')
    parser.add_argument('--context_repr_dim', type=int, default=256,
                        help='Context representation dimension')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent variable dimension')
    parser.add_argument('--architecture_mode', type=str, default='anp',
                        choices=['deterministic', 'latent', 'anp', 'cnp'],
                        help='Architecture mode')
    parser.add_argument('--num_attention_heads', type=int, default=16,
                        help='Number of attention heads')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (number of tiles)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--buffer_size', type=float, default=0.1,
                        help='Buffer size in degrees for spatial CV')
    parser.add_argument('--min_shots_per_tile', type=int, default=10,
                        help='Minimum GEDI shots per tile')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--lr_scheduler_patience', type=int, default=5,
                        help='LR scheduler patience (epochs)')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5,
                        help='LR scheduler reduction factor')
    parser.add_argument('--kl_weight_max', type=float, default=0.01,
                        help='Maximum KL weight')
    parser.add_argument('--kl_warmup_epochs', type=int, default=10,
                        help='Number of epochs to warm up KL weight')

    # Dataset arguments
    parser.add_argument('--agbd_scale', type=float, default=200.0,
                        help='AGBD scale factor for normalization')
    parser.add_argument('--log_transform_agbd', type=lambda x: x.lower() == 'true', default=True,
                        help='Apply log transform to AGBD')
    parser.add_argument('--augment_coords', action='store_true', default=True,
                        help='Add coordinate augmentation')
    parser.add_argument('--coord_noise_std', type=float, default=0.01,
                        help='Standard deviation for coordinate noise')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for models and logs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    # Other
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, optimizer, device, kl_weight=1.0):
    model.train()
    total_loss = 0
    total_nll = 0
    total_kl = 0
    n_tiles = 0

    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()

        batch_loss = 0
        batch_nll = 0
        batch_kl = 0
        n_tiles_in_batch = 0

        for i in range(len(batch['context_coords'])):
            context_coords = batch['context_coords'][i].to(device)
            context_embeddings = batch['context_embeddings'][i].to(device)
            context_agbd = batch['context_agbd'][i].to(device)
            target_coords = batch['target_coords'][i].to(device)
            target_embeddings = batch['target_embeddings'][i].to(device)
            target_agbd = batch['target_agbd'][i].to(device)

            if len(target_coords) == 0:
                continue

            pred_mean, pred_log_var, z_mu_context, z_log_sigma_context, z_mu_all, z_log_sigma_all = model(
                context_coords,
                context_embeddings,
                context_agbd,
                target_coords,
                target_embeddings,
                query_agbd=target_agbd,
                training=True
            )

            loss, loss_dict = neural_process_loss(
                pred_mean, pred_log_var, target_agbd,
                z_mu_context, z_log_sigma_context,
                z_mu_all, z_log_sigma_all,
                kl_weight
            )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected, skipping batch.")
                continue

            batch_loss += loss
            batch_nll += loss_dict['nll']
            batch_kl += loss_dict['kl']
            n_tiles_in_batch += 1

        if n_tiles_in_batch > 0:
            batch_loss = batch_loss / n_tiles_in_batch
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            total_nll += batch_nll / n_tiles_in_batch
            total_kl += batch_kl / n_tiles_in_batch
            n_tiles += n_tiles_in_batch

    return {
        'loss': total_loss / max(n_tiles, 1),
        'nll': total_nll / max(n_tiles, 1),
        'kl': total_kl / max(n_tiles, 1)
    }


def validate(model, dataloader, device, kl_weight=1.0, agbd_scale=200.0,
             log_transform_agbd=True):
    _, _, _, metrics, loss_dict = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        max_context_shots=100000,
        max_targets_per_chunk=10000,
        compute_loss=True,
        kl_weight=kl_weight,
        agbd_scale=agbd_scale,
        log_transform_agbd=log_transform_agbd,
        denormalize_for_reporting=False
    )
    return loss_dict, metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    save_config(config, output_dir / 'config.json')

    embedding_channels = EMBEDDING_SOURCES[args.embedding_source]['channels']

    print("=" * 80)
    print("Static GEDI Neural Process Training")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Region: {args.region_bbox}")
    print(f"Time range: {args.start_time} to {args.end_time}")
    print(f"Embedding: {args.embedding_source} ({embedding_channels}D)")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Query GEDI data
    print("Step 1: Querying GEDI data...")
    querier = GEDIQuerier(cache_dir=args.cache_dir)
    gedi_df = querier.query_region_tiles(
        region_bbox=args.region_bbox,
        tile_size=0.1,
        start_time=args.start_time,
        end_time=args.end_time,
        max_agbd=500.0
    )
    print(f"Retrieved {len(gedi_df)} GEDI shots across {gedi_df['tile_id'].nunique()} tiles")

    if len(gedi_df) == 0:
        print("No GEDI data found in region. Exiting.")
        return

    # Step 2: Extract embeddings
    print(f"\nStep 2: Extracting {args.embedding_source} embeddings...")
    extractor = create_embedding_extractor(
        source=args.embedding_source,
        year=args.embedding_year,
        patch_size=args.patch_size,
        embeddings_dir=args.embeddings_dir,
        ee_project=args.ee_project,
    )
    gedi_df = extractor.extract_patches_batch(gedi_df, verbose=True, cache_dir=args.cache_dir)
    gedi_df = gedi_df[gedi_df['embedding_patch'].notna()]
    print(f"Retained {len(gedi_df)} shots with valid embeddings")
    print()

    # Save processed data
    with open(output_dir / 'processed_data.pkl', 'wb') as f:
        pickle.dump(gedi_df, f)

    # Step 3: Spatial split (buffered)
    print("Step 3: Creating buffered spatial split...")
    splitter = BufferedSpatialSplitter(
        gedi_df,
        buffer_size=args.buffer_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
    train_df, val_df, test_df = splitter.split()
    print()

    global_bounds = (
        train_df['longitude'].min(),
        train_df['latitude'].min(),
        train_df['longitude'].max(),
        train_df['latitude'].max()
    )
    config['global_bounds'] = list(global_bounds)
    config['embedding_channels'] = embedding_channels
    save_config(config, output_dir / 'config.json')

    print(f"Global bounds: lon [{global_bounds[0]:.4f}, {global_bounds[2]:.4f}], "
          f"lat [{global_bounds[1]:.4f}, {global_bounds[3]:.4f}]")

    # Step 4: Create datasets
    print("\nStep 4: Creating datasets...")
    train_dataset = GEDINeuralProcessDataset(
        train_df,
        min_shots_per_tile=args.min_shots_per_tile,
        agbd_scale=args.agbd_scale,
        log_transform_agbd=args.log_transform_agbd,
        augment_coords=args.augment_coords,
        coord_noise_std=args.coord_noise_std,
        global_bounds=global_bounds
    )
    val_dataset = GEDINeuralProcessDataset(
        val_df,
        min_shots_per_tile=args.min_shots_per_tile,
        agbd_scale=args.agbd_scale,
        log_transform_agbd=args.log_transform_agbd,
        augment_coords=False,
        coord_noise_std=0.0,
        global_bounds=global_bounds
    )
    test_dataset = GEDINeuralProcessDataset(
        test_df,
        min_shots_per_tile=args.min_shots_per_tile,
        agbd_scale=args.agbd_scale,
        log_transform_agbd=args.log_transform_agbd,
        augment_coords=False,
        coord_noise_std=0.0,
        global_bounds=global_bounds
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_neural_process, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_neural_process, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_neural_process, num_workers=args.num_workers
    )
    print()

    # Step 5: Initialize model (2D coords - spatial only)
    print("Step 5: Initializing model...")
    print(f"Architecture mode: {args.architecture_mode}")
    model = GEDINeuralProcess(
        patch_size=args.patch_size,
        embedding_channels=embedding_channels,
        embedding_feature_dim=args.embedding_feature_dim,
        context_repr_dim=args.context_repr_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        output_uncertainty=True,
        architecture_mode=args.architecture_mode,
        num_attention_heads=args.num_attention_heads
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience
    )

    # Step 6: Training
    print("Step 6: Training...")
    best_val_loss = float('inf')
    best_r2 = float('-inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    training_start_time = time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        if args.kl_warmup_epochs > 0:
            kl_weight = min(1.0, epoch / args.kl_warmup_epochs) * args.kl_weight_max
        else:
            kl_weight = args.kl_weight_max

        train_metrics = train_epoch(model, train_loader, optimizer, args.device, kl_weight)
        train_losses.append(train_metrics['loss'])

        val_losses_dict, val_metrics = validate(
            model, val_loader, args.device, kl_weight,
            agbd_scale=args.agbd_scale,
            log_transform_agbd=args.log_transform_agbd,
        )
        val_losses.append(val_losses_dict['loss'])

        print(f"Train Loss: {train_metrics['loss']:.6e} (NLL: {train_metrics['nll']:.6e}, KL: {train_metrics['kl']:.6e})")
        print(f"Val Loss:   {val_losses_dict['loss']:.6e} (NLL: {val_losses_dict['nll']:.6e}, KL: {val_losses_dict['kl']:.6e})")

        if val_metrics:
            print(f"Val Log R²:       {val_metrics.get('log_r2', 0):.4f}")
            print(f"Val Log RMSE:     {val_metrics.get('log_rmse', 0):.4f}")
            print(f"Val Log MAE:      {val_metrics.get('log_mae', 0):.4f}")
            print(f"Val Linear RMSE:  {val_metrics.get('linear_rmse', 0):.2f} Mg/ha")
            print(f"Val Linear MAE:   {val_metrics.get('linear_mae', 0):.2f} Mg/ha")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6e}, KL Weight: {kl_weight:.4f}")

        scheduler.step(val_losses_dict['loss'])

        if val_losses_dict['loss'] < best_val_loss:
            best_val_loss = val_losses_dict['loss']
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses_dict['loss'],
                'val_metrics': val_metrics
            }, output_dir / 'best_model.pt')
            print("Saved best model (lowest val loss)")
        else:
            epochs_without_improvement += 1

        current_r2 = val_metrics.get('log_r2', float('-inf')) if val_metrics else float('-inf')
        if current_r2 > best_r2:
            best_r2 = current_r2
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses_dict['loss'],
                'val_metrics': val_metrics,
                'r2': current_r2
            }, output_dir / 'best_r2_model.pt')
            print(f"Saved best R² model (log-space R² = {best_r2:.4f})")

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')

    training_time = time() - training_start_time

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.6e}")
    print(f"Best R² score (log space): {best_r2:.4f}")
    print(f"Models saved to: {output_dir}")
    print("=" * 80)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_r2_model.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_losses_dict, test_metrics = validate(
        model, test_loader, args.device, kl_weight=1.0,
        agbd_scale=args.agbd_scale,
        log_transform_agbd=args.log_transform_agbd,
    )

    print(f"Test Loss:        {test_losses_dict['loss']:.6e}")
    if test_metrics:
        print(f"\nLog-space metrics:")
        print(f"  Test Log R²:    {test_metrics.get('log_r2', 0):.4f}")
        print(f"  Test Log RMSE:  {test_metrics.get('log_rmse', 0):.4f}")
        print(f"  Test Log MAE:   {test_metrics.get('log_mae', 0):.4f}")
        print(f"\nLinear-space metrics (Mg/ha):")
        print(f"  Test RMSE:      {test_metrics.get('linear_rmse', 0):.2f} Mg/ha")
        print(f"  Test MAE:       {test_metrics.get('linear_mae', 0):.2f} Mg/ha")
        if 'coverage_1sigma' in test_metrics:
            print(f"\nUncertainty Calibration:")
            print(f"  Z-score mean: {test_metrics.get('z_mean', 0):.3f} (ideal: 0.0)")
            print(f"  Z-score std:  {test_metrics.get('z_std', 0):.3f} (ideal: 1.0)")
            print(f"  Coverage 1σ:  {test_metrics.get('coverage_1sigma', 0):.1f}% (ideal: 68.3%)")
            print(f"  Coverage 2σ:  {test_metrics.get('coverage_2sigma', 0):.1f}% (ideal: 95.4%)")
            print(f"  Coverage 3σ:  {test_metrics.get('coverage_3sigma', 0):.1f}% (ideal: 99.7%)")

    checkpoint['test_metrics'] = test_metrics
    torch.save(checkpoint, output_dir / 'best_r2_model.pt')
    print("Added test metrics to best model checkpoint")

    results = {
        'train_time': training_time,
        'best_val_loss': best_val_loss,
        'best_r2': best_r2,
        'test_loss': test_losses_dict,
        'test_metrics': test_metrics,
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(_make_serializable(results), f, indent=2)

    print(f"Saved results to results.json")


if __name__ == '__main__':
    main()
