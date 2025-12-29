import argparse
import json
from pathlib import Path
import pickle
from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from data.gedi import GEDIQuerier
from data.embeddings import EmbeddingExtractor
from data.dataset import GEDINeuralProcessDataset, collate_neural_process
from data.spatial_cv import SpatialTileSplitter, BufferedSpatialSplitter
from models.neural_process import (
    GEDINeuralProcess,
    neural_process_loss,
)
from diagnostics import generate_all_diagnostics
from utils.evaluation import compute_metrics
from utils.config import save_config, _make_serializable
from utils.evaluation import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train GEDI Neural Process')

    # Data arguments
    parser.add_argument('--region_bbox', type=float, nargs=4, required=True,
                        help='Region bounding box: min_lon min_lat max_lon max_lat')
    parser.add_argument('--start_time', type=str, default='2022-01-01',
                        help='Start date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--end_time', type=str, default='2022-12-31',
                        help='End date for GEDI data (YYYY-MM-DD)')
    parser.add_argument('--train_years', type=int, nargs='+', default=None,
                        help='Specific years to use for training (e.g., 2019 2020 2021). '
                             'If specified, only GEDI shots from these years will be used '
                             'for training, enabling temporal validation on held-out years.')
    parser.add_argument('--embedding_year', type=int, default=2022,
                        help='Year of GeoTessera embeddings')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='Directory for caching GEDI query results')
    parser.add_argument('--embeddings_dir', type=str, default='./embeddings',
                        help='Directory where geotessera stores embedding tiles')

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
                        help='Architecture mode: deterministic (attention only), '
                             'latent (stochastic only), anp (both), cnp (baseline)')
    parser.add_argument('--num_attention_heads', type=int, default=16,
                        help='Number of attention heads')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (number of tiles)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (L2 regularization) for AdamW optimizer')
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
                        help='Maximum KL weight (for beta-VAE style training)')
    parser.add_argument('--kl_warmup_epochs', type=int, default=10,
                        help='Number of epochs to warm up KL weight from 0 to max')

    # Dataset arguments
    parser.add_argument('--agbd_scale', type=float, default=200.0,
                        help='AGBD scale factor for normalization (default: 200.0 Mg/ha)')
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
    parser.add_argument('--generate_diagnostics', action='store_true', default=True,
                        help='Generate diagnostic plots after training (default: True)')
    parser.add_argument('--n_diagnostic_samples', type=int, default=5,
                        help='Number of sample tiles to plot in diagnostics')

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

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected in training! Skipping batch.")
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


def validate(model, dataloader, device, kl_weight=1.0, agbd_scale=200.0, log_transform_agbd=True, denormalize_for_reporting=False):
    _, _, _, metrics, loss_dict = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        max_context_shots=100000,  # very high limit to avoid subsampling
        max_targets_per_chunk=10000,  # large chunks for training validation
        compute_loss=True,
        kl_weight=kl_weight,
        agbd_scale=agbd_scale,
        log_transform_agbd=log_transform_agbd,
        denormalize_for_reporting=denormalize_for_reporting
    )

    return loss_dict, metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_config(vars(args), output_dir / 'config.json')

    print("=" * 80)
    print("GEDI Neural Process Training")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Region: {args.region_bbox}")
    print(f"Output: {output_dir}")
    print()

    print("Step 1: Querying GEDI data...")
    querier = GEDIQuerier(cache_dir=args.cache_dir)
    gedi_df = querier.query_region_tiles(
        region_bbox=args.region_bbox,
        tile_size=0.1,
        start_time=args.start_time,
        end_time=args.end_time,
        max_agbd=500.0  # cap at 500 to remove unrealistic outliers
    )
    print(f"Retrieved {len(gedi_df)} GEDI shots across {gedi_df['tile_id'].nunique()} tiles")

    if args.train_years is not None:
        print(f"\nApplying temporal filtering: using only years {args.train_years} for training")

        if 'time' in gedi_df.columns:
            gedi_df['year'] = pd.to_datetime(gedi_df['time']).dt.year
        elif 'date_time' in gedi_df.columns:
            gedi_df['year'] = pd.to_datetime(gedi_df['date_time']).dt.year
        elif 'datetime' in gedi_df.columns:
            gedi_df['year'] = pd.to_datetime(gedi_df['datetime']).dt.year
        else:
            try:
                gedi_df['year'] = pd.to_datetime(gedi_df.index).year
            except:
                print("Warning: Could not find timestamp column for temporal filtering.")
                print(f"Available columns: {list(gedi_df.columns)}")
                print("Skipping temporal filtering.")
                args.train_years = None

        if args.train_years is not None:
            n_before = len(gedi_df)
            gedi_df = gedi_df[gedi_df['year'].isin(args.train_years)]
            n_after = len(gedi_df)
            print(f"Filtered from {n_before} to {n_after} shots ({n_after/n_before*100:.1f}% retained)")
            print(f"Shots per year: {dict(gedi_df['year'].value_counts().sort_index())}")

    print(f"\nFinal dataset: {len(gedi_df)} GEDI shots across {gedi_df['tile_id'].nunique()} tiles")
    print()

    if len(gedi_df) == 0:
        print("No GEDI data found in region. Exiting.")
        return

    print("Step 2: Extracting GeoTessera embeddings...")
    extractor = EmbeddingExtractor(
        year=args.embedding_year,
        patch_size=args.patch_size,
        embeddings_dir=args.embeddings_dir
    )
    gedi_df = extractor.extract_patches_batch(gedi_df, verbose=True)
    print()

    gedi_df = gedi_df[gedi_df['embedding_patch'].notna()]
    print(f"Retained {len(gedi_df)} shots with valid embeddings")
    print()

    with open(output_dir / 'processed_data.pkl', 'wb') as f:
        pickle.dump(gedi_df, f)

    print("Step 3: Creating spatial train/val/test split...")
    print(f"Using BufferedSpatialSplitter with buffer_size={args.buffer_size}° (~{args.buffer_size*111:.0f}km)")
    splitter = BufferedSpatialSplitter(
        gedi_df,
        buffer_size=args.buffer_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
    train_df, val_df, test_df = splitter.split()
    print()

    def prepare_for_parquet(df):
        df_copy = df.copy()
        df_copy['embedding_patch'] = df_copy['embedding_patch'].apply(
            lambda x: x.flatten().tolist() if x is not None else None
        )
        return df_copy

    prepare_for_parquet(train_df).to_parquet(output_dir / 'train_split.parquet', index=False)
    prepare_for_parquet(val_df).to_parquet(output_dir / 'val_split.parquet', index=False)
    prepare_for_parquet(test_df).to_parquet(output_dir / 'test_split.parquet', index=False)

    print(f"Saved splits to Parquet files with flattened embeddings")

    global_bounds = (
        train_df['longitude'].min(),
        train_df['latitude'].min(),
        train_df['longitude'].max(),
        train_df['latitude'].max()
    )
    print(f"Global bounds: lon [{global_bounds[0]:.4f}, {global_bounds[2]:.4f}], "
          f"lat [{global_bounds[1]:.4f}, {global_bounds[3]:.4f}]")

    config = vars(args)
    config['global_bounds'] = list(global_bounds)
    if args.train_years is not None:
        config['train_years'] = args.train_years
    save_config(config, output_dir / 'config.json')

    print("Step 4: Creating datasets...")
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
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_neural_process,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_neural_process,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_neural_process,
        num_workers=args.num_workers
    )
    print()

    print("Step 5: Initializing model...")
    print(f"Architecture mode: {args.architecture_mode}")
    model = GEDINeuralProcess(
        patch_size=args.patch_size,
        embedding_channels=128,
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
        optimizer,
        mode='min',
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience
    )

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

        # KL weight with warmup (linear from 0 to max)
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
            denormalize_for_reporting=False
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
            print(f"No improvement in validation loss for {args.early_stopping_patience} epochs")
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

    print("\nEvaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_r2_model.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_losses_dict, test_metrics = validate(
        model, test_loader, args.device, kl_weight=1.0,
        agbd_scale=args.agbd_scale,
        log_transform_agbd=args.log_transform_agbd,
        denormalize_for_reporting=False
    )

    print(f"Test Loss:        {test_losses_dict['loss']:.6e}")
    if test_metrics:
        print(f"\nLog-space metrics (aligned with training):")
        print(f"  Test Log R²:    {test_metrics.get('log_r2', 0):.4f}")
        print(f"  Test Log RMSE:  {test_metrics.get('log_rmse', 0):.4f}")
        print(f"  Test Log MAE:   {test_metrics.get('log_mae', 0):.4f}")
        print(f"\nLinear-space metrics (Mg/ha, for interpretability):")
        print(f"  Test RMSE:      {test_metrics.get('linear_rmse', 0):.2f} Mg/ha")
        print(f"  Test MAE:       {test_metrics.get('linear_mae', 0):.2f} Mg/ha")

    checkpoint['test_metrics'] = test_metrics
    torch.save(checkpoint, output_dir / 'best_r2_model.pt')
    print("Added test metrics to best model checkpoint")

    results = {
        'neural_process': {
            'train_time': training_time,
            'val_metrics': checkpoint.get('val_metrics', {}),
            'test_metrics': test_metrics
        }
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(_make_serializable(results), f, indent=2)
    print("Saved results to results.json")

    if args.generate_diagnostics:
        print("\nGenerating post-training diagnostics...")
        try:
            generate_all_diagnostics(
                model_dir=output_dir,
                device=args.device,
                n_sample_plots=args.n_diagnostic_samples
            )
        except Exception as e:
            print(f"Warning: Failed to generate diagnostics: {e}")
            print("Training completed successfully, but diagnostics could not be generated.")


if __name__ == '__main__':
    main()
