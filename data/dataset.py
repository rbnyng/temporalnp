import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
import random

from utils.normalization import normalize_coords, normalize_agbd


def compute_temporal_encoding(
    timestamps: pd.Series,
    temporal_bounds: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Compute temporal encoding for GEDI shots.

    Args:
        timestamps: Series of datetime timestamps
        temporal_bounds: (t_min, t_max) as unix timestamps for normalization.
                        If None, computed from data.

    Returns:
        Array of shape (N, 3) with [sin(2π*doy), cos(2π*doy), normalized_time]
    """
    timestamps = pd.to_datetime(timestamps)

    # Day of year normalized to [0, 1]
    day_of_year = timestamps.dt.dayofyear
    tau = day_of_year / 365.0

    # Cyclical encoding
    sin_doy = np.sin(2 * np.pi * tau)
    cos_doy = np.cos(2 * np.pi * tau)

    # Normalized timestamp within study period
    unix_time = timestamps.astype(np.int64) / 1e9  # Convert to seconds

    if temporal_bounds is None:
        t_min, t_max = unix_time.min(), unix_time.max()
    else:
        t_min, t_max = temporal_bounds

    if t_max > t_min:
        tau_hat = (unix_time - t_min) / (t_max - t_min)
    else:
        tau_hat = np.zeros_like(unix_time)

    return np.stack([sin_doy, cos_doy, tau_hat], axis=1).astype(np.float32)


class GEDINeuralProcessDataset(Dataset):
    """
    Dataset for Neural Process training with GEDI shots and embeddings.

    Each sample is a tile with multiple GEDI shots. The dataset creates
    context/target splits for Neural Process training.
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        min_shots_per_tile: int = 10,
        max_shots_per_tile: Optional[int] = None,
        context_ratio_range: Tuple[float, float] = (0.3, 0.7),
        normalize_coords: bool = True,
        normalize_agbd: bool = True,
        agbd_scale: float = 200.0,  # Typical max AGBD in Mg/ha
        log_transform_agbd: bool = True,
        augment_coords: bool = True,
        coord_noise_std: float = 0.01,
        global_bounds: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Initialize dataset.

        Args:
            data_df: DataFrame with columns: latitude, longitude, agbd, embedding_patch, tile_id
            min_shots_per_tile: Minimum number of shots per tile to include
            max_shots_per_tile: Maximum shots per tile (subsample if exceeded)
            context_ratio_range: Range of context/total ratios for training (min, max)
            normalize_coords: Normalize coordinates to [0, 1] using global bounds
            normalize_agbd: Normalize AGBD values
            agbd_scale: Scale factor for AGBD normalization
            log_transform_agbd: Apply log(1+x) transform to AGBD
            augment_coords: Add small random noise to coordinates during training
            coord_noise_std: Standard deviation of coordinate noise
            global_bounds: Global coordinate bounds (lon_min, lat_min, lon_max, lat_max).
                          If None, computed from data_df. Should be computed from training
                          data and shared across train/val/test for proper normalization.
        """
        self.data_df = data_df[data_df['embedding_patch'].notna()].copy()

        self.tiles = []
        for tile_id, group in self.data_df.groupby('tile_id'):
            if len(group) >= min_shots_per_tile:
                # Subsample if too many shots
                if max_shots_per_tile and len(group) > max_shots_per_tile:
                    group = group.sample(n=max_shots_per_tile, random_state=42)
                self.tiles.append(group)

        self.min_shots_per_tile = min_shots_per_tile
        self.max_shots_per_tile = max_shots_per_tile
        self.context_ratio_range = context_ratio_range
        self.normalize_coords = normalize_coords
        self.normalize_agbd = normalize_agbd
        self.agbd_scale = agbd_scale
        self.log_transform_agbd = log_transform_agbd
        self.augment_coords = augment_coords
        self.coord_noise_std = coord_noise_std

        if global_bounds is None:
            self.lon_min = self.data_df['longitude'].min()
            self.lon_max = self.data_df['longitude'].max()
            self.lat_min = self.data_df['latitude'].min()
            self.lat_max = self.data_df['latitude'].max()
        else:
            self.lon_min, self.lat_min, self.lon_max, self.lat_max = global_bounds

        print(f"Dataset initialized with {len(self.tiles)} tiles")
        if len(self.tiles) > 0:
            shots_per_tile = [len(t) for t in self.tiles]
            print(f"Shots per tile: min={min(shots_per_tile)}, "
                  f"max={max(shots_per_tile)}, mean={np.mean(shots_per_tile):.1f}")

    def __len__(self) -> int:
        return len(self.tiles)

    def _normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """
        Normalize coordinates to [0, 1] range using global bounds.

        This ensures the model can learn latitude-dependent patterns (e.g., climate zones)
        since coordinates are normalized consistently across all tiles.

        Args:
            coords: (N, 2) array of [lon, lat]

        Returns:
            Normalized coordinates (N, 2)
        """
        # global bounds for normalization
        global_bounds = (self.lon_min, self.lat_min, self.lon_max, self.lat_max)
        return normalize_coords(coords, global_bounds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns a dict with:
            - context_coords: (n_context, 2) coordinates [lon, lat]
            - context_embeddings: (n_context, patch_size, patch_size, 128)
            - context_agbd: (n_context, 1) AGBD values
            - target_coords: (n_target, 2) coordinates
            - target_embeddings: (n_target, patch_size, patch_size, 128)
            - target_agbd: (n_target, 1) AGBD values
        """
        tile_data = self.tiles[idx].copy()
        n_shots = len(tile_data)

        context_ratio = random.uniform(*self.context_ratio_range)
        n_context = max(1, int(n_shots * context_ratio))

        context_indices = random.sample(range(n_shots), n_context)
        target_indices = [i for i in range(n_shots) if i not in context_indices]

        tile_array = tile_data.to_numpy()
        coords = tile_data[['longitude', 'latitude']].values
        embeddings = np.stack(tile_data['embedding_patch'].values)
        agbd = tile_data['agbd'].values[:, None]

        if self.normalize_coords:
            coords = self._normalize_coordinates(coords)

        if self.augment_coords:
            coords = coords + np.random.normal(0, self.coord_noise_std, coords.shape)
            coords = np.clip(coords, 0, 1)

        if self.normalize_agbd:
            agbd = normalize_agbd(agbd, self.agbd_scale, self.log_transform_agbd)

        context_coords = coords[context_indices]
        context_embeddings = embeddings[context_indices]
        context_agbd = agbd[context_indices]

        target_coords = coords[target_indices]
        target_embeddings = embeddings[target_indices]
        target_agbd = agbd[target_indices]

        return {
            'context_coords': torch.from_numpy(context_coords).float(),
            'context_embeddings': torch.from_numpy(context_embeddings).float(),
            'context_agbd': torch.from_numpy(context_agbd).float(),
            'target_coords': torch.from_numpy(target_coords).float(),
            'target_embeddings': torch.from_numpy(target_embeddings).float(),
            'target_agbd': torch.from_numpy(target_agbd).float(),
        }


def collate_neural_process(batch):
    """
    Custom collate function for Neural Process batches.

    Handles variable numbers of context and target points across tiles.

    Args:
        batch: List of dicts from GEDINeuralProcessDataset

    Returns:
        Batched dict with lists of tensors (one per tile in batch)
    """
    # Since tiles can have different numbers of shots, return lists
    return {
        'context_coords': [item['context_coords'] for item in batch],
        'context_embeddings': [item['context_embeddings'] for item in batch],
        'context_agbd': [item['context_agbd'] for item in batch],
        'target_coords': [item['target_coords'] for item in batch],
        'target_embeddings': [item['target_embeddings'] for item in batch],
        'target_agbd': [item['target_agbd'] for item in batch],
    }


class GEDIInferenceDataset(Dataset):
    def __init__(
        self,
        context_df: pd.DataFrame,
        query_lons: np.ndarray,
        query_lats: np.ndarray,
        query_embeddings: np.ndarray,
        normalize_coords: bool = True,
        normalize_agbd: bool = True,
        agbd_scale: float = 200.0,
        log_transform_agbd: bool = True,
        global_bounds: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Initialize inference dataset.

        Args:
            context_df: DataFrame with context GEDI shots
            query_lons: Array of query longitudes
            query_lats: Array of query latitudes
            query_embeddings: Array of query embeddings (N, patch_size, patch_size, 128)
            normalize_coords: Normalize coordinates
            normalize_agbd: Normalize AGBD
            agbd_scale: AGBD scale factor
            log_transform_agbd: Apply log transform to AGBD
            global_bounds: Global coordinate bounds (lon_min, lat_min, lon_max, lat_max).
                          If None, computed from context + query data.
        """
        self.context_df = context_df[context_df['embedding_patch'].notna()].copy()
        self.query_lons = query_lons
        self.query_lats = query_lats
        self.query_embeddings = query_embeddings

        self.normalize_coords = normalize_coords
        self.normalize_agbd = normalize_agbd
        self.agbd_scale = agbd_scale
        self.log_transform_agbd = log_transform_agbd

        # global bounds for normalization
        if global_bounds is None:
            # from context + query data
            all_lons = np.concatenate([context_df['longitude'].values, query_lons])
            all_lats = np.concatenate([context_df['latitude'].values, query_lats])
            self.lon_min = all_lons.min()
            self.lon_max = all_lons.max()
            self.lat_min = all_lats.min()
            self.lat_max = all_lats.max()
        else:
            self.lon_min, self.lat_min, self.lon_max, self.lat_max = global_bounds

        self.context_coords = self.context_df[['longitude', 'latitude']].values
        self.context_embeddings = np.stack(self.context_df['embedding_patch'].values)
        self.context_agbd = self.context_df['agbd'].values[:, None]

        # Normalize context AGBD
        if self.normalize_agbd:
            self.context_agbd = normalize_agbd(self.context_agbd, self.agbd_scale, self.log_transform_agbd)

        # Store global bounds for coordinate normalization
        self.global_bounds = (self.lon_min, self.lat_min, self.lon_max, self.lat_max)

    def __len__(self) -> int:
        return len(self.query_lons)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query_coord = np.array([[self.query_lons[idx], self.query_lats[idx]]])
        query_embedding = self.query_embeddings[idx:idx+1]

        # Normalize coordinates using utility function
        if self.normalize_coords:
            context_coords_norm = normalize_coords(self.context_coords, self.global_bounds)
            query_coord_norm = normalize_coords(query_coord, self.global_bounds)
        else:
            context_coords_norm = self.context_coords
            query_coord_norm = query_coord

        return {
            'context_coords': torch.from_numpy(context_coords_norm).float(),
            'context_embeddings': torch.from_numpy(self.context_embeddings).float(),
            'context_agbd': torch.from_numpy(self.context_agbd).float(),
            'query_coord': torch.from_numpy(query_coord_norm).float(),
            'query_embedding': torch.from_numpy(query_embedding).float(),
        }


class GEDISpatiotemporalDataset(Dataset):
    """
    Dataset for Neural Process training with spatiotemporal features.

    Extends GEDINeuralProcessDataset to include temporal encoding:
    - sin(2π * day_of_year/365)
    - cos(2π * day_of_year/365)
    - normalized timestamp within study period

    This enables learning spatiotemporal covariance and forecasting.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        min_shots_per_tile: int = 10,
        max_shots_per_tile: Optional[int] = None,
        context_ratio_range: Tuple[float, float] = (0.3, 0.7),
        normalize_coords: bool = True,
        normalize_agbd: bool = True,
        agbd_scale: float = 200.0,
        log_transform_agbd: bool = True,
        augment_coords: bool = True,
        coord_noise_std: float = 0.01,
        global_bounds: Optional[Tuple[float, float, float, float]] = None,
        temporal_bounds: Optional[Tuple[float, float]] = None,
        time_column: str = 'time',
        include_temporal: bool = True
    ):
        """
        Initialize spatiotemporal dataset.

        Args:
            data_df: DataFrame with columns: latitude, longitude, agbd, embedding_patch,
                    tile_id, and a timestamp column
            min_shots_per_tile: Minimum number of shots per tile to include
            max_shots_per_tile: Maximum shots per tile (subsample if exceeded)
            context_ratio_range: Range of context/total ratios for training
            normalize_coords: Normalize coordinates to [0, 1]
            normalize_agbd: Normalize AGBD values
            agbd_scale: Scale factor for AGBD normalization
            log_transform_agbd: Apply log(1+x) transform to AGBD
            augment_coords: Add small random noise to coordinates during training
            coord_noise_std: Standard deviation of coordinate noise
            global_bounds: (lon_min, lat_min, lon_max, lat_max) for normalization
            temporal_bounds: (t_min, t_max) as unix timestamps for temporal normalization
            time_column: Name of the timestamp column in data_df
            include_temporal: If True, include temporal encoding (5D coords).
                            If False, use spatial coords only (2D coords).
        """
        self.data_df = data_df[data_df['embedding_patch'].notna()].copy()
        self.time_column = time_column
        self.include_temporal = include_temporal

        # Ensure timestamp column exists
        if time_column not in self.data_df.columns:
            raise ValueError(f"Time column '{time_column}' not found in dataframe")

        # Compute temporal encoding for all data (even if not used, for consistency)
        if temporal_bounds is None:
            timestamps = pd.to_datetime(self.data_df[time_column])
            unix_time = timestamps.astype(np.int64) / 1e9
            self.temporal_bounds = (unix_time.min(), unix_time.max())
        else:
            self.temporal_bounds = temporal_bounds

        self.data_df['temporal_encoding'] = list(compute_temporal_encoding(
            self.data_df[time_column],
            self.temporal_bounds
        ))

        # Group by tiles
        self.tiles = []
        for tile_id, group in self.data_df.groupby('tile_id'):
            if len(group) >= min_shots_per_tile:
                if max_shots_per_tile and len(group) > max_shots_per_tile:
                    group = group.sample(n=max_shots_per_tile, random_state=42)
                self.tiles.append(group)

        self.min_shots_per_tile = min_shots_per_tile
        self.max_shots_per_tile = max_shots_per_tile
        self.context_ratio_range = context_ratio_range
        self.normalize_coords = normalize_coords
        self.normalize_agbd = normalize_agbd
        self.agbd_scale = agbd_scale
        self.log_transform_agbd = log_transform_agbd
        self.augment_coords = augment_coords
        self.coord_noise_std = coord_noise_std

        if global_bounds is None:
            self.lon_min = self.data_df['longitude'].min()
            self.lon_max = self.data_df['longitude'].max()
            self.lat_min = self.data_df['latitude'].min()
            self.lat_max = self.data_df['latitude'].max()
        else:
            self.lon_min, self.lat_min, self.lon_max, self.lat_max = global_bounds

        mode = "spatiotemporal (5D)" if include_temporal else "spatial-only (2D)"
        print(f"Dataset initialized with {len(self.tiles)} tiles, coords: {mode}")
        if len(self.tiles) > 0:
            shots_per_tile = [len(t) for t in self.tiles]
            print(f"Shots per tile: min={min(shots_per_tile)}, "
                  f"max={max(shots_per_tile)}, mean={np.mean(shots_per_tile):.1f}")

    def __len__(self) -> int:
        return len(self.tiles)

    def _normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Normalize spatial coordinates to [0, 1] range."""
        global_bounds = (self.lon_min, self.lat_min, self.lon_max, self.lat_max)
        return normalize_coords(coords, global_bounds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample with spatiotemporal features.

        Returns a dict with:
            - context_coords: (n_context, D) where D=5 if include_temporal else D=2
            - context_embeddings: (n_context, patch_size, patch_size, 128)
            - context_agbd: (n_context, 1)
            - target_coords: (n_target, D)
            - target_embeddings: (n_target, patch_size, patch_size, 128)
            - target_agbd: (n_target, 1)
        """
        tile_data = self.tiles[idx].copy()
        n_shots = len(tile_data)

        context_ratio = random.uniform(*self.context_ratio_range)
        n_context = max(1, int(n_shots * context_ratio))

        context_indices = random.sample(range(n_shots), n_context)
        target_indices = [i for i in range(n_shots) if i not in context_indices]

        # Extract features
        spatial_coords = tile_data[['longitude', 'latitude']].values
        embeddings = np.stack(tile_data['embedding_patch'].values)
        agbd = tile_data['agbd'].values[:, None]

        # Normalize spatial coordinates
        if self.normalize_coords:
            spatial_coords = self._normalize_coordinates(spatial_coords)

        if self.augment_coords:
            spatial_coords = spatial_coords + np.random.normal(0, self.coord_noise_std, spatial_coords.shape)
            spatial_coords = np.clip(spatial_coords, 0, 1)

        # Build coordinate vector: spatial only or spatiotemporal
        if self.include_temporal:
            temporal_encoding = np.stack(tile_data['temporal_encoding'].values)
            coords = np.concatenate([spatial_coords, temporal_encoding], axis=1)
        else:
            coords = spatial_coords

        if self.normalize_agbd:
            agbd = normalize_agbd(agbd, self.agbd_scale, self.log_transform_agbd)

        context_coords = coords[context_indices]
        context_embeddings = embeddings[context_indices]
        context_agbd = agbd[context_indices]

        target_coords = coords[target_indices]
        target_embeddings = embeddings[target_indices]
        target_agbd = agbd[target_indices]

        return {
            'context_coords': torch.from_numpy(context_coords).float(),
            'context_embeddings': torch.from_numpy(context_embeddings).float(),
            'context_agbd': torch.from_numpy(context_agbd).float(),
            'target_coords': torch.from_numpy(target_coords).float(),
            'target_embeddings': torch.from_numpy(target_embeddings).float(),
            'target_agbd': torch.from_numpy(target_agbd).float(),
        }


class GEDICausalSpatiotemporalDataset(Dataset):
    """
    Dataset for causal spatiotemporal training.

    Instead of random context/target splits, this dataset enforces causal splits:
    - Context: all observations with t <= t_now
    - Target: random subsample of observations with t > t_now

    This trains the model to forecast future biomass from past observations.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        min_shots_per_tile: int = 10,
        max_shots_per_tile: Optional[int] = None,
        max_target_points: int = 50,
        temporal_margin_days: int = 30,
        normalize_coords: bool = True,
        normalize_agbd: bool = True,
        agbd_scale: float = 200.0,
        log_transform_agbd: bool = True,
        augment_coords: bool = True,
        coord_noise_std: float = 0.01,
        global_bounds: Optional[Tuple[float, float, float, float]] = None,
        temporal_bounds: Optional[Tuple[float, float]] = None,
        time_column: str = 'time'
    ):
        """
        Initialize causal spatiotemporal dataset.

        Args:
            data_df: DataFrame with GEDI observations
            min_shots_per_tile: Minimum shots per tile
            max_shots_per_tile: Maximum shots per tile
            max_target_points: Maximum number of target points to sample per episode
            temporal_margin_days: Margin in days from temporal boundaries for t_now sampling
            normalize_coords: Normalize coordinates
            normalize_agbd: Normalize AGBD
            agbd_scale: AGBD normalization scale
            log_transform_agbd: Apply log transform
            augment_coords: Add coordinate noise
            coord_noise_std: Coordinate noise std
            global_bounds: Spatial bounds for normalization
            temporal_bounds: Temporal bounds for normalization
            time_column: Timestamp column name
        """
        self.data_df = data_df[data_df['embedding_patch'].notna()].copy()
        self.time_column = time_column
        self.max_target_points = max_target_points
        self.temporal_margin_days = temporal_margin_days

        if time_column not in self.data_df.columns:
            raise ValueError(f"Time column '{time_column}' not found in dataframe")

        # Convert timestamps
        self.data_df['_timestamp'] = pd.to_datetime(self.data_df[time_column])
        self.data_df['_unix_time'] = self.data_df['_timestamp'].astype(np.int64) / 1e9

        # Compute temporal bounds
        if temporal_bounds is None:
            self.temporal_bounds = (
                self.data_df['_unix_time'].min(),
                self.data_df['_unix_time'].max()
            )
        else:
            self.temporal_bounds = temporal_bounds

        # Precompute temporal encoding for all data
        self.data_df['temporal_encoding'] = list(compute_temporal_encoding(
            self.data_df[time_column],
            self.temporal_bounds
        ))

        # Group by tiles
        self.tiles = []
        for tile_id, group in self.data_df.groupby('tile_id'):
            if len(group) >= min_shots_per_tile:
                if max_shots_per_tile and len(group) > max_shots_per_tile:
                    group = group.sample(n=max_shots_per_tile, random_state=42)
                # Sort by time for causal splitting
                group = group.sort_values('_unix_time')
                self.tiles.append(group)

        self.min_shots_per_tile = min_shots_per_tile
        self.max_shots_per_tile = max_shots_per_tile
        self.normalize_coords = normalize_coords
        self.normalize_agbd = normalize_agbd
        self.agbd_scale = agbd_scale
        self.log_transform_agbd = log_transform_agbd
        self.augment_coords = augment_coords
        self.coord_noise_std = coord_noise_std

        if global_bounds is None:
            self.lon_min = self.data_df['longitude'].min()
            self.lon_max = self.data_df['longitude'].max()
            self.lat_min = self.data_df['latitude'].min()
            self.lat_max = self.data_df['latitude'].max()
        else:
            self.lon_min, self.lat_min, self.lon_max, self.lat_max = global_bounds

        print(f"Causal spatiotemporal dataset initialized with {len(self.tiles)} tiles")

    def __len__(self) -> int:
        return len(self.tiles)

    def _normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        global_bounds = (self.lon_min, self.lat_min, self.lon_max, self.lat_max)
        return normalize_coords(coords, global_bounds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a causally-split training sample.

        Samples a random t_now and splits:
        - Context: all points with t <= t_now
        - Target: random subsample of points with t > t_now
        """
        tile_data = self.tiles[idx].copy()

        # Get temporal range for this tile
        unix_times = tile_data['_unix_time'].values
        t_min_tile = unix_times.min()
        t_max_tile = unix_times.max()

        # Margin in seconds
        margin = self.temporal_margin_days * 24 * 3600

        # Sample t_now
        t_now_min = t_min_tile + margin
        t_now_max = t_max_tile - margin

        if t_now_max <= t_now_min:
            # Not enough temporal range, fall back to median split
            t_now = np.median(unix_times)
        else:
            t_now = random.uniform(t_now_min, t_now_max)

        # Split into context (past) and potential targets (future)
        context_mask = unix_times <= t_now
        target_mask = unix_times > t_now

        context_data = tile_data[context_mask]
        potential_targets = tile_data[target_mask]

        # Handle edge cases
        if len(context_data) == 0 or len(potential_targets) == 0:
            # Fall back to 50/50 split by time
            mid_idx = len(tile_data) // 2
            context_data = tile_data.iloc[:max(1, mid_idx)]
            potential_targets = tile_data.iloc[mid_idx:]

        # Subsample targets
        if len(potential_targets) > self.max_target_points:
            target_data = potential_targets.sample(n=self.max_target_points)
        else:
            target_data = potential_targets

        # Extract features for context
        context_spatial = context_data[['longitude', 'latitude']].values
        context_temporal = np.stack(context_data['temporal_encoding'].values)
        context_embeddings = np.stack(context_data['embedding_patch'].values)
        context_agbd = context_data['agbd'].values[:, None]

        # Extract features for targets
        target_spatial = target_data[['longitude', 'latitude']].values
        target_temporal = np.stack(target_data['temporal_encoding'].values)
        target_embeddings = np.stack(target_data['embedding_patch'].values)
        target_agbd = target_data['agbd'].values[:, None]

        # Normalize spatial coordinates
        if self.normalize_coords:
            context_spatial = self._normalize_coordinates(context_spatial)
            target_spatial = self._normalize_coordinates(target_spatial)

        if self.augment_coords:
            context_spatial = context_spatial + np.random.normal(
                0, self.coord_noise_std, context_spatial.shape)
            context_spatial = np.clip(context_spatial, 0, 1)

        # Combine spatial and temporal
        context_coords = np.concatenate([context_spatial, context_temporal], axis=1)
        target_coords = np.concatenate([target_spatial, target_temporal], axis=1)

        # Normalize AGBD
        if self.normalize_agbd:
            context_agbd = normalize_agbd(context_agbd, self.agbd_scale, self.log_transform_agbd)
            target_agbd = normalize_agbd(target_agbd, self.agbd_scale, self.log_transform_agbd)

        return {
            'context_coords': torch.from_numpy(context_coords).float(),
            'context_embeddings': torch.from_numpy(context_embeddings).float(),
            'context_agbd': torch.from_numpy(context_agbd).float(),
            'target_coords': torch.from_numpy(target_coords).float(),
            'target_embeddings': torch.from_numpy(target_embeddings).float(),
            'target_agbd': torch.from_numpy(target_agbd).float(),
        }
