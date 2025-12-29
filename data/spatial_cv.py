import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Set
from sklearn.model_selection import KFold
import random


class SpatialTileSplitter:
    def __init__(
        self,
        data_df: pd.DataFrame,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ):
        self.data_df = data_df
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        self.tile_ids = data_df['tile_id'].unique()
        self.n_tiles = len(self.tile_ids)

        random.seed(random_state)
        np.random.seed(random_state)

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Sort tiles to consistent order before shuffling
        sorted_tiles = np.sort(self.tile_ids)

        # Shuffle tiles
        shuffled_tiles = sorted_tiles.copy()
        np.random.shuffle(shuffled_tiles)

        # split sizes
        n_test = max(1, int(self.n_tiles * self.test_ratio)) if self.test_ratio > 0 else 0
        n_val = max(1, int(self.n_tiles * self.val_ratio)) if self.val_ratio > 0 else 0
        n_train = self.n_tiles - n_test - n_val

        # Split tiles
        train_tiles = shuffled_tiles[:n_train]
        val_tiles = shuffled_tiles[n_train:n_train + n_val]
        test_tiles = shuffled_tiles[n_train + n_val:]

        # dataframe splits
        train_df = self.data_df[self.data_df['tile_id'].isin(train_tiles)]
        val_df = self.data_df[self.data_df['tile_id'].isin(val_tiles)]
        test_df = self.data_df[self.data_df['tile_id'].isin(test_tiles)]

        print(f"Spatial split created:")
        print(f"  Train: {len(train_tiles)} tiles, {len(train_df)} shots")
        print(f"  Val:   {len(val_tiles)} tiles, {len(val_df)} shots")
        print(f"  Test:  {len(test_tiles)} tiles, {len(test_df)} shots")

        return train_df, val_df, test_df

    def k_fold_split(self, n_folds: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)

        splits = []
        for train_idx, val_idx in kf.split(self.tile_ids):
            train_tiles = self.tile_ids[train_idx]
            val_tiles = self.tile_ids[val_idx]

            train_df = self.data_df[self.data_df['tile_id'].isin(train_tiles)]
            val_df = self.data_df[self.data_df['tile_id'].isin(val_tiles)]

            splits.append((train_df, val_df))

        print(f"Created {n_folds}-fold spatial CV")
        for i, (train_df, val_df) in enumerate(splits):
            print(f"  Fold {i+1}: Train {len(train_df)} shots, Val {len(val_df)} shots")

        return splits


class BufferedSpatialSplitter:
    def __init__(
        self,
        data_df: pd.DataFrame,
        buffer_size: float = 0.1,  # degrees
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ):
        """
        Initialize buffered splitter.

        Args:
            data_df: DataFrame with 'tile_lon', 'tile_lat', 'tile_id' columns
            buffer_size: Buffer distance in degrees between splits
            val_ratio: Fraction of tiles for validation
            test_ratio: Fraction of tiles for test
            random_state: Random seed
        """
        self.data_df = data_df
        self.buffer_size = buffer_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        random.seed(random_state)
        np.random.seed(random_state)

        # tile centers
        self.tile_info = (
            data_df[['tile_id', 'tile_lon', 'tile_lat']]
            .drop_duplicates()
            .set_index('tile_id')
        )

    def _compute_distance(self, tile1: str, tile2: str) -> float:
        lon1, lat1 = self.tile_info.loc[tile1, ['tile_lon', 'tile_lat']]
        lon2, lat2 = self.tile_info.loc[tile2, ['tile_lon', 'tile_lat']]

        # Euclidean distance (good enough for small regions)
        return np.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        tile_ids = np.sort(self.tile_info.index.values)
        n_tiles = len(tile_ids)

        np.random.shuffle(tile_ids)
        n_test = max(1, int(n_tiles * self.test_ratio)) if self.test_ratio > 0 else 0
        test_tiles = tile_ids[:n_test]

        # Find tiles within buffer of test tiles
        test_buffer_tiles = set()
        for test_tile in test_tiles:
            for tile in tile_ids:
                if tile != test_tile:
                    dist = self._compute_distance(test_tile, tile)
                    if dist < self.buffer_size:
                        test_buffer_tiles.add(tile)

        # Remaining tiles excluding buffer
        remaining_tiles = [t for t in tile_ids if t not in test_tiles and t not in test_buffer_tiles]

        if len(remaining_tiles) < 2:
            print("Warning: Not enough tiles for buffered split, falling back to simple split")
            return SpatialTileSplitter(
                self.data_df, self.val_ratio, self.test_ratio, self.random_state
            ).split()

        # Select validation tiles from remaining
        if self.val_ratio > 0:
            n_val = max(1, int(len(remaining_tiles) * self.val_ratio / (1 - self.test_ratio)))
            n_val = min(n_val, len(remaining_tiles) - 1)
        else:
            n_val = 0
        val_tiles = remaining_tiles[:n_val]

        # Find tiles within buffer of val tiles
        val_buffer_tiles = set()
        for val_tile in val_tiles:
            for tile in remaining_tiles:
                if tile != val_tile:
                    dist = self._compute_distance(val_tile, tile)
                    if dist < self.buffer_size:
                        val_buffer_tiles.add(tile)

        # Train tiles
        train_tiles = [
            t for t in remaining_tiles
            if t not in val_tiles and t not in val_buffer_tiles
        ]

        train_df = self.data_df[self.data_df['tile_id'].isin(train_tiles)]
        val_df = self.data_df[self.data_df['tile_id'].isin(val_tiles)]
        test_df = self.data_df[self.data_df['tile_id'].isin(test_tiles)]

        print(f"Buffered spatial split created (buffer={self.buffer_size}°):")
        print(f"  Train: {len(train_tiles)} tiles, {len(train_df)} shots")
        print(f"  Val:   {len(val_tiles)} tiles, {len(val_df)} shots")
        print(f"  Test:  {len(test_tiles)} tiles, {len(test_df)} shots")
        print(f"  Excluded (buffers): {len(test_buffer_tiles) + len(val_buffer_tiles)} tiles")

        return train_df, val_df, test_df


class SpatiotemporalSplitter:
    """
    Spatiotemporal splitter for fire detection / temporal holdout experiments.

    This splitter ensures:
    1. Spatial blocks are assigned once and remain consistent across years
    2. Test blocks are NEVER seen in training (any year)
    3. Test year is completely held out from training

    Split structure:
        Train: train_blocks × train_years
        Val: val_blocks × train_years
        Test: test_blocks × test_year

    This is ideal for evaluating temporal generalization (e.g., detecting
    a 2021 fire when training on 2019-2020 and 2022-2023).
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        train_years: List[int],
        test_year: int,
        buffer_size: float = 0.1,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
        time_column: str = 'time'
    ):
        """
        Initialize spatiotemporal splitter.

        Args:
            data_df: DataFrame with GEDI observations (must have tile_id, time columns)
            train_years: List of years to use for training (e.g., [2019, 2020, 2022, 2023])
            test_year: Year to hold out for testing (e.g., 2021)
            buffer_size: Spatial buffer in degrees between train/test blocks
            val_ratio: Fraction of tiles for validation
            test_ratio: Fraction of tiles for test
            random_state: Random seed for reproducibility
            time_column: Name of timestamp column
        """
        self.data_df = data_df.copy()
        self.train_years = train_years
        self.test_year = test_year
        self.buffer_size = buffer_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.time_column = time_column

        # Extract year from timestamp
        if 'year' not in self.data_df.columns:
            if time_column in self.data_df.columns:
                self.data_df['year'] = pd.to_datetime(self.data_df[time_column]).dt.year
            else:
                raise ValueError(f"Time column '{time_column}' not found and 'year' column missing")

        # Validate years
        available_years = set(self.data_df['year'].unique())
        missing_train = set(train_years) - available_years
        if missing_train:
            print(f"Warning: Train years {missing_train} not found in data")
        if test_year not in available_years:
            print(f"Warning: Test year {test_year} not found in data")

        random.seed(random_state)
        np.random.seed(random_state)

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create spatiotemporal train/val/test split.

        Returns:
            (train_df, val_df, test_df)
        """
        # Step 1: Get all unique tiles (from all years)
        tile_info = (
            self.data_df[['tile_id', 'tile_lon', 'tile_lat']]
            .drop_duplicates()
            .set_index('tile_id')
        )
        tile_ids = np.sort(tile_info.index.values)
        n_tiles = len(tile_ids)

        # Step 2: Shuffle and assign to train/val/test blocks
        np.random.shuffle(tile_ids)

        n_test = max(1, int(n_tiles * self.test_ratio)) if self.test_ratio > 0 else 0
        test_tiles_initial = set(tile_ids[:n_test])

        # Step 3: Find buffer tiles around test tiles
        test_buffer_tiles = set()
        for test_tile in test_tiles_initial:
            test_lon, test_lat = tile_info.loc[test_tile, ['tile_lon', 'tile_lat']]
            for tile in tile_ids:
                if tile not in test_tiles_initial:
                    tile_lon, tile_lat = tile_info.loc[tile, ['tile_lon', 'tile_lat']]
                    dist = np.sqrt((tile_lon - test_lon)**2 + (tile_lat - test_lat)**2)
                    if dist < self.buffer_size:
                        test_buffer_tiles.add(tile)

        # Step 4: Remaining tiles (excluding test and buffer)
        remaining_tiles = [t for t in tile_ids if t not in test_tiles_initial and t not in test_buffer_tiles]

        if len(remaining_tiles) < 2:
            print("Warning: Not enough tiles after buffer exclusion, reducing buffer")
            remaining_tiles = [t for t in tile_ids if t not in test_tiles_initial]

        # Step 5: Split remaining into train/val
        if self.val_ratio > 0:
            n_val = max(1, int(len(remaining_tiles) * self.val_ratio / (1 - self.test_ratio)))
            n_val = min(n_val, len(remaining_tiles) - 1)
        else:
            n_val = 0

        val_tiles = set(remaining_tiles[:n_val])

        # Find buffer tiles around val tiles
        val_buffer_tiles = set()
        for val_tile in val_tiles:
            val_lon, val_lat = tile_info.loc[val_tile, ['tile_lon', 'tile_lat']]
            for tile in remaining_tiles:
                if tile not in val_tiles:
                    tile_lon, tile_lat = tile_info.loc[tile, ['tile_lon', 'tile_lat']]
                    dist = np.sqrt((tile_lon - val_lon)**2 + (tile_lat - val_lat)**2)
                    if dist < self.buffer_size:
                        val_buffer_tiles.add(tile)

        train_tiles = set(t for t in remaining_tiles if t not in val_tiles and t not in val_buffer_tiles)
        test_tiles = test_tiles_initial

        # Step 6: Apply temporal splits
        # Train: train_tiles × train_years
        # Val: val_tiles × train_years
        # Test: test_tiles × test_year
        train_df = self.data_df[
            (self.data_df['tile_id'].isin(train_tiles)) &
            (self.data_df['year'].isin(self.train_years))
        ]

        val_df = self.data_df[
            (self.data_df['tile_id'].isin(val_tiles)) &
            (self.data_df['year'].isin(self.train_years))
        ]

        test_df = self.data_df[
            (self.data_df['tile_id'].isin(test_tiles)) &
            (self.data_df['year'] == self.test_year)
        ]

        print(f"Spatiotemporal split created:")
        print(f"  Train years: {self.train_years}")
        print(f"  Test year: {self.test_year}")
        print(f"  Buffer size: {self.buffer_size}°")
        print(f"  Spatial blocks: {len(train_tiles)} train, {len(val_tiles)} val, {len(test_tiles)} test")
        print(f"  Excluded (buffers): {len(test_buffer_tiles) + len(val_buffer_tiles)} tiles")
        print(f"  Train: {len(train_df)} shots")
        print(f"  Val: {len(val_df)} shots")
        print(f"  Test: {len(test_df)} shots")

        # Store tile assignments for analysis
        self.train_tiles = train_tiles
        self.val_tiles = val_tiles
        self.test_tiles = test_tiles

        return train_df, val_df, test_df

    def get_tile_assignments(self) -> Dict[str, set]:
        """Get the tile assignments after split() has been called."""
        return {
            'train': self.train_tiles,
            'val': self.val_tiles,
            'test': self.test_tiles
        }


def analyze_spatial_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict:
    def get_extent(df):
        return {
            'lon_range': (df['longitude'].min(), df['longitude'].max()),
            'lat_range': (df['latitude'].min(), df['latitude'].max()),
            'center': (df['longitude'].mean(), df['latitude'].mean())
        }

    analysis = {
        'train': {
            'n_tiles': df['tile_id'].nunique(),
            'n_shots': len(train_df),
            'extent': get_extent(train_df),
            'agbd_stats': {
                'mean': train_df['agbd'].mean(),
                'std': train_df['agbd'].std(),
                'min': train_df['agbd'].min(),
                'max': train_df['agbd'].max()
            }
        },
        'val': {
            'n_tiles': val_df['tile_id'].nunique(),
            'n_shots': len(val_df),
            'extent': get_extent(val_df),
            'agbd_stats': {
                'mean': val_df['agbd'].mean(),
                'std': val_df['agbd'].std(),
                'min': val_df['agbd'].min(),
                'max': val_df['agbd'].max()
            }
        },
        'test': {
            'n_tiles': test_df['tile_id'].nunique(),
            'n_shots': len(test_df),
            'extent': get_extent(test_df),
            'agbd_stats': {
                'mean': test_df['agbd'].mean(),
                'std': test_df['agbd'].std(),
                'min': test_df['agbd'].min(),
                'max': test_df['agbd'].max()
            }
        }
    }

    return analysis
