import numpy as np
from geotessera import GeoTessera
from typing import Optional, Tuple, Dict
import pandas as pd
from pyproj import Transformer
from tqdm import tqdm
import hashlib
import json
from pathlib import Path


class EmbeddingExtractor:
    def __init__(
        self,
        year: int = 2024,
        patch_size: int = 3,
        embeddings_dir: Optional[str] = None
    ):
        """
        Initialize embedding extractor.

        Args:
            year: Year of embeddings to use (2017-2024)
            patch_size: Size of patch to extract around each shot (e.g., 3 = 3x3 = 30m x 30m)
            embeddings_dir: Directory where geotessera will store/read embedding tiles.
                          If provided, geotessera will reuse existing tiles from this directory
                          and only download missing ones. Expected structure:
                          embeddings/{year}/grid_{lon}_{lat}.npy and _scales.npy
                          landmasks/landmask_{lon}_{lat}.tif
        """
        # Let geotessera handle disk storage with embeddings_dir
        self.gt = GeoTessera(embeddings_dir=embeddings_dir)
        self.year = year
        self.patch_size = patch_size

        # in-memory cache for performance (avoids re-reading from disk)
        self.tile_cache: Dict[Tuple[float, float], Tuple[np.ndarray, object, object]] = {}

        # Cache for coordinate transformers to avoid repeated creation
        self.transformer_cache: Dict[str, Transformer] = {}

    def set_year(self, year: int) -> None:
        """
        Change the year for embedding extraction.

        This allows reusing the same GeoTessera instance across multiple years,
        avoiding repeated registry loading and initialization.

        Args:
            year: New year to use for embeddings (2017-2024)
        """
        if year != self.year:
            self.year = year
            # Clear tile cache since tiles are year-specific
            self.tile_cache.clear()

    def _get_tile_coords(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        Get tile center coordinates for a given point.

        GeoTessera tiles are 0.1° x 0.1° centered at (lon, lat) where
        lon and lat are multiples of 0.05 offset by 0.05.

        Args:
            lon: Longitude
            lat: Latitude

        Returns:
            (tile_lon, tile_lat) center coordinates
        """
        # Round to nearest 0.1 degree grid aligned at 0.05, 0.15, 0.25, etc.
        tile_lon = round((lon - 0.05) / 0.1) * 0.1 + 0.05
        tile_lat = round((lat - 0.05) / 0.1) * 0.1 + 0.05
        return tile_lon, tile_lat

    def _load_tile(
        self,
        tile_lon: float,
        tile_lat: float
    ) -> Optional[Tuple[np.ndarray, object, object]]:
        """
        Load a tile from memory cache or fetch from geotessera.

        GeoTessera v0.7.0+ handles disk caching automatically when embeddings_dir is set,
        so we only maintain an in-memory cache for performance.

        Args:
            tile_lon: Tile center longitude
            tile_lat: Tile center latitude

        Returns:
            (embedding, crs, transform) or None if tile unavailable
        """
        tile_key = (tile_lon, tile_lat)

        # Check memory cache
        if tile_key in self.tile_cache:
            return self.tile_cache[tile_key]

        # Fetch from GeoTessera (will use embeddings_dir if set, or download to temp)
        try:
            embedding, crs, transform = self.gt.fetch_embedding(
                lon=tile_lon,
                lat=tile_lat,
                year=self.year
            )

            tile_data = (embedding, crs, transform)
            self.tile_cache[tile_key] = tile_data

            return tile_data

        except Exception as e:
            print(f"Warning: Could not fetch tile at ({tile_lon}, {tile_lat}): {e}")
            return None

    def _lonlat_to_pixel(
        self,
        lon: float,
        lat: float,
        transform,
        crs
    ) -> Tuple[int, int]:
        """
        Convert lon/lat to pixel coordinates using affine transform.

        Args:
            lon: Longitude (WGS84)
            lat: Latitude (WGS84)
            transform: Rasterio affine transform
            crs: Target CRS of the raster

        Returns:
            (row, col) pixel indices
        """
        # Get CRS code as string
        crs_str = str(crs)

        # Get or create transformer from WGS84 to tile CRS
        if crs_str not in self.transformer_cache:
            self.transformer_cache[crs_str] = Transformer.from_crs(
                "EPSG:4326",  # WGS84 (lon/lat)
                crs,          # Tile CRS (e.g., UTM)
                always_xy=True
            )

        transformer = self.transformer_cache[crs_str]

        # Transform lon/lat to tile CRS coordinates
        x, y = transformer.transform(lon, lat)

        # Apply inverse affine transform: (x, y) -> (col, row)
        col, row = ~transform * (x, y)

        return int(row), int(col)

    def extract_patch(
        self,
        lon: float,
        lat: float
    ) -> Optional[np.ndarray]:
        """
        Extract a patch around a point location.

        Args:
            lon: Longitude of center point
            lat: Latitude of center point

        Returns:
            Patch array of shape (patch_size, patch_size, 128) or None if extraction fails
        """
        # Get tile coordinates
        tile_lon, tile_lat = self._get_tile_coords(lon, lat)

        # Load tile
        tile_data = self._load_tile(tile_lon, tile_lat)
        if tile_data is None:
            return None

        embedding, crs, transform = tile_data

        # Convert to pixel coordinates
        try:
            row, col = self._lonlat_to_pixel(lon, lat, transform, crs)
        except Exception as e:
            print(f"Warning: Could not convert coordinates ({lon}, {lat}): {e}")
            return None

        # Extract patch
        height, width, channels = embedding.shape
        half_patch = self.patch_size // 2

        # Check bounds
        if (row - half_patch < 0 or row + half_patch + 1 > height or
            col - half_patch < 0 or col + half_patch + 1 > width):
            return None

        patch = embedding[
            row - half_patch:row + half_patch + 1,
            col - half_patch:col + half_patch + 1,
            :
        ]

        return patch

    def extract_patches_batch(
        self,
        gedi_df: pd.DataFrame,
        verbose: bool = True,
        desc: str = None,
        cache_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract patches for all GEDI shots in a DataFrame.

        Args:
            gedi_df: DataFrame with 'longitude', 'latitude' columns
            verbose: Show progress bar
            desc: Description for progress bar (default: "Extracting embeddings")
            cache_dir: Directory for caching extracted embedding-shot pairs.
                      If provided, will load from cache if available, otherwise
                      extract and save to cache for future runs.

        Returns:
            DataFrame with added 'embedding_patch' column (None for failed extractions)
        """
        # Check cache first
        if cache_dir is not None:
            cache_key = self._generate_extraction_cache_key(gedi_df)
            cache_path = self._get_extraction_cache_path(cache_dir, cache_key)

            if cache_path.exists():
                if verbose:
                    print(f"  Loading cached embeddings from {cache_path}")
                cached_df = self._load_cached_extractions(cache_path, gedi_df)
                if cached_df is not None:
                    successful = cached_df['embedding_patch'].notna().sum()
                    if verbose:
                        print(f"  {successful}/{len(cached_df)} shots with valid embeddings "
                              f"({100*successful/len(cached_df):.1f}%) [from cache]")
                    return cached_df

        # Extract embeddings
        patches = []
        successful = 0

        if desc is None:
            desc = f"Extracting embeddings (year {self.year})"

        iterator = gedi_df.iterrows()
        if verbose:
            iterator = tqdm(iterator, total=len(gedi_df), desc=desc)

        for idx, row in iterator:
            patch = self.extract_patch(row['longitude'], row['latitude'])
            patches.append(patch)
            if patch is not None:
                successful += 1

        gedi_df = gedi_df.copy()
        gedi_df['embedding_patch'] = patches

        if verbose:
            print(f"  {successful}/{len(gedi_df)} shots with valid embeddings "
                  f"({100*successful/len(gedi_df):.1f}%)")

        # Save to cache
        if cache_dir is not None:
            self._save_extractions_to_cache(cache_path, gedi_df)

        return gedi_df

    def extract_dense_grid(
        self,
        tile_lon: float,
        tile_lat: float,
        spacing: float = 0.001  # ~100m at equator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract patches on a dense grid for inference/prediction.

        Args:
            tile_lon: Tile center longitude
            tile_lat: Tile center latitude
            spacing: Grid spacing in degrees

        Returns:
            (longitudes, latitudes, patches) where patches is (N, patch_size, patch_size, 128)
        """
        tile_data = self._load_tile(tile_lon, tile_lat)
        if tile_data is None:
            return None, None, None

        embedding, crs, transform = tile_data

        # Create grid of points
        half_size = 0.05  # Tile is 0.1° wide
        lons = np.arange(tile_lon - half_size, tile_lon + half_size, spacing)
        lats = np.arange(tile_lat - half_size, tile_lat + half_size, spacing)

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        lon_flat = lon_grid.flatten()
        lat_flat = lat_grid.flatten()

        patches = []
        valid_lons = []
        valid_lats = []

        for lon, lat in zip(lon_flat, lat_flat):
            patch = self.extract_patch(lon, lat)
            if patch is not None:
                patches.append(patch)
                valid_lons.append(lon)
                valid_lats.append(lat)

        if len(patches) == 0:
            return None, None, None

        return (
            np.array(valid_lons),
            np.array(valid_lats),
            np.array(patches)
        )

    def clear_cache(self):
        """Clear in-memory tile cache."""
        self.tile_cache.clear()

    def _generate_extraction_cache_key(
        self,
        gedi_df: pd.DataFrame,
    ) -> str:
        """
        Generate a cache key for extracted embedding-shot pairs.

        The key is based on:
        - Shot locations (lat/lon rounded to 6 decimals)
        - Extraction parameters (year, patch_size)

        Args:
            gedi_df: DataFrame with GEDI shots

        Returns:
            MD5 hash string
        """
        # Create a deterministic representation of the shots
        # Round coordinates to avoid floating point issues
        coords = gedi_df[['longitude', 'latitude']].round(6).values
        coords_bytes = coords.tobytes()

        # Include extraction parameters
        params = {
            'year': self.year,
            'patch_size': self.patch_size,
            'n_shots': len(gedi_df),
        }
        params_str = json.dumps(params, sort_keys=True)

        # Combine and hash
        combined = coords_bytes + params_str.encode()
        return hashlib.md5(combined).hexdigest()

    def _get_extraction_cache_path(
        self,
        cache_dir: str,
        cache_key: str
    ) -> Path:
        """Get the path for an extraction cache file."""
        cache_path = Path(cache_dir) / 'extracted_pairs'
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path / f'embeddings_{cache_key}.npz'

    def _load_cached_extractions(
        self,
        cache_path: Path,
        gedi_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Load cached embedding-shot pairs.

        Args:
            cache_path: Path to cache file
            gedi_df: Original DataFrame to attach embeddings to

        Returns:
            DataFrame with embedding_patch column, or None if cache invalid
        """
        try:
            data = np.load(cache_path, allow_pickle=True)
            patches = data['patches']

            # Verify alignment
            if len(patches) != len(gedi_df):
                print(f"  Cache size mismatch: {len(patches)} vs {len(gedi_df)} shots")
                return None

            # Verify coordinates hash if available (for cache integrity)
            if 'coords_hash' in data:
                expected_hash = hashlib.md5(
                    gedi_df[['longitude', 'latitude']].round(6).values.tobytes()
                ).hexdigest()
                if str(data['coords_hash']) != expected_hash:
                    print(f"  Cache coords mismatch, re-extracting")
                    return None

            # Reconstruct DataFrame
            gedi_df = gedi_df.copy()
            gedi_df['embedding_patch'] = [
                p if not np.isnan(p).all() else None
                for p in patches
            ]

            return gedi_df

        except Exception as e:
            print(f"  Failed to load cache: {e}")
            return None

    def _save_extractions_to_cache(
        self,
        cache_path: Path,
        gedi_df: pd.DataFrame
    ) -> None:
        """
        Save extracted embedding-shot pairs to cache.

        Args:
            cache_path: Path to cache file
            gedi_df: DataFrame with embedding_patch column
        """
        # Convert patches to numpy array (use NaN for None patches)
        patches = []
        patch_shape = None

        # First pass: find valid patch shape
        for patch in gedi_df['embedding_patch']:
            if patch is not None:
                patch_shape = patch.shape
                break

        if patch_shape is None:
            print("  Warning: No valid patches to cache")
            return

        # Second pass: convert to array with NaN for missing
        for patch in gedi_df['embedding_patch']:
            if patch is not None:
                patches.append(patch)
            else:
                patches.append(np.full(patch_shape, np.nan))

        patches_array = np.array(patches)

        # Save to cache with metadata for verification
        coords_hash = hashlib.md5(
            gedi_df[['longitude', 'latitude']].round(6).values.tobytes()
        ).hexdigest()

        np.savez_compressed(
            cache_path,
            patches=patches_array,
            year=self.year,
            patch_size=self.patch_size,
            coords_hash=coords_hash
        )

        # Report cache size
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"  Saved extraction cache ({cache_size_mb:.1f} MB): {cache_path}")
