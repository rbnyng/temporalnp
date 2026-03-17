import numpy as np
from typing import Optional, Tuple, Dict
import pandas as pd
from pyproj import Transformer
from tqdm import tqdm
import hashlib
import json
from pathlib import Path
from abc import ABC, abstractmethod


class BaseEmbeddingExtractor(ABC):
    """
    Abstract base class for embedding extractors.

    Provides a common interface for extracting embedding patches from different
    sources (GeoTessera, AlphaEarth, etc.) at GEDI shot locations.
    """

    def __init__(
        self,
        year: int,
        patch_size: int = 3,
        embedding_channels: int = 128,
    ):
        self.year = year
        self.patch_size = patch_size
        self.embedding_channels = embedding_channels
        self.tile_cache: Dict[Tuple[float, float], Tuple[np.ndarray, object, object]] = {}
        self.transformer_cache: Dict[str, Transformer] = {}

    @property
    def source_name(self) -> str:
        """Return a human-readable name for this embedding source."""
        return self.__class__.__name__

    def set_year(self, year: int) -> None:
        if year != self.year:
            self.year = year
            self.tile_cache.clear()

    @abstractmethod
    def _get_tile_coords(self, lon: float, lat: float) -> Tuple[float, float]:
        """Get tile center coordinates for a given point."""
        ...

    @abstractmethod
    def _load_tile(
        self, tile_lon: float, tile_lat: float
    ) -> Optional[Tuple[np.ndarray, object, object]]:
        """Load a tile (embedding, crs, transform) or None if unavailable."""
        ...

    def _lonlat_to_pixel(
        self, lon: float, lat: float, transform, crs
    ) -> Tuple[int, int]:
        crs_str = str(crs)
        if crs_str not in self.transformer_cache:
            self.transformer_cache[crs_str] = Transformer.from_crs(
                "EPSG:4326", crs, always_xy=True
            )
        transformer = self.transformer_cache[crs_str]
        x, y = transformer.transform(lon, lat)
        col, row = ~transform * (x, y)
        return int(row), int(col)

    def extract_patch(self, lon: float, lat: float) -> Optional[np.ndarray]:
        tile_lon, tile_lat = self._get_tile_coords(lon, lat)
        tile_data = self._load_tile(tile_lon, tile_lat)
        if tile_data is None:
            return None

        embedding, crs, transform = tile_data
        try:
            row, col = self._lonlat_to_pixel(lon, lat, transform, crs)
        except Exception as e:
            print(f"Warning: Could not convert coordinates ({lon}, {lat}): {e}")
            return None

        height, width, channels = embedding.shape
        half_patch = self.patch_size // 2

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

        patches = []
        successful = 0

        if desc is None:
            desc = f"Extracting {self.source_name} embeddings (year {self.year})"

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

        if cache_dir is not None:
            self._save_extractions_to_cache(cache_path, gedi_df)

        return gedi_df

    def extract_dense_grid(
        self,
        tile_lon: float,
        tile_lat: float,
        spacing: float = 0.001
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tile_data = self._load_tile(tile_lon, tile_lat)
        if tile_data is None:
            return None, None, None

        half_size = 0.05
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
        self.tile_cache.clear()

    def _generate_extraction_cache_key(self, gedi_df: pd.DataFrame) -> str:
        coords = gedi_df[['longitude', 'latitude']].round(6).values
        sorted_indices = np.lexsort((coords[:, 1], coords[:, 0]))
        coords_sorted = coords[sorted_indices]
        coords_bytes = coords_sorted.tobytes()

        params = {
            'year': self.year,
            'patch_size': self.patch_size,
            'n_shots': len(gedi_df),
            'source': self.source_name,
        }
        params_str = json.dumps(params, sort_keys=True)

        combined = coords_bytes + params_str.encode()
        return hashlib.md5(combined).hexdigest()

    def _get_extraction_cache_path(self, cache_dir: str, cache_key: str) -> Path:
        cache_path = Path(cache_dir) / 'extracted_pairs'
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path / f'embeddings_{cache_key}.npz'

    def _load_cached_extractions(
        self, cache_path: Path, gedi_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        try:
            data = np.load(cache_path, allow_pickle=True)
            patches = data['patches']

            if len(patches) != len(gedi_df):
                print(f"  Cache size mismatch: {len(patches)} vs {len(gedi_df)} shots")
                return None

            coords = gedi_df[['longitude', 'latitude']].round(6).values
            sorted_indices = np.lexsort((coords[:, 1], coords[:, 0]))

            if 'coords_hash' in data:
                coords_sorted = coords[sorted_indices]
                expected_hash = hashlib.md5(coords_sorted.tobytes()).hexdigest()
                if str(data['coords_hash']) != expected_hash:
                    print(f"  Cache coords mismatch, re-extracting")
                    return None

            inverse_indices = np.argsort(sorted_indices)
            patches_reordered = patches[inverse_indices]

            gedi_df = gedi_df.copy()
            gedi_df['embedding_patch'] = [
                p if not np.isnan(p).all() else None
                for p in patches_reordered
            ]
            return gedi_df

        except Exception as e:
            print(f"  Failed to load cache: {e}")
            return None

    def _save_extractions_to_cache(
        self, cache_path: Path, gedi_df: pd.DataFrame
    ) -> None:
        coords = gedi_df[['longitude', 'latitude']].round(6).values
        sorted_indices = np.lexsort((coords[:, 1], coords[:, 0]))

        patches_list = list(gedi_df['embedding_patch'])
        patch_shape = None
        for patch in patches_list:
            if patch is not None:
                patch_shape = patch.shape
                break

        if patch_shape is None:
            print("  Warning: No valid patches to cache")
            return

        patches = []
        for idx in sorted_indices:
            patch = patches_list[idx]
            if patch is not None:
                patches.append(patch)
            else:
                patches.append(np.full(patch_shape, np.nan))

        patches_array = np.array(patches)
        coords_sorted = coords[sorted_indices]
        coords_hash = hashlib.md5(coords_sorted.tobytes()).hexdigest()

        np.savez_compressed(
            cache_path,
            patches=patches_array,
            year=self.year,
            patch_size=self.patch_size,
            coords_hash=coords_hash
        )

        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"  Saved extraction cache ({cache_size_mb:.1f} MB): {cache_path}")


class EmbeddingExtractor(BaseEmbeddingExtractor):
    """GeoTessera embedding extractor (128-channel, 10m resolution)."""

    def __init__(
        self,
        year: int = 2024,
        patch_size: int = 3,
        embeddings_dir: Optional[str] = None
    ):
        from geotessera import GeoTessera
        super().__init__(year=year, patch_size=patch_size, embedding_channels=128)
        self.gt = GeoTessera(embeddings_dir=embeddings_dir)

    @property
    def source_name(self) -> str:
        return "GeoTessera"

    def _get_tile_coords(self, lon: float, lat: float) -> Tuple[float, float]:
        tile_lon = round((lon - 0.05) / 0.1) * 0.1 + 0.05
        tile_lat = round((lat - 0.05) / 0.1) * 0.1 + 0.05
        return tile_lon, tile_lat

    def _load_tile(
        self, tile_lon: float, tile_lat: float
    ) -> Optional[Tuple[np.ndarray, object, object]]:
        tile_key = (tile_lon, tile_lat)
        if tile_key in self.tile_cache:
            return self.tile_cache[tile_key]

        try:
            embedding, crs, transform = self.gt.fetch_embedding(
                lon=tile_lon, lat=tile_lat, year=self.year
            )
            tile_data = (embedding, crs, transform)
            self.tile_cache[tile_key] = tile_data
            return tile_data
        except Exception as e:
            print(f"Warning: Could not fetch GeoTessera tile at ({tile_lon}, {tile_lat}): {e}")
            return None


class AlphaEarthExtractor(BaseEmbeddingExtractor):
    """
    AlphaEarth Foundations satellite embedding extractor (64-channel, 10m resolution).

    Uses Google Earth Engine to access the GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
    image collection produced by AlphaEarth Foundations (Google DeepMind).

    Instead of downloading entire tiles (which exceeds EE's 48MB limit),
    this extractor uses batch point sampling via ee.Image.sampleRegions()
    with neighborhoodToArray() to extract small patches directly at shot
    locations. This is much faster and avoids size limits entirely.

    Requires:
        - Earth Engine Python API: pip install earthengine-api
        - Authentication: earthengine authenticate
        - A Google Cloud project with Earth Engine enabled
    """

    COLLECTION_ID = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
    EMBEDDING_BANDS = [f'A{i:02d}' for i in range(64)]
    EMBEDDING_CHANNELS = 64

    # Max points per EE sampleRegions call (EE limit is ~5000 features)
    BATCH_SIZE = 4000

    def __init__(
        self,
        year: int = 2024,
        patch_size: int = 3,
        project: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(year=year, patch_size=patch_size, embedding_channels=self.EMBEDDING_CHANNELS)
        self.project = project
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._ee_initialized = False
        self._ee_image = None
        self._ee_patch_image = None  # Image with neighborhoodToArray applied

    def _ensure_ee_initialized(self):
        """Lazily initialize Earth Engine on first use."""
        if self._ee_initialized:
            return

        import ee

        try:
            if self.project:
                ee.Initialize(project=self.project)
            else:
                ee.Initialize()
        except Exception:
            try:
                ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
            except Exception as e:
                raise RuntimeError(
                    f"Could not initialize Earth Engine. Run 'earthengine authenticate' first. "
                    f"Error: {e}"
                )

        self._ee_initialized = True
        self._load_ee_image()

    def _load_ee_image(self):
        """Load the Earth Engine image and prepare neighborhood arrays."""
        import ee

        collection = ee.ImageCollection(self.COLLECTION_ID)
        self._ee_image = (
            collection
            .filterDate(f'{self.year}-01-01', f'{self.year}-12-31')
            .select(self.EMBEDDING_BANDS)
            .mosaic()
        )

        # Pre-apply neighborhoodToArray: converts each band to a patch_size x patch_size
        # array at each pixel. This allows sampleRegions to return patches.
        radius = self.patch_size // 2
        kernel = ee.Kernel.square(radius=radius, units='pixels')
        self._ee_patch_image = self._ee_image.neighborhoodToArray(kernel)

    def set_year(self, year: int) -> None:
        if year != self.year:
            self.year = year
            self.tile_cache.clear()
            if self._ee_initialized:
                self._load_ee_image()

    @property
    def source_name(self) -> str:
        return "AlphaEarth"

    # -- Tile-based methods (not used, but required by ABC) --

    def _get_tile_coords(self, lon: float, lat: float) -> Tuple[float, float]:
        tile_lon = round((lon - 0.05) / 0.1) * 0.1 + 0.05
        tile_lat = round((lat - 0.05) / 0.1) * 0.1 + 0.05
        return tile_lon, tile_lat

    def _load_tile(
        self, tile_lon: float, tile_lat: float
    ) -> Optional[Tuple[np.ndarray, object, object]]:
        # Not used — this extractor uses batch point sampling instead
        return None

    # -- Batch point sampling (overrides the base class loop) --

    def extract_patches_batch(
        self,
        gedi_df: pd.DataFrame,
        verbose: bool = True,
        desc: str = None,
        cache_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract patches for all GEDI shots using batch EE point sampling.

        Uses neighborhoodToArray + sampleRegions to extract patches at all
        shot locations in batches, avoiding per-tile downloads entirely.
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

        self._ensure_ee_initialized()

        if desc is None:
            desc = f"Extracting AlphaEarth embeddings (year {self.year})"

        n_total = len(gedi_df)
        patches = [None] * n_total
        successful = 0

        # Process in batches
        indices = list(range(n_total))
        n_batches = (n_total + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        if verbose:
            batch_iter = tqdm(range(n_batches), desc=desc)
        else:
            batch_iter = range(n_batches)

        for batch_idx in batch_iter:
            start = batch_idx * self.BATCH_SIZE
            end = min(start + self.BATCH_SIZE, n_total)
            batch_indices = indices[start:end]

            batch_df = gedi_df.iloc[batch_indices]
            batch_patches = self._sample_batch(batch_df)

            for i, (idx, patch) in enumerate(zip(batch_indices, batch_patches)):
                patches[idx] = patch
                if patch is not None:
                    successful += 1

        gedi_df = gedi_df.copy()
        gedi_df['embedding_patch'] = patches

        if verbose:
            print(f"  {successful}/{n_total} shots with valid embeddings "
                  f"({100*successful/n_total:.1f}%)")

        if cache_dir is not None:
            self._save_extractions_to_cache(cache_path, gedi_df)

        return gedi_df

    def _sample_batch(self, batch_df: pd.DataFrame) -> list:
        """
        Sample patches for a batch of points via EE sampleRegions.

        Returns list of numpy arrays (patch_size, patch_size, 64) or None.
        """
        import ee

        # Create FeatureCollection from points
        features = []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            point = ee.Geometry.Point([float(row['longitude']), float(row['latitude'])])
            features.append(ee.Feature(point, {'idx': i}))

        fc = ee.FeatureCollection(features)

        try:
            # Sample the neighborhood-array image at all points
            # Each feature gets properties like A00 (a patch_size×patch_size list), A01, ...
            sampled = self._ee_patch_image.sampleRegions(
                collection=fc,
                scale=10,
                geometries=False
            )

            results = sampled.getInfo()

            if results is None or 'features' not in results:
                return [None] * len(batch_df)

            # Parse results back into patches, indexed by 'idx'
            patches_by_idx = {}
            ps = self.patch_size

            for feature in results['features']:
                props = feature['properties']
                idx = props['idx']

                try:
                    band_arrays = []
                    for band in self.EMBEDDING_BANDS:
                        arr = props.get(band)
                        if arr is None:
                            raise ValueError(f"Missing band {band}")
                        band_arrays.append(np.array(arr, dtype=np.float32).reshape(ps, ps))

                    # Stack bands -> (patch_size, patch_size, 64)
                    patch = np.stack(band_arrays, axis=-1)
                    patches_by_idx[idx] = patch
                except Exception:
                    patches_by_idx[idx] = None

            # Return in original order
            return [patches_by_idx.get(i) for i in range(len(batch_df))]

        except Exception as e:
            print(f"Warning: EE sampleRegions failed for batch of {len(batch_df)}: {e}")
            return [None] * len(batch_df)


def create_embedding_extractor(
    source: str = 'geotessera',
    year: int = 2024,
    patch_size: int = 3,
    embeddings_dir: Optional[str] = None,
    ee_project: Optional[str] = None,
) -> BaseEmbeddingExtractor:
    """
    Factory function to create the appropriate embedding extractor.

    Args:
        source: Embedding source - 'geotessera' or 'alphaearth'
        year: Year of embeddings
        patch_size: Patch size around each shot
        embeddings_dir: Directory for caching/storing embeddings
        ee_project: Google Cloud project for Earth Engine (alphaearth only)

    Returns:
        An embedding extractor instance
    """
    if source == 'geotessera':
        return EmbeddingExtractor(
            year=year,
            patch_size=patch_size,
            embeddings_dir=embeddings_dir
        )
    elif source == 'alphaearth':
        return AlphaEarthExtractor(
            year=year,
            patch_size=patch_size,
            project=ee_project,
            cache_dir=embeddings_dir,
        )
    else:
        raise ValueError(
            f"Unknown embedding source: {source}. "
            f"Choose from: 'geotessera', 'alphaearth'"
        )


# Embedding source metadata for configuring downstream models
EMBEDDING_SOURCES = {
    'geotessera': {
        'channels': 128,
        'resolution_m': 10,
        'years': range(2017, 2025),
        'description': 'GeoTessera foundation model embeddings (Sentinel-2 optical)',
    },
    'alphaearth': {
        'channels': 64,
        'resolution_m': 10,
        'years': range(2017, 2026),
        'description': 'AlphaEarth Foundations embeddings (multi-sensor fusion: S2, S1, Landsat, LiDAR)',
    },
}
