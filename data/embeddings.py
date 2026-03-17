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

    Embeddings are 64-dimensional unit-norm vectors derived from multi-sensor fusion
    (Sentinel-2, Sentinel-1, Landsat, LiDAR, elevation, climate data).

    Requires:
        - Earth Engine Python API: pip install earthengine-api
        - Authentication: earthengine authenticate
        - A Google Cloud project with Earth Engine enabled
    """

    # AlphaEarth tiles are ~163,840m x 163,840m in UTM, organized by UTM zone.
    # We access them as a continuous image collection and sample at point locations.
    COLLECTION_ID = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
    EMBEDDING_BANDS = [f'A{i:02d}' for i in range(64)]
    EMBEDDING_CHANNELS = 64

    def __init__(
        self,
        year: int = 2024,
        patch_size: int = 3,
        project: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize AlphaEarth embedding extractor.

        Args:
            year: Year of embeddings to use (2017-2025)
            patch_size: Size of patch to extract around each shot (e.g., 3 = 3x3 = 30m x 30m)
            project: Google Cloud project ID for Earth Engine. If None, uses default.
            cache_dir: Directory for caching downloaded GeoTIFF tiles to disk.
                      Tiles are cached as numpy arrays for fast reloading.
        """
        super().__init__(year=year, patch_size=patch_size, embedding_channels=self.EMBEDDING_CHANNELS)
        self.project = project
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._ee_initialized = False
        self._ee_image = None

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
            # If default init fails, try high-volume endpoint
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
        """Load the Earth Engine image for the current year."""
        import ee

        collection = ee.ImageCollection(self.COLLECTION_ID)
        self._ee_image = (
            collection
            .filterDate(f'{self.year}-01-01', f'{self.year}-12-31')
            .select(self.EMBEDDING_BANDS)
            .mosaic()
        )

    def set_year(self, year: int) -> None:
        if year != self.year:
            self.year = year
            self.tile_cache.clear()
            if self._ee_initialized:
                self._load_ee_image()

    @property
    def source_name(self) -> str:
        return "AlphaEarth"

    def _get_tile_coords(self, lon: float, lat: float) -> Tuple[float, float]:
        # Use same 0.1° grid as GeoTessera for consistency
        tile_lon = round((lon - 0.05) / 0.1) * 0.1 + 0.05
        tile_lat = round((lat - 0.05) / 0.1) * 0.1 + 0.05
        return tile_lon, tile_lat

    def _get_tile_cache_path(self, tile_lon: float, tile_lat: float) -> Optional[Path]:
        """Get disk cache path for a tile."""
        if self.cache_dir is None:
            return None
        tile_dir = self.cache_dir / 'alphaearth' / str(self.year)
        tile_dir.mkdir(parents=True, exist_ok=True)
        return tile_dir / f'tile_{tile_lon:.2f}_{tile_lat:.2f}.npz'

    def _load_tile(
        self, tile_lon: float, tile_lat: float
    ) -> Optional[Tuple[np.ndarray, object, object]]:
        tile_key = (tile_lon, tile_lat)

        # Memory cache
        if tile_key in self.tile_cache:
            return self.tile_cache[tile_key]

        # Disk cache
        cache_path = self._get_tile_cache_path(tile_lon, tile_lat)
        if cache_path is not None and cache_path.exists():
            try:
                data = np.load(cache_path, allow_pickle=True)
                from rasterio.transform import Affine
                transform = Affine(*data['transform'])
                crs_str = str(data['crs'])
                tile_data = (data['embedding'], crs_str, transform)
                self.tile_cache[tile_key] = tile_data
                return tile_data
            except Exception as e:
                print(f"Warning: Could not load cached tile ({tile_lon}, {tile_lat}): {e}")

        # Fetch from Earth Engine
        try:
            tile_data = self._fetch_tile_from_ee(tile_lon, tile_lat)
            if tile_data is not None:
                self.tile_cache[tile_key] = tile_data
                # Save to disk cache
                if cache_path is not None:
                    self._save_tile_to_cache(cache_path, tile_data)
            return tile_data
        except Exception as e:
            print(f"Warning: Could not fetch AlphaEarth tile at ({tile_lon}, {tile_lat}): {e}")
            return None

    def _fetch_tile_from_ee(
        self, tile_lon: float, tile_lat: float
    ) -> Optional[Tuple[np.ndarray, str, object]]:
        """Fetch a tile from Earth Engine as a numpy array."""
        import ee
        from rasterio.transform import Affine

        self._ensure_ee_initialized()

        # Define tile region (0.1° x 0.1° around center)
        half = 0.05
        region = ee.Geometry.Rectangle([
            tile_lon - half, tile_lat - half,
            tile_lon + half, tile_lat + half
        ])

        # Get the appropriate UTM CRS for this location
        utm_zone = int((tile_lon + 180) / 6) + 1
        hemisphere = 'N' if tile_lat >= 0 else 'S'
        epsg_code = 32600 + utm_zone if hemisphere == 'N' else 32700 + utm_zone
        crs_string = f'EPSG:{epsg_code}'

        # Download pixels via computePixels or getPixels
        try:
            # Use ee.data.computePixels for efficient download
            request = {
                'expression': self._ee_image,
                'fileFormat': 'NUMPY_NDARRAY',
                'grid': {
                    'dimensions': {'width': 1100, 'height': 1100},
                    'affineTransform': {
                        'scaleX': 10,
                        'shearX': 0,
                        'translateX': 0,
                        'scaleY': -10,
                        'shearY': 0,
                        'translateY': 0,
                    },
                    'crsCode': crs_string,
                },
                'bandIds': self.EMBEDDING_BANDS,
            }

            # Get UTM bounds to set the affine transform correctly
            proj = Transformer.from_crs("EPSG:4326", crs_string, always_xy=True)
            x_min, y_min = proj.transform(tile_lon - half, tile_lat - half)
            x_max, y_max = proj.transform(tile_lon + half, tile_lat + half)

            width = int((x_max - x_min) / 10)
            height = int((y_max - y_min) / 10)

            request['grid']['dimensions'] = {'width': width, 'height': height}
            request['grid']['affineTransform']['translateX'] = x_min
            request['grid']['affineTransform']['translateY'] = y_max  # Top-left Y

            pixels = ee.data.computePixels(request)

            # Convert structured array to (H, W, C)
            bands = [pixels[band] for band in self.EMBEDDING_BANDS]
            embedding = np.stack(bands, axis=-1).astype(np.float32)

            # Build affine transform
            transform = Affine(10, 0, x_min, 0, -10, y_max)

            return (embedding, crs_string, transform)

        except Exception as e:
            # Fallback: use getRegion for smaller areas
            print(f"  computePixels failed for ({tile_lon}, {tile_lat}), trying getRegion: {e}")
            return self._fetch_tile_via_get_region(tile_lon, tile_lat, region, crs_string)

    def _fetch_tile_via_get_region(
        self, tile_lon: float, tile_lat: float,
        region, crs_string: str
    ) -> Optional[Tuple[np.ndarray, str, object]]:
        """Fallback: fetch tile using getRegion (slower, for smaller areas)."""
        import ee
        from rasterio.transform import Affine

        try:
            # Sample at 10m scale
            result = self._ee_image.sampleRectangle(
                region=region,
                defaultValue=0
            )
            info = result.getInfo()

            if info is None or 'properties' not in info:
                return None

            props = info['properties']
            bands = []
            for band_name in self.EMBEDDING_BANDS:
                if band_name in props:
                    bands.append(np.array(props[band_name], dtype=np.float32))
                else:
                    return None

            embedding = np.stack(bands, axis=-1)

            # Approximate affine transform
            half = 0.05
            proj = Transformer.from_crs("EPSG:4326", crs_string, always_xy=True)
            x_min, y_min = proj.transform(tile_lon - half, tile_lat - half)
            x_max, y_max = proj.transform(tile_lon + half, tile_lat + half)

            height, width = embedding.shape[:2]
            pixel_size_x = (x_max - x_min) / width
            pixel_size_y = (y_max - y_min) / height

            transform = Affine(pixel_size_x, 0, x_min, 0, -pixel_size_y, y_max)

            return (embedding, crs_string, transform)

        except Exception as e:
            print(f"Warning: getRegion also failed for ({tile_lon}, {tile_lat}): {e}")
            return None

    def _save_tile_to_cache(
        self, cache_path: Path, tile_data: Tuple[np.ndarray, str, object]
    ) -> None:
        """Save a downloaded tile to disk cache."""
        embedding, crs, transform = tile_data
        try:
            np.savez_compressed(
                cache_path,
                embedding=embedding,
                crs=str(crs),
                transform=np.array([
                    transform.a, transform.b, transform.c,
                    transform.d, transform.e, transform.f
                ])
            )
        except Exception as e:
            print(f"Warning: Could not cache tile: {e}")


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
