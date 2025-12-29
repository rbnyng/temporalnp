import gedidb as gdb
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
from typing import Optional, Union, List
import numpy as np
import tiledb
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class GEDIQuerier:
    def __init__(
        self,
        storage_type: str = 's3',
        s3_bucket: str = "dog.gedidb.gedi-l2-l4-v002",
        url: str = "https://s3.gfz-potsdam.de",
        local_path: Optional[str] = None,
        memory_budget_mb: int = 512,
        max_workers: int = 8,
        cache_dir: Optional[str] = None
    ):
        # Configure TileDB memory limits to prevent huge allocations
        memory_budget_bytes = memory_budget_mb * 1024 * 1024
        config = tiledb.Config()
        config["sm.memory_budget"] = str(memory_budget_bytes)
        config["sm.memory_budget_var"] = str(memory_budget_bytes * 2)
        config["sm.tile_cache_size"] = str(memory_budget_bytes // 2)
        config["sm.compute_concurrency_level"] = "1"
        config["sm.io_concurrency_level"] = "1"
        config["sm.enable_signal_handlers"] = "false"

        logger.info(f"TileDB memory budget set to {memory_budget_mb} MB")

        try:
            tiledb.default_ctx(config=config)
        except Exception as e:
            logger.warning(f"Could not set TileDB default context: {e}")
            logger.warning("Continuing without custom TileDB configuration")

        # Store config for later use
        self.tiledb_config = config
        self.memory_budget_mb = memory_budget_mb
        self.max_workers = max_workers

        # Set up caching
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"GEDI query caching enabled: {self.cache_dir}")

        if storage_type == 's3':
            self.provider = gdb.GEDIProvider(
                storage_type='s3',
                s3_bucket=s3_bucket,
                url=url
            )
        else:
            self.provider = gdb.GEDIProvider(
                storage_type='local',
                local_path=local_path
            )

    def _generate_cache_key(
        self,
        bbox: tuple,
        start_time: str,
        end_time: str,
        variables: List[str],
        quality_filter: bool,
        min_agbd: float,
        max_agbd: Optional[float]
    ) -> str:
        """
        Generate a unique cache key for a query.

        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            start_time: Start date
            end_time: End date
            variables: List of variables
            quality_filter: Quality filter flag
            min_agbd: Minimum AGBD threshold
            max_agbd: Maximum AGBD threshold

        Returns:
            Hash string for cache lookup
        """
        # deterministic representation of query parameters
        cache_params = {
            'bbox': [round(x, 6) for x in bbox],  # Round to ~10cm precision
            'start_time': start_time,
            'end_time': end_time,
            'variables': sorted(variables),  # Sort for consistency
            'quality_filter': quality_filter,
            'min_agbd': round(min_agbd, 2),
            'max_agbd': round(max_agbd, 2) if max_agbd is not None else None
        }

        params_str = json.dumps(cache_params, sort_keys=True)
        hash_obj = hashlib.md5(params_str.encode())
        return hash_obj.hexdigest()

    def query_bbox(
        self,
        bbox: tuple,
        start_time: str = "2023-01-01",
        end_time: str = "2023-12-31",
        variables: Optional[List[str]] = None,
        quality_filter: bool = True,
        min_agbd: float = 0.0,
        max_agbd: Optional[float] = None,
        use_dask: bool = False
    ) -> pd.DataFrame:
        """
        Query GEDI shots within a bounding box.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            start_time: Start date (YYYY-MM-DD)
            end_time: End date (YYYY-MM-DD)
            variables: List of variables to retrieve. If None, uses defaults.
            quality_filter: Apply quality filtering based on L4 quality flags
            min_agbd: Minimum AGBD threshold (Mg/ha)
            max_agbd: Maximum AGBD threshold (Mg/ha), None for no upper limit
            use_dask: Use dask for lazy loading (reduces memory usage)

        Returns:
            DataFrame with columns: latitude, longitude, agbd, quality metrics
        """
        if variables is None:
            variables = [
                "agbd",
            ]

        if self.cache_dir:
            cache_key = self._generate_cache_key(
                bbox, start_time, end_time, variables,
                quality_filter, min_agbd, max_agbd
            )
            cache_file = self.cache_dir / f"gedi_{cache_key}.parquet"

            if cache_file.exists():
                logger.info(f"Loading cached GEDI data from {cache_file.name}")
                try:
                    df = pd.read_parquet(cache_file)
                    logger.info(f"Loaded {len(df)} cached shots")
                    return df
                except Exception as e:
                    logger.warning(f"Failed to load cache file: {e}, querying fresh data")

        logger.info(f"Querying GEDI data for bbox {bbox}")
        logger.info(f"Time range: {start_time} to {end_time}")
        logger.info(f"Variables: {variables}")

        bbox_geom = box(*bbox)
        roi = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs="EPSG:4326")

        try:
            return_type = 'xarray'

            gedi_data = self.provider.get_data(
                variables=variables,
                query_type="bounding_box",
                geometry=roi,
                start_time=start_time,
                end_time=end_time,
                return_type=return_type
            )

            logger.info(f"Query successful, converting to DataFrame...")

            if hasattr(gedi_data, 'to_dataframe'):
                df = gedi_data.to_dataframe().reset_index()
            else:
                df = pd.DataFrame(gedi_data)

            logger.info(f"Retrieved {len(df)} shots before filtering")

            # Handle empty results
            if len(df) == 0:
                logger.info("No data found, returning empty DataFrame")
                return pd.DataFrame()

            # Filter by AGBD range
            if 'agbd' in df.columns:
                df = df[df['agbd'] >= min_agbd]
                if max_agbd is not None:
                    df = df[df['agbd'] <= max_agbd]

                # Remove NaN values
                df = df.dropna(subset=['latitude', 'longitude', 'agbd'])
            else:
                logger.warning("'agbd' column not found in results")
                df = df.dropna(subset=['latitude', 'longitude'])

            logger.info(f"Returning {len(df)} shots after filtering")

            # Cache the result if caching is enabled
            if self.cache_dir and len(df) > 0:
                try:
                    cache_key = self._generate_cache_key(
                        bbox, start_time, end_time, variables,
                        quality_filter, min_agbd, max_agbd
                    )
                    cache_file = self.cache_dir / f"gedi_{cache_key}.parquet"
                    df.to_parquet(cache_file, index=False)
                    logger.info(f"Cached query result to {cache_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to cache result: {e}")

            return df

        except MemoryError as e:
            logger.error(f"MemoryError during query: {e}")
            raise
        except Exception as e:
            if "MemoryError" in str(e) or "Unable to allocate" in str(e):
                logger.error(f"TileDB memory allocation error: {e}")
                raise MemoryError(str(e))
            logger.error(f"Error querying GEDI data: {e}")
            raise

    def _query_bbox_chunked(
        self,
        bbox: tuple,
        chunk_size: float = 0.1,
        **kwargs
    ) -> pd.DataFrame:
        """
        Query large bounding box by splitting into smaller chunks.

        This is a workaround for TileDB memory allocation issues.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            chunk_size: Size of each chunk in degrees
            **kwargs: Additional arguments passed to query_bbox

        Returns:
            Combined DataFrame from all chunks
        """
        min_lon, min_lat, max_lon, max_lat = bbox

        # Calculate number of chunks needed
        lon_chunks = int(np.ceil((max_lon - min_lon) / chunk_size))
        lat_chunks = int(np.ceil((max_lat - min_lat) / chunk_size))

        total_chunks = lon_chunks * lat_chunks
        logger.info(f"Splitting query into {total_chunks} chunks ({lon_chunks}x{lat_chunks})")
        logger.info(f"Using {self.max_workers} parallel workers")

        # Build list of chunk bboxes
        chunk_bboxes = []
        for i in range(lon_chunks):
            for j in range(lat_chunks):
                chunk_min_lon = min_lon + i * chunk_size
                chunk_max_lon = min(chunk_min_lon + chunk_size, max_lon)
                chunk_min_lat = min_lat + j * chunk_size
                chunk_max_lat = min(chunk_min_lat + chunk_size, max_lat)
                chunk_bbox = (chunk_min_lon, chunk_min_lat, chunk_max_lon, chunk_max_lat)
                chunk_bboxes.append(chunk_bbox)

        all_data = []

        def query_single_chunk(chunk_info):
            chunk_idx, chunk_bbox = chunk_info
            try:
                chunk_df = self.query_bbox(chunk_bbox, **kwargs)
                return chunk_idx, chunk_df, None
            except Exception as e:
                return chunk_idx, None, e

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(query_single_chunk, (idx, bbox)): idx
                for idx, bbox in enumerate(chunk_bboxes)
            }

            for future in as_completed(futures):
                chunk_idx, chunk_df, error = future.result()
                chunk_bbox = chunk_bboxes[chunk_idx]

                logger.info(f"Chunk {chunk_idx+1}/{total_chunks} ({chunk_bbox}): ", end="")

                if error:
                    logger.warning(f"Failed - {error}")
                    continue

                if chunk_df is not None and len(chunk_df) > 0:
                    all_data.append(chunk_df)
                    logger.info(f"Retrieved {len(chunk_df)} shots")
                else:
                    logger.info(f"No shots found")

        if len(all_data) == 0:
            logger.warning("No data retrieved from any chunks")
            return pd.DataFrame()

        # Combine all chunks
        combined_df = pd.concat(all_data, ignore_index=True)
        # Remove duplicates (shots may appear in multiple chunks at boundaries)
        if 'shot_number' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['shot_number'])
        else:
            combined_df = combined_df.drop_duplicates(subset=['latitude', 'longitude'])

        logger.info(f"Total shots after merging chunks: {len(combined_df)}")
        return combined_df

    def query_tile(
        self,
        tile_lon: float,
        tile_lat: float,
        tile_size: float = 0.1,
        use_chunked: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Query GEDI shots within a single tile.

        Args:
            tile_lon: Tile center longitude
            tile_lat: Tile center latitude
            tile_size: Tile size in degrees (default 0.1° for GeoTessera alignment)
            use_chunked: If True, use chunked query (workaround for memory issues)
            **kwargs: Additional arguments passed to query_bbox

        Returns:
            DataFrame of GEDI shots within the tile
        """
        half_size = tile_size / 2
        bbox = (
            tile_lon - half_size,
            tile_lat - half_size,
            tile_lon + half_size,
            tile_lat + half_size
        )

        if use_chunked:
            return self._query_bbox_chunked(bbox, chunk_size=0.05, **kwargs)
        else:
            try:
                return self.query_bbox(bbox, **kwargs)
            except MemoryError as e:
                logger.warning(f"Memory error in direct query, falling back to chunked approach")
                logger.warning(f"Original error: {e}")
                return self._query_bbox_chunked(bbox, chunk_size=0.05, **kwargs)

    def query_region_tiles(
        self,
        region_bbox: tuple,
        tile_size: float = 0.1,
        **kwargs
    ) -> pd.DataFrame:
        """
        Query GEDI shots across multiple tiles in a region.

        Args:
            region_bbox: (min_lon, min_lat, max_lon, max_lat) for entire region
            tile_size: Tile size in degrees
            **kwargs: Additional arguments passed to query_bbox

        Returns:
            DataFrame with additional 'tile_id' column for spatial CV
        """
        min_lon, min_lat, max_lon, max_lat = region_bbox

        # tile centers
        lon_centers = np.arange(
            min_lon + tile_size/2,
            max_lon,
            tile_size
        )
        lat_centers = np.arange(
            min_lat + tile_size/2,
            max_lat,
            tile_size
        )

        # list of tile centers
        tile_centers = []
        for lon_center in lon_centers:
            for lat_center in lat_centers:
                tile_centers.append((lon_center, lat_center))

        total_tiles = len(tile_centers)
        logger.info(f"Querying {total_tiles} tiles in parallel with {self.max_workers} workers")

        all_shots = []

        def query_single_tile(tile_info):
            tile_idx, (lon_center, lat_center) = tile_info
            try:
                tile_df = self.query_tile(lon_center, lat_center, tile_size, **kwargs)
                if len(tile_df) > 0:
                    # Add tile identifier
                    tile_df['tile_id'] = f"tile_{lon_center:.2f}_{lat_center:.2f}"
                    tile_df['tile_lon'] = lon_center
                    tile_df['tile_lat'] = lat_center
                    return tile_idx, tile_df, None
                return tile_idx, None, None
            except Exception as e:
                return tile_idx, None, e

        # queries in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(query_single_tile, (idx, center)): idx
                for idx, center in enumerate(tile_centers)
            }

            for future in as_completed(futures):
                tile_idx, tile_df, error = future.result()
                lon_center, lat_center = tile_centers[tile_idx]

                if error:
                    logger.warning(f"Tile {tile_idx+1}/{total_tiles} ({lon_center:.2f}, {lat_center:.2f}): Failed - {error}")
                    continue

                if tile_df is not None:
                    all_shots.append(tile_df)
                    logger.info(f"Tile {tile_idx+1}/{total_tiles} ({lon_center:.2f}, {lat_center:.2f}): Retrieved {len(tile_df)} shots")
                else:
                    logger.debug(f"Tile {tile_idx+1}/{total_tiles} ({lon_center:.2f}, {lat_center:.2f}): No shots found")

        if len(all_shots) == 0:
            return pd.DataFrame()

        return pd.concat(all_shots, ignore_index=True)


def get_gedi_statistics(df: pd.DataFrame) -> dict:
    stats = {
        'n_shots': len(df),
        'agbd_mean': df['agbd'].mean(),
        'agbd_std': df['agbd'].std(),
        'agbd_min': df['agbd'].min(),
        'agbd_max': df['agbd'].max(),
        'spatial_extent': {
            'lon_range': (df['longitude'].min(), df['longitude'].max()),
            'lat_range': (df['latitude'].min(), df['latitude'].max())
        }
    }

    if 'tile_id' in df.columns:
        stats['n_tiles'] = df['tile_id'].nunique()
        stats['shots_per_tile'] = df.groupby('tile_id').size().describe().to_dict()

    return stats
