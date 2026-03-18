
# mcfarland fire (trinity-shasta)
python run_experiment.py \
  --region_bbox -123 40 -122.5 40.5 \
  --train_years 2019 2020 2022 2023 \
  --test_year 2021 \
  --n_seeds 1 \
  --output_dir ./results/mcfarland

# dixie fire
python run_experiment.py \
  --region_bbox -121.5 39.7 -120.5 40.7 \
  --train_years 2019 2020 2022 2023 \
  --test_year 2021 \
  --n_seeds 1 \
  --output_dir ./results/dixie

# --- static (single-year) embedding benchmarks ---

# guaviare - single year
python run_embedding_benchmark.py \
  --region_bbox -73 2 -72 3 \
  --year 2019 \
  --sources geotessera \
  --n_seeds 5 \
  --output_dir ./results/guaviare_2019_bench

# guaviare - multi year comparison
python run_embedding_benchmark.py \
  --region_bbox -73 2 -72 3 \
  --year 2019 2020 2022 \
  --sources geotessera alphaearth \
  --include_rf \
  --n_seeds 5 \
  --output_dir ./results/guaviare_multi_bench \
  --ee_project your-gcp-project

# static single-year training (standalone)
python train_static.py \
  --region_bbox -73 2 -72 3 \
  --start_time 2022-01-01 \
  --end_time 2022-12-31 \
  --embedding_year 2022 \
  --output_dir ./results/guaviare_2022
