
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
