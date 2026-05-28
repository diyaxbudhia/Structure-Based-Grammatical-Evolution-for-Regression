# Structure-Based Grammatical Evolution
### Name: Diya Budhia

## Files
- NB: csv data file should be stored in a folder called "data"
- `ge_load_forecasting.py`: command-line interface for data loading, experiment execution, and summary output.
- `ge_core.py`: grammatical evolution engine, structure-based selection, and fitness evaluation.
- `ge_tree.py`: expression tree representation, evaluation, and structure signatures.
- `data_loader.py`: dataset parsing, time-series conversion, and train/validation/test splitting.
- `report.tex`: LaTeX report draft.
- `requirements.txt`: Python dependencies.

## How to run with Docker 

### Build:

```bash
docker build -t cos710-a3:latest .
```

### Run:
```bash
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/outputs:/app/outputs" cos710-a3:latest
```

## Requirements
Install Python dependencies in a virtualenv or your system Python:

```bash
python3 -m pip install -r requirements.txt
```

Note: `requirements.txt` contains `numpy` and `pandas`.

Figure generation has been removed from this repository; the plotting helper and matplotlib are not included by default.

## How to run Locally
All commands assume your current working directory is the repository root. Replace `python3` with your Python executable as needed.


```bash
cd Assignment3
python3 ge_load_forecasting.py
```

### Assignment run (10+ runs)

```bash
cd Assignment3
python3 ge_load_forecasting.py \
	--data "data/Residential_Energy_Dataset_UK- 2014-2020.csv" \
	--target-col Electricity_load \
	--start-row 0 \
	--max-rows 20000 \
	--mode previous_days \
	--lag-count 7 \
	--points-per-day 0 \
	--runs 10 \
	--P 120 \
	--generations 60 \
	--out-dir outputs/assignment
```

### Alternative mode: previous values (m previous values)

```bash
cd Assignment3
python3 ge_load_forecasting.py \
	--data "data/Residential_Energy_Dataset_UK- 2014-2020.csv" \
	--target-col Electricity_load \
	--start-row 0 \
	--max-rows 20000 \
	--mode previous_values \
	--lag-count 24 \
	--runs 10 \
	--P 120 \
	--generations 60 \
	--out-dir outputs/prev_values
```

Notes about flags
- `--P` / `--population-size`: population size (default 120)
- `--generations`: number of generations (default 60)
- `--R`: crossover/mutation rates as `"crossover,mutation"` (default `"0.85,0.15"`)
- `--S` / `--tournament-size`: tournament size (default 4)
- `--Dm`: max tree depth (default 9)
- `--DI`: initial max init depth (default 4)
- `--out-dir`: output directory for CSV and text results (default `outputs`)

