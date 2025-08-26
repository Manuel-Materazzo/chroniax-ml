# chroniax-ml 🫀📊

**chroniax-ml** is a Python library and CLI tool designed for calibrating and analyzing heart rate data exported from Withings ScanWatch devices to T10 datasets exported by the chroniax app. Built specifically to support the chroniax ecosystem, this library applies monotonic regression models (PCHIP, Isotonic) with contextual awareness (rest, active, global) to align wearable sensor heart rate measurements to reference values. Perfect for scientific and personal heart rate analytics.

***

## ✨ Features

- 🕺 **Contextual Calibration:** Calibrate ScanWatch heart rate to T10 using context-specific models (rest, active, global).
- 📈 **Monotonic Modeling:** Supports monotone PCHIP and Isotonic regression for robust calibration.
- ⏱️ **Minute-Level Pairing:** Joins ScanWatch and T10 data at minute resolution, with time-weighted aggregation and context annotation.
- 📊 **Metrics Calculation:** Computes error metrics (MAE, bias) grouped by heart rate zones.
- 💤🏃 **Sleep and Sport Contexts:** Annotates data with sleep and sport periods for improved context-dependent modeling.
- 💻 **CLI Tool:** Easily run calibrations via command line, with configurable output and model type.

***

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/Manuel-Materazzo/chroniax-ml.git
cd chroniax-ml
```

***

## 🛠️ Usage

### 🐍 As a Library

Install dependencies:

```bash
pip install -r requirements.txt
```

Use chroniax-ml as a Python library for custom workflows:

```python
from service.ml.contextual_model_trainer import ContextualModelTrainer

trainer = ContextualModelTrainer(model_kind="pchip", local_tz="Europe/Rome", min_scan_coverage_s=30)
pairs, summary = trainer.train_and_apply("scanwatch.csv", "chroniax.sqlite", user_id=1, bin_size="1min")
```


***

### 🖥️ Command Line

Install dependencies:

```bash
pip install -r requirements.txt
```

Run calibration and model fitting:

```bash
python main.py \
    --scan_csv path/to/scanwatch.csv \
    --sqlite path/to/chroniax.sqlite \
    --user_id 1 \
    --freq 1min \
    --model pchip \
    --out_pairs paired_dataset.csv \
    --out_models models_and_metrics.json
```

**Arguments:**

- 📄 `--scan_csv`: Path to ScanWatch CSV (must have columns: start, duration, value as arrays)
- 🗄️ `--sqlite`: Path to chroniax SQLite database
- 👤 `--user_id`: Optional userId to filter T10 data
- ⏳ `--freq`: Bin size (e.g., `1min`, `30S`)
- 🧮 `--model`: Calibration model (`pchip` or `isotonic`)
- 💾 `--out_pairs`: Output CSV for paired and predicted data
- 📝 `--out_models`: Output JSON for fitted models and metrics

***

### 🐳 Docker

The project can be built and run using Docker to isolate the environment and provide a reproducible setup.

#### 🏗️ Build the Docker image

```bash
docker build -t chroniax-ml:latest .
```


#### ▶️ Run the CLI directly

```bash
docker run --rm -v "$(pwd):/app" chroniax-ml:latest \
    python main.py \
    --scan_csv path/to/scanwatch.csv \
    --sqlite path/to/chroniax.sqlite \
    --user_id 1 \
    --freq 1min \
    --model pchip \
    --out_pairs paired_dataset.csv \
    --out_models models_and_metrics.json
```


#### 🧩 Docker Compose

For convenience, a `docker-compose.yml` file is included.

```bash
docker-compose up --rm
```


***

## 📂 Data Requirements

**ScanWatch CSV**

- Columns: `start` (ISO datetime), `duration` (list of ints), `value` (list of floats)

**Chroniax SQLite**

- Heart rate table (`HeartRateItemEntity`)
- Sleep table (`SleepItemEntity`)
- Sport table (`SportRecordEntity`)

***

## 🧠 Model Types

- **PCHIP** (Piecewise Cubic Hermite Interpolating Polynomial): Monotonic cubic interpolation using binned medians for robust knots.
- **Isotonic Regression:** Non-parametric monotonic mapping for calibration.

***

## 📤 Output

- **Paired CSV:** Minute-level merged data with predicted/calibrated heart rates, context annotations.
- **Models \& Metrics JSON:** Model parameters per context and evaluation metrics.

***

## 📝 Example Workflow

1. 📑 Prepare ScanWatch CSV and chroniax SQLite database.
2. 🛠️ Run the CLI tool.
3. ✅ Analyze metrics and output files for calibration quality.

***

## 🧑‍💻 Development

- 🐍 Python 3.10+ recommended
- 📦 All dependencies in `requirements.txt`
- 🏗️ Modular codebase for easy extension

***

## 📜 License

MIT License