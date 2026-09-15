# Wine 90+ Classification

This project predicts whether a wine received a score of 90 or higher from its binary sensory-review features.

## Algorithms

The program implements three classifiers from scratch:

1. **Bernoulli Naïve Bayes** uses Laplace-smoothed probabilities for the presence and absence of each sensory term.
2. **K-Nearest Neighbors (KNN)** uses Euclidean distance and majority voting. Five-fold stratified cross-validation on the training dataset selects `k` from 3, 5, 7, 9, and 11. The testing labels are not used during this selection.
3. **Decision Tree** uses Gini impurity to choose binary sensory-feature splits. Five-fold stratified cross-validation selects the maximum depth from 2, 3, 4, 5, and 6 without using the testing labels.

## Run

### Docker (recommended)

Install Docker Desktop on Windows or Docker Engine with the Compose plugin on Linux. Then run this command from the project folder:

```bash
docker compose up --build
```

The results appear in the terminal and remain available in the `output/` folder after the container exits. Each filename uses `DDMMYYYY_HHMMSS`, such as `prediction_results_15092026_174530.csv`.

To run it again without displaying old container logs:

```bash
docker compose run --rm wine-classifier
```

To remove the stopped project container and network:

```bash
docker compose down
```

### Run without Docker

Keep `main.py`, `Training dataset.xlsx`, and `Testing dataset.xlsx` in the same folder. Then run:

```bash
python -m pip install -r requirements.txt
python main.py
```

The terminal displays both accuracies and the real-versus-predicted table. The program also creates a timestamped CSV such as `prediction_results_15092026_174530.csv` with these columns:

- Wine
- Real grade
- Naive Bayes predicted grade
- KNN predicted grade
- Decision Tree predicted grade

Label `1` means a grade of 90 or higher. Label `0` means a grade below 90.

## Method

- The first Excel column contains wine names and is excluded from model inputs.
- The second column contains the real class label.
- The remaining 305 columns are binary sensory-review features.
- All three final models train on all 200 training wines.
- Accuracy is calculated as `correct predictions / 19 testing wines`.
