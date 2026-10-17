# Vision-to-Action — Baseline training

These files accompany your existing **model.py**. Do not replace it.

## 1. Download Udacity sample driving data

From the root of your VS Code `vision-to-action` folder (with `vision-ai` activated):

```bash
curl -fL --retry 2 -o udacity-data.zip \
  'https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip'
unzip -q udacity-data.zip
ls data/driving_log.csv data/IMG | head
```

The source is the publicly referenced Udacity simulator training sample. The download URL is an older archive and its availability is **not guaranteed**. If downloading fails, stop and report the exact error rather than substituting unlabeled imagery.

If ZIP extracts into another folder, move it so that your project has `data/driving_log.csv` and `data/IMG/*.jpg`.

## 2. Protect data and model weights from Git

Append the following to the project's `.gitignore` (only once):

```
/data/
/udacity-data.zip
/checkpoints/
/experiments/
```

## 3. Train and evaluate

```bash
conda activate vision-ai
python train.py --epochs 5 --batch-size 64
python evaluate.py
```

Outputs:
- `checkpoints/best.pt`: weights from the epoch with lowest validation RMSE.
- `experiments/history.csv`: per-epoch learning curves.

This is an **offline behavioral-cloning baseline**, using only the center camera. Validation holds out the final 20% of the recording, rather than randomly shuffling nearly identical neighboring frames into both splits. A strong next step is evaluating on an entirely separate driving session/track and testing in closed loop.

**Scientific caveat:** A low steering prediction RMSE is not evidence that the vehicle can drive autonomously. Models can learn to predict near-zero steering when the data is dominated by straight roads; compare against the zero-steering and training-mean baselines printed by the scripts.
