# Vision-to-Action

### Learning Autonomous Driving from Visual Demonstrations

**An independent machine learning research project exploring vision-based autonomous driving through imitation learning.**

Vision-to-Action investigates how neural networks can learn steering behavior directly from camera images, and how model architecture, training strategies, and data distribution affect performance.

## Research Objective

**Can a neural network learn reliable steering behavior from visual observations alone?**

The project explores three key questions:

1. How accurately can CNN-based models predict steering angles from driving images?
2. How do training strategies and architectural choices affect generalization?
3. Why do imitation-learning models fail in unfamiliar driving situations, and how can these failures be reduced?

## System Architecture

```text
Camera Images
      |
      v
Image Preprocessing
      |
      v
Convolutional Neural Network
      |
      v
Steering Angle Prediction
      |
      v
Model Evaluation
```

The initial approach uses supervised behavioral cloning, where a neural network learns to predict steering actions from demonstrations.

## Technology Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Computer Vision:** CNNs, image preprocessing
- **Learning Methods:** Supervised Learning, Behavioral Cloning
- **Hardware Acceleration:** Apple Silicon MPS
- **Evaluation:** Regression metrics, baseline comparisons and error analysis

## Project Structure

```text
vision-to-action/
├── model.py          # CNN architecture
├── dataset.py        # Dataset loading and preprocessing
├── train.py          # Model training
├── evaluate.py       # Evaluation and metrics
├── README.md         # Project overview
└── NEXT_STEPS.md     # Development instructions
```

## Current Progress

**Status: Early development**

- [x] PyTorch environment configured
- [x] CNN architecture implemented and forward pass verified
- [ ] Train baseline on real driving data
- [ ] Evaluate on held-out driving data
- [ ] Compare alternative training strategies
- [ ] Investigate model robustness and failure modes
- [ ] Explore closed-loop simulation

## Getting Started

Create a Python environment and install the required libraries:

```bash
conda create -n vision-ai python=3.11 -y
conda activate vision-ai
python -m pip install torch torchvision numpy matplotlib pillow tqdm
```

Verify the neural network:

```bash
python model.py
```

Dataset setup and training instructions will be finalized as the baseline experiments are validated.

## Research Results

Training results, model comparisons, visualizations, and experimental findings will be documented here as experiments are completed.

No autonomous driving performance claims are made at this stage.

## Author

**Eric Liu**

UBC — MEng Mechatronics  
BASc Manufacturing Engineering

Research interests: Deep Learning, Computer Vision, Foundation Models, and Robotics AI.

---

*Independent research and portfolio project. Not affiliated with Tesla or any autonomous-driving company.*