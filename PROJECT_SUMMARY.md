# Project Summary

## Elevator pitch

This project predicts IPL match winners using only information available before the match starts, then simulates the full IPL 2026 season to estimate title probabilities.

It combines:
- historical IPL match results
- ball-by-ball player performance data
- 2026 fixtures
- official squad information
- likely XI assumptions
- simulation-based tournament forecasting

## Problem

Most sports models either:
- stop at a single-match prediction, or
- use in-game/live features that are not available before the toss or first ball

This project focuses on a harder and more realistic problem:

**Can we produce strong pre-match IPL predictions and turn them into a full-season title forecast?**

## ML framing

### Prediction target

Binary classification:
- `1` if `team_1` wins
- `0` if `team_2` wins

### Training data

Historical IPL matches aggregated from ball-by-ball data in [IPL.csv](C:\Users\aarav\Desktop\IPL.csv).

### Model type

AdaBoost classifier with held-out sigmoid calibration.

### Validation strategy

Time-aware evaluation only:
- latest-season holdout
- rolling walk-forward validation across seasons

No random train/test shuffle is used for final evaluation.

## What makes the feature set interesting

The model is not built on generic sports features alone. It includes:

- Elo-based team strength
- recent form
- recent-season win rates
- venue-specific history
- team batting-first/chasing style
- player batting and bowling strength
- powerplay, middle-over, and death-phase strength
- batting and bowling depth
- lineup-informed 2026 priors

This makes the project much closer to a real sports analytics system than a classroom dataset exercise.

## Current results

From [artifacts/training_metrics.csv](C:\Users\aarav\Desktop\IPL\artifacts\training_metrics.csv):

- holdout accuracy: `70.00%`
- holdout ROC-AUC: `0.8630`
- holdout log-loss: `0.5145`

From [artifacts/training_cv_metrics.csv](C:\Users\aarav\Desktop\IPL\artifacts\training_cv_metrics.csv):

- mean CV accuracy: `68.05%`
- mean CV ROC-AUC: `0.8022`
- mean CV log-loss: `0.5638`

These are strong numbers for a pre-match cricket model.

## Engineering strengths

This repo includes real project engineering, not just modeling:

- reusable data-prep scripts
- modular source package
- GitHub branch workflow
- CI checks
- reproducible artifacts
- explicit model evaluation outputs
- simulator debugging and tie-break logic

## What I learned building it

The most important lessons from this project were not just "which model works."

They were:
- how much data cleaning matters
- how easy it is to leak time information in sports prediction
- how simulation realism can affect tournament odds even when match accuracy stays constant
- how lineup assumptions and current-season priors matter in franchise leagues with roster churn

## Best next upgrades

If this project were extended further, the best next steps would be:

- better uncertainty analysis across multiple large simulation runs
- richer toss and batting-order modeling
- injury/availability updates
- automated report generation with plots
- model comparison dashboard across feature families

## Resume-friendly project description

Built an end-to-end machine learning system to predict IPL match winners and simulate the 2026 tournament champion using historical ball-by-ball cricket data, engineered pre-match team and player strength features, time-aware validation, and probability-driven season simulation. Achieved `70%` holdout accuracy and `68%` mean walk-forward CV accuracy on pre-match predictions.
