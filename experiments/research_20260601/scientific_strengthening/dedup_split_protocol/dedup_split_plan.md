# Deduplicated Split Protocol

## Motivation

The original `Casting Product Image Data for Quality Inspection` split contains cross-split exact duplicates and many near-duplicates. This can inflate accuracy, F1, and defective recall because visually identical or near-identical samples may appear both in training and testing.

Current final results are kept unchanged and should be reported as results on the original split. A clean follow-up experiment should use a deduplicated group split.

## Grouping Rule

Each image is assigned to a duplicate group using:

1. Exact content hash: SHA1 over image bytes.
2. Perceptual hash: average hash over resized grayscale image.
3. Near-duplicate linkage: images with perceptual hash Hamming distance `<= 4` are joined into the same group.

The grouping forms connected components: if A is near B and B is near C, all three images stay in one group.

## Split Rule

The split must be performed by duplicate groups, not by individual files.

Required property:

```text
for every dedup_group:
  all files in dedup_group must belong to exactly one of train/val/test
```

This prevents exact or near duplicates from leaking across train, validation, and test.

## Class Balance

Groups are assigned with a fixed random seed and target ratios close to the existing dataset:

- train: 0.74
- val: 0.16
- test: 0.10

The assignment tries to preserve the OK/DEFECTIVE class balance at split level. If a dedup group contains mixed classes, the group is kept intact and treated as a limitation.

## Recomputed Metrics for Future Work

On the clean split, the following must be recomputed:

- accuracy;
- precision;
- recall;
- F1;
- defective recall;
- confusion matrix;
- confidence triage metrics;
- robustness metrics;
- latency/resource metrics can be reused only as model-level proxy, but quality metrics must be recomputed.

## Reporting

The original final tables must not be silently replaced. The clean split should be reported as a separate validation scenario:

- original split: comparable with existing Kaggle/project results;
- deduplicated group split: stricter estimate of generalization.

The expected outcome may be lower quality metrics. This is scientifically acceptable and improves validity.
