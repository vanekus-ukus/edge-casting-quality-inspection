# Release summary

- PDF: `reports/final_nir/final_nir.pdf`
- LaTeX: `reports/final_nir/main.tex`
- Exact TFLite INT8: `experiments/research_20260601/dedup_training/exported/micro_edge_dedup_exact_int8.tflite`
- Tables: `experiments/research_20260601/dedup_training/tables/`
- Metrics: `experiments/research_20260601/dedup_training/metrics/`
- Dedup protocol: `experiments/research_20260601/scientific_strengthening/dedup_split_protocol/`

Snapshot excludes the Kaggle dataset, virtual environment, raw/processed data, large model weights, and logs.

## Figure readability update

Final NIR figures were regenerated in a clean black-and-white academic style: grayscale fills, thin grid, no dense hatching. Learning curves are not included in the final PDF.

## Figure layout update

Fixed figure label layout: size axis note moved below the plot, Pareto labels separated, triage title shortened. Repository URL added to the NIR appendix.

## Final figure label cleanup

Removed problematic vertical axis labels from quality and triage figures. Adjusted Pareto layout so labels stay inside the plot area.
