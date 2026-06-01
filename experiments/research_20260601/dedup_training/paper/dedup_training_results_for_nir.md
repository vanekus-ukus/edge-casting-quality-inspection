# Deduplicated training results for NIR

## 1. Зачем понадобился deduplicated split
В исходном split найдены cross-split exact/near duplicates, поэтому исходные accuracy/F1 могли быть завышены.

## 2. Clean split
Использован group split по SHA1 и average-hash near duplicates. Датасет не изменялся; split материализован отдельно в `experiments/research_20260601/dedup_training/data/processed_clean`.

## 3. Clean test metrics
profile,accuracy,precision,recall,f1,defective_recall,false_ok_count,false_defective_count,model_size_kb,params,latency_ms,p95_ms,p99_ms,fps
baseline_pc_dedup,0.9689119170984456,0.9698924731182795,0.9783080260303688,0.9740820734341252,0.9783080260303688,10,14,8919.5107421875,2226434,4.450851983335724,10.549919099810268,10.945016339799166,224.67608533019396
edge_sbc_dedup,0.9702072538860104,0.995475113122172,0.9544468546637744,0.9745293466223699,0.9544468546637744,21,2,6057.6025390625,1519906,4.267013563336756,9.25991669986388,11.01667683000187,234.3559459459537
micro_edge_dedup,0.9766839378238342,0.9825708061002179,0.9783080260303688,0.9804347826086957,0.9783080260303688,10,8,101.3193359375,23938,0.4787479399988115,0.9147048501517928,1.0274880000633857,2088.781833719185


## 4. Original vs dedup
profile,original_accuracy,dedup_accuracy,delta_accuracy,original_f1,dedup_f1,delta_f1,original_defective_recall,dedup_defective_recall,delta_defective_recall,original_size_kb,dedup_size_kb,notes
baseline_pc_dedup,0.993,0.9689119170984456,-0.024088082901554397,0.9945,0.9740820734341252,-0.020417926565874822,0.9956,0.9783080260303688,-0.017291973969631225,8919.552,8919.5107421875,deduplicated group split; original results unchanged
edge_sbc_dedup,0.9902,0.9702072538860104,-0.0199927461139896,0.9923,0.9745293466223699,-0.017770653377630063,0.9934,0.9544468546637744,-0.038953145336225514,6057.5744,6057.6025390625,deduplicated group split; original results unchanged
micro_edge_dedup,0.993,0.9766839378238342,-0.01631606217616577,0.9945,0.9804347826086957,-0.014065217391304397,0.989,0.9783080260303688,-0.010691973969631174,101.3,101.3193359375,deduplicated group split; original results unchanged


## 5. Exact TFLite/INT8
micro_edge_dedup INT8: accuracy=0.9754, f1=0.9795, defective_recall=0.9848, size_kb=31.0, latency_ms=0.441.

## 6. Confidence triage
Best zero-unsafe INT8 threshold:
not found

## 7. Ограничения
Latency является proxy на текущей GPU/CPU машине. Raspberry Pi/OpenMV аппаратно не тестировались. Clean split построен эвристически по perceptual hash и требует описания в НИР.

## 8. Вывод
Если качество на clean split сохраняется, методика подтверждается на более строгой валидации; если проседает, исходные результаты следует трактовать как завышенные из-за leakage.
