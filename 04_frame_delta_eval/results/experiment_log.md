# Frame-Delta Evaluation Experiment Log

## 2026-05-25 - SKT elbow quality-repair frame-delta check

- Goal: test whether the SKT elbow-chain quality repair from `01_stereo_triangulation` also improves the frame-delta agreement with the Xsens-derived reference.
- Candidate repaired files:
  - `01_stereo_triangulation/results/skt_model_fusion/elbow_quality_repair_gap12/skt_elbow_quality_repaired.npz`
  - `01_stereo_triangulation/results/skt_model_fusion/elbow_quality_repair_gap20/skt_elbow_quality_repaired.npz`
- Important methodology note:
  - `--enable-quality-filter` was **not** used in this check, because the repair itself fills some frames that would otherwise be masked again by the evaluator's quality filter.
  - This means the comparison here is "original SKT vs repaired SKT", not "old quality-mask pipeline vs repaired SKT".
- Output:
  - `04_frame_delta_eval/results/skt_elbow_repair_compare/summary.md`

### Main command pattern

- Original:
  - `/opt/anaconda3/envs/pose/bin/python 04_frame_delta_eval/src/01_compute_elbow_deltas.py --skt-npz 01_stereo_triangulation/results/historical_best_20260324/recovered_baseline/optimized_pose.npz --skip-afh --fastsam-trc '../10 Aitor/fastsam3d_2.trc' --merge-trc '../10 Aitor/merged_output_2.trc' --k-frame-list 1,6 --smooth-method moving_average --smooth-window-ms 200 --out-dir 04_frame_delta_eval/results/skt_elbow_repair_compare/original_no_quality_filter --skip-plots`
- Gap-12 repair:
  - `/opt/anaconda3/envs/pose/bin/python 04_frame_delta_eval/src/01_compute_elbow_deltas.py --skt-npz 01_stereo_triangulation/results/skt_model_fusion/elbow_quality_repair_gap12/skt_elbow_quality_repaired.npz --skip-afh --fastsam-trc '../10 Aitor/fastsam3d_2.trc' --merge-trc '../10 Aitor/merged_output_2.trc' --k-frame-list 1,6 --smooth-method moving_average --smooth-window-ms 200 --out-dir 04_frame_delta_eval/results/skt_elbow_repair_compare/gap12_no_quality_filter --skip-plots`
- Gap-20 repair:
  - same as above, with `elbow_quality_repair_gap20`.

### SKT vs XsensFair summary

| Run | Left K1 Pearson / Path | Left K6 Pearson / Path | Right K1 Pearson / Path | Right K6 Pearson / Path |
|---|---:|---:|---:|---:|
| original | `0.177 / 2.630` | `0.310 / 1.912` | `0.172 / 2.618` | `0.282 / 1.899` |
| gap12 | `0.182 / 2.501` | `0.314 / 1.843` | `0.188 / 2.408` | `0.311 / 1.793` |
| gap20 | `0.169 / 2.351` | `0.285 / 1.722` | `0.207 / 2.325` | `0.350 / 1.753` |

### Interpretation

- `gap12` gives a consistent modest improvement across both elbows and K values: Pearson rises, RMSE falls, and path ratio falls.
- `gap20` reduces path ratio further, but worsens LeftElbow Pearson and is less defensible because 20 frames is about `1.6 s` at 12.5 fps.
- Current recommendation:
  - keep `gap12` as the next candidate for visual inspection and possible report discussion;
  - treat `gap20` as an upper-bound diagnostic, not a default setting.

## 2026-05-18 - Aitor TRC FastSAM3D/Merge integration and Phase 4 rerun

- Snapshot before experiment: `0cdf8c9` (`chore: 修改前快照 before integrating Aitor TRC methods`).
- Trigger: Aitor provided two sibling-folder TRC files in `../10 Aitor/`: `fastsam3d_2.trc` for unfiltered FastSAM3D and `merged_output_2.trc` for the ViscandoXFastSAM3D Merge approach.
- Code changes:
  - `01_compute_elbow_deltas.py`: added TRC ingestion, COCO-17 marker mapping, unit conversion (`mm -> cm`), optional `--fastsam-trc`, `--merge-trc`, `--extra-trc NAME=PATH`, and `--skip-afh`.
  - `FastSAM3D` has `3015` frames, matching left-camera metadata, so it is aligned by synced left-frame indices to the `2801`-frame shared timeline.
  - `Merge` has `2801` frames, matching the synced pose timeline, so it is aligned by synced frame index.
  - `02_plot_delta_curves.py`, `03_segment_rom_eval.py`, and `04_phase4_ablation.py` now infer systems dynamically instead of assuming only `SKT/AFH/Xsens`.
- Main command:
  - `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --skip-afh --fastsam-trc '../10 Aitor/fastsam3d_2.trc' --merge-trc '../10 Aitor/merged_output_2.trc' --enable-quality-filter --smooth-method moving_average --smooth-window-ms 200 --wrist-smooth-radius 0 --out-dir 05_frame_delta_eval/results/phase4_aitor_trc_methods_ma200_quality`
  - Output: `05_frame_delta_eval/results/phase4_aitor_trc_methods_ma200_quality/`.
- Segment/DTW command:
  - `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/03_segment_rom_eval.py --combined-csv 05_frame_delta_eval/results/phase4_aitor_trc_methods_ma200_quality/elbow_delta_combined.csv --out-dir 05_frame_delta_eval/results/phase4_aitor_trc_segment_rom_dtw_ma200_quality`
  - Output: `05_frame_delta_eval/results/phase4_aitor_trc_segment_rom_dtw_ma200_quality/`.
- K6 frame-delta agreement against XsensFair:
  - Left: SKT Pearson `0.413`, FastSAM3D `0.587`, Merge `0.027`, XsensNative anchor `0.988`.
  - Right: SKT Pearson `0.398`, FastSAM3D `0.736`, Merge `0.084`, XsensNative anchor `0.996`.
  - FastSAM3D has lower K6 path ratio than SKT: Left `1.153` vs `1.761`, Right `1.135` vs `1.764`.
- Segment ROM/RULA/DTW under 200 ms moving-average policy:
  - Segment count: Left `8`, Right `9`.
  - Left ROM MAE / DTW median: SKT `19.31 deg / 0.0163`, FastSAM3D `13.47 deg / 0.0087`, Merge `16.45 deg / 0.0211`, XsensNative anchor `1.34 deg / 0.0026`.
  - Right ROM MAE / DTW median: SKT `21.33 deg / 0.0186`, FastSAM3D `9.68 deg / 0.0095`, Merge `20.84 deg / 0.0177`, XsensNative anchor `1.70 deg / 0.0018`.
  - RULA-like bin agreement: SKT Left/Right `62.5%/66.7%`; FastSAM3D `62.5%/33.3%`; Merge `50.0%/55.6%`; XsensNative anchor `87.5%/88.9%`.
- Sensitivity run:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/04_phase4_ablation.py --skip-afh --fastsam-trc '../10 Aitor/fastsam3d_2.trc' --merge-trc '../10 Aitor/merged_output_2.trc' --enable-quality-filter --windows-ms 100,200,300 --activity-thresholds 8,10,12 --dtw-preprocesses mean_l2,mean,none --out-dir 05_frame_delta_eval/results/phase4_aitor_trc_ablation_quality`
  - Output table: `05_frame_delta_eval/results/phase4_aitor_trc_ablation_quality/headline_table.md`.
  - Note: on the 12.5 fps timeline, `200 ms` and `300 ms` both round to the same 3-frame centered window (`~240 ms`), so their filter-window rows are identical.
- 400 ms diagnostic:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --skip-afh --fastsam-trc '../10 Aitor/fastsam3d_2.trc' --merge-trc '../10 Aitor/merged_output_2.trc' --enable-quality-filter --smooth-method moving_average --smooth-window-ms 400 --wrist-smooth-radius 0 --out-dir 05_frame_delta_eval/results/phase4_aitor_trc_methods_ma400_quality --skip-plots`
  - Segment command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/03_segment_rom_eval.py --combined-csv 05_frame_delta_eval/results/phase4_aitor_trc_methods_ma400_quality/elbow_delta_combined.csv --out-dir 05_frame_delta_eval/results/phase4_aitor_trc_segment_rom_dtw_ma400_quality --skip-plots`
  - 400 ms maps to a 5-frame centered window. K6 Pearson becomes SKT Left/Right `0.433/0.418`, FastSAM3D `0.590/0.744`, Merge `0.029/0.081`.
  - Segment ROM MAE / DTW median: SKT Left `16.40 deg / 0.0149`, Right `21.82 deg / 0.0155`; FastSAM3D Left `13.94 deg / 0.0080`, Right `10.67 deg / 0.0090`; Merge Left `17.33 deg / 0.0211`, Right `20.93 deg / 0.0170`.
- Interpretation:
  - The new unfiltered FastSAM3D TRC looks stronger than SKT and Merge on most motion-shape and segment-ROM metrics in this single-session elbow evaluation.
  - Merge has near-zero K6 Pearson despite path ratios near `1.1`, suggesting it has roughly similar motion amount but poor signed temporal/shape agreement with XsensFair in this setup.
  - XsensNative vs XsensFair remains a useful reference-system anchor/floor, but Xsens should still be described as an external comparison system rather than absolute ground truth.

## 2026-05-16 - Phase 4 unified filtering and DTW evaluation

- Snapshot before experiment: `ba20e15` (`修改前快照 before Phase 4 evaluation updates`).
- Trigger: Aitor/Amrit feedback identified two methodological issues in the previous 05 evaluation: mixed source-side smoothing levels and residual time-sync uncertainty.
- Code changes:
  - `05_frame_delta_eval/src/01_compute_elbow_deltas.py`: added Phase 4 per-system smoothing policy.
  - XsensFair/XsensNative now receive no extra project smoothing after interpolation to the shared timeline.
  - SKT/AFH camera systems now use centered moving average before delta calculation; default request is `200 ms`.
  - Current 12.5 fps timeline maps `200 ms` to a 3-frame centered window (`~240 ms`, radius `1`).
  - Legacy `--smooth-method median`, `--smooth-radius`, and `--wrist-smooth-radius` are retained for reproducibility; Phase 4 default wrist median is `0`.
  - Added `--afh-filter-status` so old AFH outputs are explicitly marked provisional until Aitor provides the unfiltered FastSAM3D/AFH NPZ.
  - `05_frame_delta_eval/src/03_segment_rom_eval.py`: added per-segment DTW shape agreement, `segment_dtw.csv`, and DTW summaries in `segment_rom_summary.json`.
  - `05_frame_delta_eval/src/04_phase4_ablation.py`: added a no-PNG sensitivity runner for filter window, activity threshold, and DTW preprocessing.
- Main Phase 4 run:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --smooth-method moving_average --smooth-window-ms 200 --wrist-smooth-radius 0 --afh-filter-status unknown_butterworth --out-dir 05_frame_delta_eval/results/phase4_ma200_unfiltered_xsens --skip-plots`
  - Plot command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/02_plot_delta_curves.py --combined-csv 05_frame_delta_eval/results/phase4_ma200_unfiltered_xsens/elbow_delta_combined.csv --summary-json 05_frame_delta_eval/results/phase4_ma200_unfiltered_xsens/elbow_delta_summary.json --out-dir 05_frame_delta_eval/results/phase4_ma200_unfiltered_xsens`
  - Segment/DTW command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/03_segment_rom_eval.py --combined-csv 05_frame_delta_eval/results/phase4_ma200_unfiltered_xsens/elbow_delta_combined.csv --out-dir 05_frame_delta_eval/results/phase4_segment_rom_dtw_ma200`
- Main K-delta results under unified filter:
  - SKT vs XsensFair K6 Pearson: Left `0.310`, Right `0.282`.
  - SKT K6 path ratio: Left `1.912`, Right `1.899`.
  - XsensNative vs XsensFair remains the anchor/floor; no extra project smoothing is applied to either Xsens stream.
  - AFH rows are not final because this NPZ may still include upstream 6 Hz Butterworth filtering.
- Segment ROM/RULA/DTW under unified filter:
  - Segment count: Left `8`, Right `9`.
  - XsensNative vs XsensFair anchor: ROM MAE Left `1.34 deg`, Right `1.70 deg`; DTW median Left `0.0026`, Right `0.0018`; RULA agreement Left `87.5%`, Right `88.9%`.
  - SKT vs XsensFair: ROM MAE Left `21.09 deg`, Right `30.56 deg`; DTW median Left `0.0153`, Right `0.0164`; RULA agreement Left `62.5%`, Right `66.7%`.
  - AFH vs XsensFair provisional: ROM MAE Left `17.70 deg`, Right `14.58 deg`; DTW median Left `0.0105`, Right `0.0127`; RULA agreement Left `62.5%`, Right `44.4%`.
- Quality-filter diagnostic under the same Phase 4 smoothing policy:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --enable-quality-filter --smooth-method moving_average --smooth-window-ms 200 --wrist-smooth-radius 0 --afh-filter-status unknown_butterworth --out-dir 05_frame_delta_eval/results/phase4_ma200_quality_unfiltered_xsens --skip-plots`
  - Segment/DTW command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/03_segment_rom_eval.py --combined-csv 05_frame_delta_eval/results/phase4_ma200_quality_unfiltered_xsens/elbow_delta_combined.csv --out-dir 05_frame_delta_eval/results/phase4_segment_rom_dtw_ma200_quality`
  - Quality filtering improves SKT K6 Pearson to Left `0.413`, Right `0.398`, and reduces K6 path ratio to Left `1.761`, Right `1.764`.
  - SKT segment ROM MAE becomes Left `19.31 deg`, Right `21.33 deg`; DTW median Left `0.0163`, Right `0.0186`; RULA agreement Left `62.5%`, Right `66.7%`.
  - Quiet-frame K1 std remains above the planned `<5 deg/frame` validation target: Left `7.02`, Right `6.62`.
- 400 ms smoothing diagnostic with quality filtering:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --enable-quality-filter --smooth-method moving_average --smooth-window-ms 400 --wrist-smooth-radius 0 --afh-filter-status unknown_butterworth --out-dir 05_frame_delta_eval/results/phase4_ma400_quality_diagnostic --skip-plots`
  - 400 ms maps to a 5-frame centered window and reaches the quiet-frame validation target: K1 quiet std Left `4.37`, Right `4.49`.
  - SKT K6 Pearson is Left `0.433`, Right `0.418`; K6 path ratio Left `1.553`, Right `1.562`.
  - SKT segment ROM MAE is Left `16.40 deg`, Right `21.82 deg`; DTW median Left `0.0149`, Right `0.0155`; RULA agreement unchanged at Left `62.5%`, Right `66.7%`.
- Sensitivity run:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/04_phase4_ablation.py --out-dir 05_frame_delta_eval/results/phase4_ablation_ma_windows`
  - Output: `05_frame_delta_eval/results/phase4_ablation_ma_windows/headline_table.md`.
  - Filter window: `100 ms` maps to 1 frame and leaves high jitter; SKT ROM MAE Left/Right `38.35/33.30 deg`. `200 ms` and `300 ms` both map to the same 3-frame window on this timeline, so their results are identical.
  - Activity threshold is sensitive because lower thresholds merge nearby active periods while higher thresholds split them: threshold `8 deg` gives Left/Right `5/7` segments, `10 deg` gives `8/9`, and `12 deg` gives `13/15`.
  - DTW preprocessing: `mean_l2` gives the most interpretable floor because XsensNative vs XsensFair DTW stays near zero. Without L2 normalization, DTW is dominated by amplitude/offset scale.
- Interpretation:
  - Previous Weekly Summary numbers should not be reported as final Phase 4 results because old runs mixed Xsens internal smoothing, AFH upstream Butterworth smoothing, and project median filters.
  - The unified filter run supports Aitor/Amrit's concern: filtering choices materially affect SKT ROM MAE, so this must be discussed in the report.
  - Turning off project smoothing on Xsens changes full-sequence XsensFair ROM substantially, especially on the left elbow, which confirms that the previous median filter had a non-trivial amplitude-slimming effect and should not be stacked on Xsens.
  - DTW is now available as a shape-agreement metric, but it should be reported alongside ROM MAE and RULA agreement rather than replacing them.
  - The next blocker is receiving Aitor's unfiltered AFH NPZ; until then, AFH metrics are diagnostic/provisional only.

## 2026-05-10 - AFH timing synchronization fix and rerun

- Snapshot before experiment: `505264e` (`修改前快照 before AFH timing synchronization fix`).
- Goal: fix AFH timing at the source before comparing SKT/AFH/Xsens motion metrics.
- Code changes:
  - Updated `04_hybrid_afh1/src/03_calibrate_rotation.py` so EasyErgo-to-stereo rotation calibration queries EasyErgo time through the affine timing model.
  - Updated `04_hybrid_afh1/src/04_combine_hybrid.py` so AFH v1 skeleton generation maps `stereo_t -> xsens_t -> easyergo_t` before interpolation.
  - Timing model used: `xsens_t = 1.0102 * easyergo_t - 16.83`, and `xsens_t = stereo_t - 17.25`.
- Regenerated AFH outputs:
  - Old files backed up with `_pre_timing_fix` suffix.
  - Rotation residual after fix: mean `11.34 cm`, p95 `23.97 cm`.
  - AFH source method now records `AFH1_v1_time_aligned`.
  - EasyErgo query range: `-0.42 s` to `246.00 s`; out-of-bounds frame count `45`.
- Full-sequence K-delta rerun:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --enable-quality-filter --k-frame-list 1,6,12,25 --out-dir 05_frame_delta_eval/results/k_delta_sweep_full_afh_time_aligned`
  - AFH vs XsensFair Pearson improved: Left k6 `0.054 -> 0.300`, Left k12 `0.075 -> 0.406`, Right k6 `-0.127 -> 0.287`, Right k12 `-0.131 -> 0.330`.
  - AFH RME improved: Left k6 `11.26 -> 8.89 deg`, Left k12 `17.02 -> 12.58 deg`, Right k6 `13.82 -> 9.82 deg`, Right k12 `21.73 -> 14.56 deg`.
- Segment ROM/RULA rerun:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/03_segment_rom_eval.py --combined-csv 05_frame_delta_eval/results/k_delta_sweep_full_afh_time_aligned/elbow_delta_combined.csv --activity-threshold-deg 10.0 --min-duration-s 1.5 --min-xsens-rom-deg 15.0 --rula-bins 60,100 --out-dir 05_frame_delta_eval/results/segment_rom_afh_time_aligned`
  - AFH ROM MAE: Left `9.50 deg`, Right `13.12 deg`.
  - AFH ROM Pearson: Left `0.915`, Right `0.830`.
  - AFH RULA agreement: Left `41.7%`, Right `33.3%`.
  - SKT metrics unchanged: ROM MAE Left `14.38 deg`, Right `20.37 deg`; RULA agreement Left `58.3%`, Right `53.3%`.
- Low-jitter rerun:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --enable-quality-filter --k-frame-list 1,6,12,25 --start-time 22.48 --end-time 43.84 --out-dir 05_frame_delta_eval/results/k_delta_sweep_low_jitter_afh_time_aligned`
  - AFH low-jitter Pearson after timing fix: Left k12 `0.759`, Right k12 `0.910`.
  - Residual lag after fix is near zero: full sequence AFH best lag is `0-2` frames; low-jitter AFH best lag is `-2..0` frames depending on side/K.
- Interpretation:
  - The previous near-zero/negative AFH dynamic Pearson was largely a timing synchronization artifact.
  - After source-level timing correction, AFH has stronger ROM amplitude agreement than before, but RULA agreement remains lower than SKT.
  - AFH should not be judged using the pre-fix evaluation outputs; use `*_afh_time_aligned` result folders for current comparisons.

## 2026-05-05 - Lag-aligned low-jitter delta diagnostic

- Snapshot before experiment: `6f17334` (`chore: 修改前快照 before lag-aligned delta diagnostic`).
- Goal: visually check whether residual timing offsets explain poor delta agreement in the low-jitter window.
- Input: `05_frame_delta_eval/results/k_delta_sweep_low_jitter/elbow_delta_combined.csv`.
- Output directory: `05_frame_delta_eval/results/lag_aligned_low_jitter/`.
- K=1 lag alignment:
  - Plot: `plot_lag_aligned_k1_delta_low_jitter.png`.
  - SKT Left: best lag `+1` frame, Pearson `0.414 -> 0.546`.
  - SKT Right: best lag `0` frame, Pearson unchanged `0.378`.
  - AFH Left: best lag `-8` frames, Pearson `0.015 -> 0.408`.
  - AFH Right: best lag `-9` frames, Pearson `-0.126 -> 0.484`.
- K=12 lag alignment:
  - Plot: `plot_lag_aligned_k12_delta_low_jitter.png`.
  - SKT Left: best lag `0` frames, Pearson unchanged `0.741`.
  - SKT Right: best lag `-2` frames, Pearson `0.525 -> 0.585`.
  - AFH Left: best lag `-10` frames, Pearson `-0.043 -> 0.704`.
  - AFH Right: best lag `-10` frames, Pearson `-0.197 -> 0.869`.
- Interpretation:
  - SKT is already close to the current XsensFair timeline in this low-jitter window, with only small residual lag.
  - AFH shows a much larger timing mismatch of roughly `8-10` frames (`0.64-0.80 s` at 12.5 fps). Lag alignment substantially improves AFH delta agreement, especially at K=12.
  - This is diagnostic only: curves are shifted after the fact using XsensFair, so these numbers should not be reported as raw pipeline performance.

## 2026-05-04 - K-frame delta and segment ROM motion agreement

- Snapshot before experiment: `366e849` (`chore: 修改前快照 before k-frame motion eval`).
- Goal: move beyond single-frame delta because frame-to-frame differencing is too sensitive to keypoint jitter. Two coarser motion-space evaluations were implemented:
  - K-frame delta: `angle[i] - angle[i-K]` for `K = 1, 6, 12, 25`.
  - Activity-segment ROM: XsensFair-detected motion segments, then per-segment ROM / peak / RULA-like bin agreement.
- Code changes:
  - `05_frame_delta_eval/src/01_compute_elbow_deltas.py`: added `--k-frame-list`; summary/CSV now include per-K deltas, validity, anomalies, and pair metrics.
  - `05_frame_delta_eval/src/02_plot_delta_curves.py`: added per-K delta/scatter/cumulative plots and Pearson-vs-K headline plots.
  - `05_frame_delta_eval/src/03_segment_rom_eval.py`: new segment ROM evaluation script with timeline, scatter, bars, and RULA-like confusion plots.
- K=1 reproduction check:
  - `k_delta_sweep_full` K=1 matches `optimized_quality_filter_balanced` exactly for checked Pearson, active Pearson, slope, path ratio, and RMSE metrics.
- K-delta full sequence:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --enable-quality-filter --k-frame-list 1,6,12,25 --out-dir 05_frame_delta_eval/results/k_delta_sweep_full`
  - SKT vs XsensFair Left Pearson: `k1=0.284`, `k6=0.503`, `k12=0.564`, `k25=0.600`.
  - SKT vs XsensFair Right Pearson: `k1=0.285`, `k6=0.422`, `k12=0.526`, `k25=0.612`.
  - XsensNative vs XsensFair remains high across K: Left `0.977-0.992`, Right `0.988-0.997`.
- K-delta low-jitter window:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --enable-quality-filter --k-frame-list 1,6,12,25 --start-time 22.48 --end-time 43.84 --out-dir 05_frame_delta_eval/results/k_delta_sweep_low_jitter`
  - SKT vs XsensFair Left Pearson: `k1=0.414`, `k6=0.686`, `k12=0.741`, `k25=0.681`.
  - SKT vs XsensFair Right Pearson: `k1=0.378`, `k6=0.460`, `k12=0.525`, `k25=0.552`.
  - Interpretation: K-frame delta is a valid improvement over single-frame delta. The left elbow reaches the expected `0.7+` range at `K=12`; the right elbow improves but remains below expectation, indicating real right-elbow tracking limitations rather than just frame-level noise.
- Segment ROM:
  - Default segment detection was adjusted to `merge_gap_s=2.0` after initial `0.5s` merging produced too many fragmented segments.
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/03_segment_rom_eval.py --combined-csv 05_frame_delta_eval/results/k_delta_sweep_full/elbow_delta_combined.csv --activity-threshold-deg 10.0 --min-duration-s 1.5 --min-xsens-rom-deg 15.0 --rula-bins 60,100 --out-dir 05_frame_delta_eval/results/segment_rom`
  - Detected segments: Left `12`, Right `16`.
  - SKT vs XsensFair ROM Pearson: Left `0.883`, Right `0.814`.
  - SKT vs XsensFair ROM MAE: Left `14.38 deg`, Right `20.37 deg`.
  - SKT vs XsensFair RULA-like direct bin agreement: Left `0.583`, Right `0.533`; off-by-one agreement `1.000` for both elbows.
  - AFH vs XsensFair ROM Pearson: Left `0.791`, Right `0.943`; direct bin agreement remains lower (`0.333`, `0.467`).
  - XsensNative vs XsensFair ROM Pearson: Left `0.994`, Right `0.994`, confirming segment ROM is internally stable for the Xsens-derived references.
- Interpretation:
  - K-frame delta and segment ROM are more useful than raw frame-to-frame delta for reporting motion agreement.
  - K=12 is a good compromise for K-delta reporting: it reduces single-frame jitter while preserving temporal motion structure.
  - Segment ROM gives the strongest application-facing result because it maps better to RULA-style range/peak-angle reasoning.
  - RULA-like bin agreement is still moderate, so it should be presented as agreement with the Xsens-derived comparison system, not as proof of absolute ergonomic-score correctness.

## 2026-05-04 - Smoothing sweep for frame-delta jitter control

- Goal: check whether temporal smoothing is practically necessary for frame-to-frame elbow motion evaluation, and identify a reasonable smoothing strength rather than over-smoothing the motion.
- Constant settings:
  - SKT elbow-chain quality filter enabled.
  - Quality thresholds: `min_conf=0.20`, `epipolar<=10 px`, `reprojection<=10 px`.
  - Xsens offset source: `position`, offset `17.25 s`.
- Sweep commands:
  - `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --enable-quality-filter --smooth-radius 0 --wrist-smooth-radius 0 --skip-plots --out-dir 05_frame_delta_eval/results/smoothing_sweep_r0_w0_quality`
  - `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --enable-quality-filter --smooth-radius 2 --wrist-smooth-radius 1 --skip-plots --out-dir 05_frame_delta_eval/results/smoothing_sweep_r2_w1_quality`
  - `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --enable-quality-filter --smooth-radius 4 --wrist-smooth-radius 3 --skip-plots --out-dir 05_frame_delta_eval/results/smoothing_sweep_r4_w3_quality`
  - `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --enable-quality-filter --smooth-radius 8 --wrist-smooth-radius 5 --skip-plots --out-dir 05_frame_delta_eval/results/smoothing_sweep_r8_w5_quality`
  - `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --enable-quality-filter --smooth-radius 12 --wrist-smooth-radius 7 --skip-plots --out-dir 05_frame_delta_eval/results/smoothing_sweep_r12_w7_quality`
- Summary artifacts:
  - `05_frame_delta_eval/results/smoothing_sweep_summary.csv`
  - `05_frame_delta_eval/results/smoothing_sweep_metrics.png`
- Key SKT vs XsensFair results:
  - `r0_w0`: Pearson Left `0.172`, Right `0.187`; path ratio Left `4.03`, Right `3.72`; quiet-frame std Left `14.14`, Right `12.01 deg/frame`.
  - `r2_w1`: Pearson Left `0.257`, Right `0.277`; path ratio Left `2.17`, Right `2.12`; quiet-frame std Left `6.34`, Right `6.66 deg/frame`.
  - `r4_w3`: Pearson Left `0.284`, Right `0.285`; path ratio Left `1.81`, Right `1.77`; quiet-frame std Left `4.26`, Right `4.68 deg/frame`.
  - `r8_w5`: Pearson Left `0.288`, Right `0.256`; path ratio Left `1.65`, Right `1.42`; quiet-frame std Left `2.40`, Right `2.86 deg/frame`.
  - `r12_w7`: Pearson Left `0.204`, Right `0.226`; path ratio Left `1.43`, Right `1.49`; quiet-frame std Left `2.15`, Right `2.19 deg/frame`.
- Interpretation:
  - Smoothing is necessary for this metric: without smoothing, SKT accumulates roughly `3.7-4.0x` the XsensFair angular path and quiet-frame jitter is extremely high.
  - Moderate smoothing (`smooth_radius=4`, `wrist_smooth_radius=3`) gives the best overall trade-off for reporting: it substantially reduces jitter while preserving both elbows' dynamic correlation.
  - Stronger smoothing (`8-12` frames) continues to lower quiet-frame noise but starts compressing motion amplitude and can reduce right-elbow Pearson/slope, so it should be treated as over-smoothing for the main comparison.

## 2026-05-04 - Frame-delta evaluation optimization pass

- Snapshot before experiment: `5e7c52e` (`chore: 修改前快照 before frame delta eval optimization`).
- Scripts changed:
  - `05_frame_delta_eval/src/01_compute_elbow_deltas.py`
  - `05_frame_delta_eval/src/02_plot_delta_curves.py`
- Goal: improve the frame-to-frame elbow motion agreement evaluation after the initial result underperformed the expected Pearson target.
- Plan review: the proposed diagnosis was mostly reasonable. The strongest implementation-level issues were the disabled wrist smoothing, weak smoothing for a 12.5 fps timeline, overly loose anomaly threshold, and missing diagnostics for active-motion / quiet-frame noise / residual lag.
- Code changes:
  - Added `--wrist-smooth-radius` and changed its default to `3` frames.
  - Changed shared-timeline `--smooth-radius` default from `2` to `4` frames.
  - Changed `--anomaly-delta-deg` default from `60` to `30`.
  - Added active-motion metrics using `--active-delta-threshold` default `1.0 deg/frame`.
  - Added quiet-frame noise-floor metrics using `--noise-floor-threshold` default `0.5 deg/frame`.
  - Added `--lag-window-frames` residual lag sweep, default `10` frames.
  - Added optional SKT elbow-chain quality filtering via `--enable-quality-filter`.
  - Added active-motion scatter plots and included active/lag metrics in `plot_index.md`.
- Baseline reference from initial default run:
  - SKT vs XsensFair Pearson: Left `0.147`, Right `0.169`.
  - SKT path ratio: Left `2.40`, Right `2.40`.
- Optimized smoothing only:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --out-dir 05_frame_delta_eval/results/optimized_wrist_smooth`
  - SKT vs XsensFair Pearson: Left `0.209`, Right `0.201`.
  - Active-motion Pearson: Left `0.273`, Right `0.316`.
  - SKT path ratio: Left `2.17`, Right `2.02`.
  - Quiet-frame SKT delta std: Left `5.23 deg/frame`, Right `6.68 deg/frame`.
- Quality-filter sweep:
  - Strict filter (`min_conf=0.30`, `epi<=5 px`, `reproj<=5 px`) improved Pearson to Left `0.258`, Right `0.306`, but reduced valid delta ratio to Left `0.732`, Right `0.806`.
  - Balanced filter (`min_conf=0.20`, `epi<=10 px`, `reproj<=10 px`) gave the best trade-off: Left Pearson `0.284`, Right Pearson `0.285`, valid delta ratio Left `0.764`, Right `0.844`.
  - Confidence-only filtering preserved more right-elbow frames but did not reduce right-elbow path noise enough.
- Final recommended optimized run:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --enable-quality-filter --out-dir 05_frame_delta_eval/results/optimized_quality_filter_balanced`
  - Quality thresholds: `min_conf=0.20`, `epi<=10 px`, `reproj<=10 px`.
  - SKT vs XsensFair Pearson: Left `0.284`, Right `0.285`.
  - Active-motion Pearson: Left `0.377`, Right `0.391`.
  - SKT path ratio: Left `1.81`, Right `1.77`.
  - Quiet-frame SKT delta std: Left `4.26 deg/frame`, Right `4.68 deg/frame`.
  - Residual lag sweep: Left best lag `+1` frame (`r=0.294`), Right best lag `0` frame (`r=0.285`).
- Interpretation:
  - The optimization direction is valid: smoothing and quality filtering both reduce SKT delta noise and improve correlation.
  - The result is still below the original `0.7` Pearson expectation, so the current frame-delta metric should be reported as a diagnostic rather than a solved validation result.
  - Active-motion Pearson is meaningfully higher than whole-sequence Pearson, supporting the earlier diagnosis that quiet frames dilute correlation.
  - Quality filtering improves agreement but reduces coverage; the balanced thresholds are preferable to the strict plan thresholds for reporting.
  - XsensNative vs XsensFair remains very high after optimization, so the previous conclusion about delta-space invariance to Xsens absolute offset still holds.
- Optimized N-to-T diagnostic segment:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --enable-quality-filter --start-time 17.3 --end-time 22.2 --out-dir 05_frame_delta_eval/results/n_to_t_17_22_optimized_quality`
  - Frames: `57`, duration `4.8 s`.
  - SKT vs XsensFair Pearson: Left `0.323`, Right `0.186`.
  - SKT path ratio: Left `1.36`, Right `1.32`.
  - Interpretation: the N-to-T window improves over the initial short-window result (`0.278 / 0.085`), especially on right elbow, but remains too short for strong statistical claims.

## 2026-05-02 - Initial elbow motion-delta evaluation

- Snapshot before experiment: `9abfd10` (`修改前快照 before frame delta motion evaluation`).
- Scripts:
  - `05_frame_delta_eval/src/01_compute_elbow_deltas.py`
  - `05_frame_delta_eval/src/02_plot_delta_curves.py`
- Goal: evaluate motion agreement instead of absolute angle MAE, following Aitor/Amrit's frame-to-frame elbow-angle delta recommendation.
- Key correction: the original SKT/AFH NPZ timestamps are not monotonic (`241` non-positive diffs), so the scripts rebuild a corrected stereo-video timeline from metadata using `seconds + microseconds * 1e-6`.
- Method: sample SKT, AFH, XsensFair, and XsensNative onto the same video time axis first, then interpolate short gaps, median-smooth on that shared axis, and compute `angle[i] - angle[i-1]`.
- Default run:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py`
  - Offset: `17.25 s` (`position_best_offset_seconds`)
  - Smooth radius: `2` frames
  - Max interpolation gap: `5` frames
  - Output: `05_frame_delta_eval/results/`
  - Left elbow SKT vs XsensFair: Pearson `0.147`, slope `0.473`, path ratio `2.401`.
  - Right elbow SKT vs XsensFair: Pearson `0.169`, slope `0.498`, path ratio `2.396`.
  - Left elbow AFH vs XsensFair: Pearson `0.026`, slope `0.031`, path ratio `1.054`.
  - Right elbow AFH vs XsensFair: Pearson `-0.106`, slope `-0.115`, path ratio `1.005`.
- Offset sensitivity:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --offset-source angle --out-dir 05_frame_delta_eval/results/offset_angle_17_40`
  - Offset: `17.40 s` (`angle_refined_offset_seconds`)
  - Left elbow SKT vs XsensFair improves to Pearson `0.183`, slope `0.586`; right elbow SKT Pearson changes to `0.150`.
- N-to-T diagnostic segment:
  - Command: `/opt/anaconda3/envs/pose/bin/python 05_frame_delta_eval/src/01_compute_elbow_deltas.py --start-time 17.3 --end-time 22.2 --out-dir 05_frame_delta_eval/results/n_to_t_17_22`
  - Frames: original indices `190-245`, duration `4.8 s`.
  - Left elbow SKT vs XsensFair: Pearson `0.278`, slope `0.503`, path ratio `1.853`.
  - Right elbow SKT vs XsensFair: Pearson `0.085`, slope `0.198`, path ratio `2.734`.
  - Offset `17.40 s` version: `05_frame_delta_eval/results/n_to_t_17_22_offset_angle_17_40/`.
- Interpretation:
  - The method runs and produces the intended motion-agreement artifacts.
  - Whole-sequence SKT delta is still noisy: cumulative angular path is roughly `2.4x` XsensFair for both elbows.
  - AFH's total path ratio is near `1.0`, but its signed delta correlation is weak, so path magnitude alone is not enough; Pearson/slope and visual curves are needed together.
  - XsensNative and XsensFair agree strongly in delta space (Pearson `0.98-0.99`), so the fair/native difference matters less for frame-to-frame elbow motion than for absolute angle comparisons.
