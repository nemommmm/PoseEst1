# SKT + FastSAM3D Unfiltered Prior Fusion

## Setup

- SKT input: `01_stereo_triangulation/results/historical_best_20260324/recovered_baseline/optimized_pose.npz`
- Prior input: `../10 Aitor/fastsam3d_2.trc`
- Prior conversion output: `01_stereo_triangulation/results/skt_model_fusion/fastsam3d_unfiltered_prior.npz`
- Fused output: `01_stereo_triangulation/results/skt_model_fusion/fastsam_prior_fusion/skt_fastsam_prior_fused.npz`

The FastSAM3D TRC was aligned with the corrected stereo timeline using `left_metadata_frame_index`, because the TRC frame count matches the left-camera metadata length rather than the synchronized SKT pair count.

## Result

On common valid frames, SKT + FastSAM3D prior fusion improves dynamic agreement moderately:

- Left elbow K=6 Pearson vs XsensFair: `0.310 -> 0.384`
- Right elbow K=6 Pearson vs XsensFair: `0.282 -> 0.328`
- Left elbow K=6 Pearson vs FastSAM3D: `0.257 -> 0.402`
- Right elbow K=6 Pearson vs FastSAM3D: `0.254 -> 0.351`
- Left elbow K=1 high-delta count vs XsensFair frame set: `40 -> 21`
- Right elbow K=1 high-delta count vs XsensFair frame set: `39 -> 29`

## Interpretation

This route is promising as a quality-gated stabilization layer for SKT, especially for short high-delta jitter. It is not yet a full replacement for the stronger standalone FastSAM3D results.
