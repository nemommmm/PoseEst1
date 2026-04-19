# Direction-Preserving Bone Rescale Sanity Check

## Target lengths (cm)

- hip_width: 22.18
- shoulder_width: 47.10
- torso: 68.99
- upper_arm: 36.92
- forearm: 33.59
- thigh: 54.95
- shank: 51.40

## MAE comparison

- Original end-to-end MAE: 18.36 deg
- Rescaled end-to-end MAE: 18.59 deg
- Delta end-to-end MAE: +0.23 deg
- Original fair MAE: 17.47 deg
- Rescaled fair MAE: 17.48 deg
- Delta fair MAE: +0.02 deg

## Per-angle absolute change caused by rescale

| Angle | Mean abs delta (deg) | P95 abs delta (deg) |
|-------|----------------------|---------------------|
| LeftShoulder | 0.00 | 0.00 |
| RightShoulder | 0.00 | 0.00 |
| LeftElbow | 0.00 | 0.00 |
| RightElbow | 0.00 | 0.00 |
| LeftHip | 3.07 | 5.92 |
| RightHip | 2.91 | 6.01 |
| LeftKnee | 0.00 | 0.00 |
| RightKnee | 0.00 | 0.00 |
