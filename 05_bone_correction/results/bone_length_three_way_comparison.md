# Bone Length Three-Way Comparison

## Definition

- `Xsens` uses segment-origin proxies closest to the COCO-17 skeleton semantics:
  `shoulder_width = dist(LeftUpperArm, RightUpperArm)`
  `hip_width = dist(LeftUpperLeg, RightUpperLeg)`
  `torso = dist(mid(UpperArms), mid(UpperLegs))`
  limb lengths use adjacent segment origins.

## Comparison Table

| Segment | Xsens (cm) | Stereo (cm) | EasyErgo (cm) | Stereo/Xsens | EasyErgo/Xsens |
|---------|------------|-------------|---------------|--------------|----------------|
| shoulder_width | 33.81 | 30.27 | 47.10 | 0.895 | 1.393 |
| hip_width | 15.95 | 23.01 | 22.18 | 1.443 | 1.391 |
| torso | 45.07 | 49.22 | 68.99 | 1.092 | 1.531 |
| upper_arm | 30.17 | 22.82 | 36.92 | 0.756 | 1.224 |
| forearm | 24.66 | 24.87 | 33.59 | 1.009 | 1.362 |
| thigh | 41.11 | 31.80 | 54.95 | 0.773 | 1.336 |
| shank | 40.11 | 43.17 | 51.40 | 1.076 | 1.282 |

## Verdict

- Classification: `mixed_or_inconclusive`
