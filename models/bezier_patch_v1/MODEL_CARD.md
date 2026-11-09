# PatchBezier-v1 / generator v2

Locally trained from scratch for the Chromatic Vector experimental `neural` contour method. This directory contains the PyTorch state dictionary, exported ONNX and metadata. `v1` names the architecture; the released training run uses synthetic generator version 2. No photo fixtures, external datasets, pretrained weights or online inference service were used.

## Architecture and data

Input: 5×64×64 canonical patch (RGB, noisy region-A mask, raw boundary) and 8 context scalars. Three strided convolutions, ReLU, flattened features plus context, a 128-unit hidden layer and 6 outputs. Four decoded values specify cubic handles relative to fixed endpoints (0,0), (1,0); two logits classify corners and artificial seams. Model size is approximately 1.1 MB.

Synthetic mix: smooth monotone cubic boundaries 55%, corners 15%, artificial weak seams 15%, real small objects 15%, with randomized colors, gradients, noise and pixelation. Geometry loss is applied only to valid smooth boundaries. Training loss combines differentiable soft-coverage raster L1, handle error, binary classification loss and a small bending penalty. It does not use an arbitrary closed-path differentiable SVG renderer.

18,000 training samples (seed 23), 1600 validation (1234), 1600 independent test (198761); AdamW, batch 128, learning rate .001, 14 epochs, CPU with 3 Torch threads. Checkpoint selected at epoch 12 by minimum validation smooth-mask MAE. Total elapsed 139.93 s in the recorded environment. See metadata.json for exact history, source hashes and versions.

## Independent synthetic test

| Metric | Value |
|---|---:|
| Smooth soft-mask MAE, CNN | 0.00730234 |
| Smooth soft-mask MAE, straight reference | 0.05529191 |
| Corner accuracy | 0.950625 |
| Artificial-seam accuracy | 0.991875 |
| ONNX/OpenCV vs PyTorch maximum absolute output difference | 0.0000038147 |

Classification accuracy at .5 is not confidence calibration or evidence that .97-confidence merges are reliable on photographs. The production merge threshold is .97 and independent hard guards apply.

SHA-256 ONNX: `6a449c7ba955f97c8968b2e906bbe087cdeba5f56259423b3748a73b56adc508`.

## Intended use and limits

Small local proposals for editable vector boundaries. OpenCV CPU inference, followed by bounded source-color optimization and geometry/topology checks. Original shared endpoints remain fixed; tangent continuity is not guaranteed. Learned weights are supplied, checksum-verified and never silently substituted with random weights.

The network has synthetic-to-photo domain shift and no semantic/global image encoder. It cannot by itself remove every square artifact or merge an entire object into an artist-like vector. The pipeline may retain classical curves, especially for short boundaries, corners, alpha edges and corrections. Refinement overlays remain classical. All three photographic examples had zero accepted neural region unions. Photographic RGB error improves modestly; full-image edge F1 slightly decreases. Consult README_NEURAL.md and saved actual SVG comparisons before assessing visual quality.
