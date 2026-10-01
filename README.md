# Active Perception for Enhanced Visual Signal Recovery

Official code for the ICASSP 2023 paper. A reinforcement learning agent chooses
where to look: it repositions the camera to improve the quality of the visual
signal reaching a frozen segmentation network, recovering performance that is
lost to occlusion and viewpoint noise.

> Gaurav Chaudhary, Laxmidhar Behera, Tushar Sandhan.
> *Active Perception System for Enhanced Visual Signal Recovery Using Deep
> Reinforcement Learning.* IEEE ICASSP, Rhodes, Greece, 2023, pp. 1–5.
> [IEEE Xplore](https://ieeexplore.ieee.org/document/10097084)

## Method

A pretrained Mask R-CNN (`torchvision`'s `maskrcnn_resnet50_fpn`) is held fixed.
The RL agent acts on the camera pose in a PyBullet scene, and its reward is the
segmentation quality the frozen network achieves from the resulting view. The
policy is therefore optimised directly against downstream perception, not
against a hand-designed image metric.

| File | Role |
|---|---|
| `E2E_object_segmentation_pybullet.py` | Main end-to-end training loop. |
| `E2E_RL_segmenation_with_random_input.py` | Variant with randomised initial viewpoints. |
| `object_segmentation_pybullet.py` | Segmentation environment on its own. |
| `baseline.py` | Fixed-viewpoint baseline for comparison. |
| `kuka_diverse_object_gym_env.py` | PyBullet scene: arm, camera and object set. |
| `Mask_RCNN_segmentation_pybullet_data.py` | Runs the frozen Mask R-CNN over rendered frames. |
| `utils.py` | Geometry and frame-transform helpers. |
| `out.csv` | Ground-truth segmentation reference read by the training loop. |

## Install

```bash
git clone https://github.com/gaurav-gaurav/ICASSP_RL-2023.git
cd ICASSP_RL-2023
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.8+. PyBullet renders headless by default; a GPU is strongly
recommended since Mask R-CNN runs on every environment step.

## Usage

```bash
# Train the active-perception agent
python E2E_object_segmentation_pybullet.py --train True --epoch 40000 \
    --save_dir log/active

# Randomised starting viewpoints
python E2E_RL_segmenation_with_random_input.py --train True

# Fixed-viewpoint baseline
python baseline.py

# Evaluate a trained policy
python E2E_object_segmentation_pybullet.py --train False \
    --eval log/active --eval_step 10
```

| Flag | Default | Meaning |
|---|---|---|
| `--epoch` | `40000` | Training epochs |
| `--train` | `True` | Train, or evaluate when `False` |
| `--save_dir` | `log/baseline` | Where checkpoints and logs are written |
| `--resume` | — | Checkpoint to resume from |
| `--eval` | `log/baseline` | Checkpoint directory to evaluate |
| `--eval_step` | `10` | Evaluation episodes |
| `--width` / `--height` | `1000` | Render resolution |
| `--env` | `1` | Scene variant |

`ur10.urdf` and `duck_vhacd2.obj` are loaded by relative path from the
repository root, so run the scripts from there.

## Scope and status

Research code released to support the paper, not a maintained library. Tidied
for release: training logs removed, one filename corrected (it contained a
space, which breaks imports and clones on some systems), and five unused
Mask R-CNN files deleted — they were a partial copy of
[matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN) that nothing
imported and that could not load anyway, since they expect an `mrcnn` package
this repository never contained. Segmentation comes from `torchvision`. No
algorithmic changes were made.

## Citation

```bibtex
@inproceedings{chaudhary2023active,
  title     = {Active Perception System for Enhanced Visual Signal Recovery Using Deep Reinforcement Learning},
  author    = {Chaudhary, Gaurav and Behera, Laxmidhar and Sandhan, Tushar},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {1--5},
  year      = {2023}
}
```

## License

MIT — see [LICENSE](LICENSE).
