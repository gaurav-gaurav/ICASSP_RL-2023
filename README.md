# Active Perception System for Object Segmentation (ICASSP 2023)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)

Official repository for the IEEE ICASSP 2023 paper:  
**"Active Perception System for Object Segmentation Using Reinforcement Learning"**  
[[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10097084) | [Project Page (if available)]()  

![Demo GIF](docs/demo.gif)  
*Example: RL agent optimizing viewpoints for object segmentation in PyBullet.*

---

## 🚀 Key Features
- **Reinforcement Learning for Active Perception**: PPO/DQN agent learns optimal viewpoint selection strategies.
- **PyBullet Simulation**: Realistic 3D environment with dynamic objects and sensor noise.
- **Joint Training**: End-to-end pipeline combining Mask R-CNN segmentation with RL policies.
- **Metrics**: Segmentation accuracy (IoU, mAP) and perception efficiency (steps/reward).

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/gaurav-gaurav/ICASSP_RL-2023.git
   cd ICASSP_RL-2023
