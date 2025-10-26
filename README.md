# DSSR-Net: Dimension scaling Super-Resolution Network

This repository contains the implementation of **DSSR-Net (Dimension scaling Super-Resolution Network)** for **high-resolution spectrum estimation
and signal reconstruction**.\
The project includes:

-   🏋️ **Training pipeline** for DSSR-Net\
-   🔬 **Simulation and validation experiments** (phase transition
    experiments, comparison with FFT)\
-   🧩 **Network modules**: complex-valued Fourier-inspired layers +
    convolutional denoising sub-networks

------------------------------------------------------------------------

## 📂 Project Structure

    .
    ├── train.py                   # Training script for DSSR-Net
    ├── test.py                # Simulation & validation experiments
    ├── net_model/
    │   ├── DSSR.py                # DSSR-Net network module
    │   └── cbam.py                # Attention module (CBAM/SAM)
    ├── data_gen/
    │   ├── data_generation.py     # Synthetic signal generation
    │   ├── fr.py                  # Guassian blurred fucntions   
    │   ├── noise.py               # Noise injection functions
    ├── complex_layers/
    │   └── complexLayers.py       # Complex-valued layers (Conv, Linear, etc.)
    └── checkpoint/                # Folder for saving models

------------------------------------------------------------------------
## Reference 
Please cite our work via
'''
@ARTICLE{Wang2025TAES,
  author={Wang, Ziwen and Wang, Jianping and Li, Pucheng and Ding, Zegang},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={Dimension Scaling SR-Net for Super-Resolution Radar Range Profiles}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TAES.2025.3614600}}
'''
