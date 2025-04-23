# PCGNet-PP

**PCGNet++: Improved Point Cloud Generation Network for Occluded Objects using Monocular Images and Radar**

This repository contains the official implementation of **PCGNet++**, a multi-modal point cloud generation network designed for enhancing perception of occluded objects in autonomous driving scenarios. The framework fuses monocular image features and millimeter-wave radar to produce high-quality, semantically meaningful point clouds.

---

## 🔧 Project Structure

```
PCGNet-PP/
├── main.py # Entry point for training/evaluation
├── setup.py # Python package setup
├── task3_env.yml # Conda environment dependencies
├── src/ # Core model implementation
├── test_ops/ # Utility modules and testing operations
├── VoxelPooling.egg-info/ # Metadata from setup
└── README.md # Project description and instructions
```


---

## 📦 Environment Setup

We recommend using **conda** to manage the dependencies.

```bash
# Create and activate conda environment
conda env create -f task3_env.yml
```
Ensure CUDA and PyTorch are correctly installed for GPU acceleration.

---

## 📊 Datasets & Evaluation

We evaluate **PCGNet++** on our self-constructed dataset, **FlexRadar**, a **multi-modal dataset specifically designed for point cloud quality evaluation**. 


> 🔒 **Note:** The FlexRadar dataset will be made publicly available upon acceptance of our dataset paper.  
> 📂 **Repository:** For more details, please visit the https://github.com/Aiuan/ourDataset_v2_postprocess, https://github.com/Aiuan/sensors_calibration_v2.1.
