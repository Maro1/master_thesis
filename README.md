# Master Thesis Code

This repository contains the code for the master thesis of spring 2023.

## Code structure

The `GET3D` folder contains a modified version of the [official GET3D release](https://github.com/nv-tlabs/GET3D). This version allows for the use of custom datasets not used by the official release.

The `PointNet` folder contains a [PyTorch PointNet implementation](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) where the `test_classification_custom.py` script has been added for evaluating the percentages obtained on each ModelNet40 category.

The `point_cloud` folder contains code for converting .obj files to point cloud .ply files and code for visualizing the point clouds.

The `pre_processing` folder contains the pre-processing code for each of the datasets.

## Setup

### GET3D

GET3D was run using Python 3.8.2 and CUDA 11.3.1. The dependencies can be installed using the command:

```bash
pip install -r requirements.txt
```

The model can then be trained following the GET3D [README](https://github.com/nv-tlabs/GET3D/blob/master/README.md).

### PointNet

PointNet was run using Python 3.8.6 and CUDA 11.7.0, and PyTorch 1.13.1. The model can be trained using the following command from the `PointNet` folder:

```bash
python train_classification.py --model pointnet2_cls_msg
```

The evaluation can then be run using the following command from the PointNet folder:

```bash
  python test_classification_custom.py --log_dir <log_dir> --num_point <num_point>
```

With the point cloud .ply files located in `PointNet/data/custom/<category_name>/`.

### Point Cloud Scripts

The point cloud scripts can be run by first installing the dependencies using:

```bash
pip install -r requirements.txt
```

The .obj to .ply converter can then be run from the `point_cloud` folder using:

```bash
python obj_to_pointcloud.py <path_to_objs> <output_folder> --num_points <num_points>
```

The visualization script can be run from the `point_cloud` folder using:
```bash
python ply_vis.py <file_path>
```

### Pre Processing

#### CO3D

The CO3D pre-processing script can be run from the `pre_processing/co3d` folder using the command:

```bash
python pre_process.py --category <category> <path_to_co3d> <path_to_colmap_output> <output_folder>
```

Where `path_to_co3d` is the path to the downloaded CO3D dataset and `path_to_colmap_output` is the path to the output folder after re-running COLMAP. The COLMAP output folder should look like this:

```
<colmap_folder>
  <category_1>
  <category_2>
  ...
```

#### Objectron

The Objectron pre-processing was run using Python 3.8.2 and CUDA 11.7.0. It can be run from the `pre_processing/objectron` folder using the following command:

```bash
python pre_processing/objectron/pre_process.py <output_folder> --category <category>
```


