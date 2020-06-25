# SGCNN

This repository contains an implementation of the `SGCNN` (Slab Graph Convolutional Neural Network) that predicts surface-related properties of crystal structures.

It provides 1) training a `SGCNN` model 2) an easy use to prediction of an adsorption energy using pretrained model.

<div align="center">
<img src="https://github.com/myungjoon/SGCNN/blob/master/achitecture.png"><br>
</div>



# Installation

This project required TensorFlow > 1.7.0

Clone this repository:

```
git clone https://github.com/myungjoon/SGCNN.git
```

# Usage

Training consists of two steps. First you need to prepare your dataset. After that, instill the dataset into `SGCNN` structure implemented by TensorFlow.

`SGCNN` takes input as bulk and surface crystal graphs. Currently, this software automatically converts `POSCAR` format file to the crystal graphs via running [cgsurface.py](https://github.com/myungjoon/SGCNN/blob/master/cg_surface.py) and [cgbulk.py](https://github.com/myungjoon/SGCNN/blob/master/cg_bulk.py).

To apply your dataset, you need to prepare two POSCAR files with same name in different directories, `surface/` and `bulk/`.

- Our model read [POSCAR](https://docs.rs/crate/vasp-poscar/0.2.0) files. If you have CIF files, you should convert it to POSCAR file.



## Features

You can add your own features to [feature.csv](https://github.com/myungjoon/SGCNN/blob/master/feature.csv) file.



## Data Format

You need to write data.txt file for the dataset. Our [sgcnn.py](https://github.com/myungjoon/SGCNN/blob/master/sgcnn.py) file will read 'data.txt' file, and input graphs and output values (binding energies) are extracted.

'data.txt' file has following format.

```
#adsorbate  crystal  face  site  atom1  atom2  adsorption energy
1 1 111 11 Au Ti -0.87
4 1 111 21 Au Ti -3.931
```

- Each row describes a structure, composition, and surface-related property (adsorption energy)

- Each column represents specific characteristic.

- Corresponding POSCAR files should be stored in `bulk/` and `surface/` directories.



## Models

The trained model will be saved as 'models/best.ckpt'

You can use your own model by running [test.py](https://github.com/myungjoon/SGCNN/blob/master/test.py)



# Pretrained Model

To use pretrained model for predictions of adsorption energy, you can simply use [pretrained.py](https://github.com/myungjoon/SGCNN/blob/master/pretrained.py). This python file reads 'test.txt' and writes results on 'result.txt'



# License

This repository is released under the [MIT license](https://github.com/myungjoon/SGCNN/blob/master/LICENSE).



# Citation

If you use `SGCNN`, please cite us using

```
  @article{Kim2020,
	author = {Kim, Myungjoon and Yeo, Byung Chul and Park, Youngtae and Lee, Hyuck Mo and Han, Sang Soo and Kim, Donghun},
	title = {Artificial Intelligence to Accelerate the Discovery of N2 Electroreduction Catalysts},
	journal = {Chemistry of Materials},
	volume = {32},
	number = {2},
	pages = {639-912},
	year = {2020},
	doi = {10.1021/acs.chemmater.9b03686}
	}
```
