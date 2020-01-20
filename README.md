# SGCNN
This repository contains an implementation of the 'SGCNN' (Slab Graph Convolutional Neural Network) that predicts surface-related properties of crystal structures.

It provides 1) training a SGCNN model 2) an easy use to prediction of an adsorption energy using pre-trained model.

# Installation
This project required TensorFlow > 1.7.0

# Training
Training consists of two steps. First you need to prepare your dataset. After that, instill the dataset into SGCNN structure implemented by TensorFlow.

'SGCNN' takes input as bulk and surface crystal graphs. Currently, this software automatically converts POSCAR file to the graphs via running 'cgsurface.py' and 'cgbulk.py'.

To apply your dataset, you need to prepare two POSCAR files with same name in different directories, /surface and /bulk.

- Our model read [POSCAR](https://docs.rs/crate/vasp-poscar/0.2.0) files. If you have CIF files, you can run 'cif_to_POSCAR.py' file.

You need to write 'data.txt' file for the dataset. Our 'sgcnn.py' file will read 'data.txt' file, and input graphs and output values (binding energy) are extracted.

# Pretrained Model
To use pretrained model for predictions of adsorption energy, you can simply use [pretrained.py](https://github.com/myungjoon/SGCNN/SGCNN_pretrained.py)

# Citation
If you use 'SGCNN', please cite us using

```
  @article{Kim2020,
	author = {Kim, Myungjoon and Yeo, Byung Chul and Park, Youngtae and Lee, Hyuck Mo and Han, Sang Soo and Kim, Donghun},
	title = {Artificial Intelligence to Accelerate the Discovery of N2 Electroreduction Catalysts},
	journal = {Chemistry of Materials},
	volume = {},
	number = {},
	pages = {},
	year = {2020},
	doi = {10.1021/acs.chemmater.9b03686}
	}
```
