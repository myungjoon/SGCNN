import cg_surface as surface
import cg_bulk as bulk
import numpy as np
from encoding import *

""" This is where input graph structures are mathematically converted to 
    vector format."""

def data_preparation(X):
    data_num = len(X)

    #fix atom, bulk number
    atom_number_bulk = 4
    bond_number_bulk = 18

    # atom_number_bulk = 0
    # bond_number_bulk = 0
    #
    # for i in range(data_num):
    #     atom_vectors_bulk, bond_vectors_bulk, neighbor_bulk_index = Bulk.poscar_to_graph(X[i])
    #     n = atom_vectors_bulk.shape[0]
    #     m = 0
    #     if atom_number_bulk < n:
    #         atom_number_bulk = n
    #     for j in range(n):
    #
    #         m = bond_vectors_bulk[j].shape[0]
    #         if bond_number_bulk < m:
    #             bond_number_bulk = m
    print("bulk : (atom,  bond) : ", atom_number_bulk, bond_number_bulk)

    atoms_bulk = np.zeros([data_num, atom_number_bulk, CATEGORY_NUM], dtype=np.int32)
    bonds_bulk = np.zeros([data_num, atom_number_bulk, bond_number_bulk, BOND_CATEGORY_NUM], dtype=np.float32)

    bonds_bulk_index1 = np.full([data_num, atom_number_bulk, bond_number_bulk], -1, dtype=np.int32)
    bonds_bulk_index2 = np.full([data_num, atom_number_bulk, bond_number_bulk], -1, dtype=np.int32)

    atom_num_bulk = np.zeros([data_num], dtype=np.int32)

    for i in range(data_num):
        atom_vectors_bulk, bond_vectors_bulk, neighbor_bulk_index = bulk.poscar_to_graph(X[i])
        n = atom_vectors_bulk.shape[0]
        atoms_bulk[i][:n] = atom_vectors_bulk
        atom_num_bulk[i] = n

        for j in range(n):
            m = bond_vectors_bulk[j].shape[0]
            for k in range(m):
                bonds_bulk[i][j][k] = bond_vectors_bulk[j][k]

                bonds_bulk_index1[i][j][k] = neighbor_bulk_index[j][k][0] + i*atom_number_bulk
                bonds_bulk_index2[i][j][k] = neighbor_bulk_index[j][k][1] + i*atom_number_bulk

    atom_number_surface = 19
    bond_number_surface = 18

    # atom_number_surface = 0
    # bond_number_surface = 0
    #
    # for i in range(data_num):
    #     atom_vectors_surface, bond_vectors_surface, neighbor_surface_index = Surface.poscar_to_graph(X[i])
    #     n = atom_vectors_surface.shape[0]
    #     m = 0
    #     if atom_number_surface < n:
    #         atom_number_surface = n
    #     for j in range(n):
    #
    #         m = bond_vectors_surface[j].shape[0]
    #
    #         if bond_number_surface < m:
    #             bond_number_surface = m

    print("surface : (atom, bond) : ", atom_number_surface, bond_number_surface)

    atoms_surface = np.zeros([data_num, atom_number_surface, CATEGORY_NUM], dtype=np.int32)
    bonds_surface = np.zeros([data_num, atom_number_surface, bond_number_surface, BOND_CATEGORY_NUM], dtype=np.float32)
    bonds_surface_index1 = np.full([data_num, atom_number_surface, bond_number_surface], -1, dtype=np.int32)
    bonds_surface_index2 = np.full([data_num, atom_number_surface, bond_number_surface], -1, dtype=np.int32)
    atom_num_surface = np.zeros(data_num, dtype=np.int32)

    for i in range(data_num):
        atom_vectors_surface, bond_vectors_surface, neighbor_surface_index = surface.poscar_to_graph(X[i])
        n = atom_vectors_surface.shape[0]
        atoms_surface[i][:n] = atom_vectors_surface
        atom_num_surface[i] = n
        for j in range(n):
            m = bond_vectors_surface[j].shape[0]
            for k in range(m):
                #bonds_surface[i][j][k] = np.concatenate([atom_vectors_surface[j], bond_vectors_surface[j][k]])  ## 수정
                try:
                    bonds_surface[i][j][k] = bond_vectors_surface[j][k]
                    bonds_surface_index1[i][j][k] = neighbor_surface_index[j][k][0] + i * atom_number_surface
                    bonds_surface_index2[i][j][k] = neighbor_surface_index[j][k][1] + i * atom_number_surface
                except:
                    print(X[i])

    return atom_number_bulk,  atoms_bulk, bonds_bulk, bonds_bulk_index1, bonds_bulk_index2, atom_num_bulk, bond_number_bulk, \
           atom_number_surface, atoms_surface, bonds_surface, bonds_surface_index1, bonds_surface_index2, atom_num_surface, bond_number_surface