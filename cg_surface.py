import numpy as np
from encoding import *

"""This is where surface graph is constructed. """

Surface_min_distance = 100

def unit_cell_expansion_slab(lattices, matrix):
    num = len(lattices)
    expanded_lattices = np.zeros([3,num*9],dtype=np.float32)
    coeff = [0, -1, 1]
    count = 0

    v1 = np.array([1,0,0])
    v2 = np.array([0,1,0])

    for i in coeff:
        for j in coeff:
            new_lattices = np.copy(lattices)
            translation = v1*i + v2*j
            new_lattices = np.transpose(np.matmul(new_lattices + translation, matrix))
            expanded_lattices[:,count*num:(count+1)*num] = new_lattices
            count = count + 1
    return expanded_lattices

def find_neighbor(lattices, expanded_lattices, elements):
    lattice_num = len(lattices[0])
    expanded_lattice_num = len(expanded_lattices[0])
    connectivity = []
    distances = []
    tolerance = 1.5
    for i in range(lattice_num):
        neighbor_num = 0
        for j in range(expanded_lattice_num):
            if i==j:
                continue

            cond1 = np.linalg.norm(lattices[:,i] - expanded_lattices[:,j]) < 6
            cond2 = np.linalg.norm(lattices[:,i] - expanded_lattices[:,j]) < (atom_features[elements[i]]['radius'] + atom_features[elements[j%len(elements)]]['radius'] + tolerance)
            if cond1 and cond2:
                connectivity.append([i,j])
                distance = np.linalg.norm(lattices[:,i] - expanded_lattices[:,j])

                distances.append(distance)
                #print([np.linalg.norm(lattices[:,i] - expanded_lattices[:,j]), elements[i], elements[j%len(elements)]])
                neighbor_num = neighbor_num + 1

    return connectivity, distances

def bond_construction(elements,connectivity,distances, features, CATEGORY_NUM, TOTAL_CATEGORY_NUM, NEIGHBOR_CATEGORY_NUM):
    BOND_CATEGORY_NUM = TOTAL_CATEGORY_NUM - 2 * CATEGORY_NUM

    bond_vectors = []
    neighbor_indices = []
    atom_num = len(elements)
    neighbor_num = np.zeros(atom_num, dtype=np.int32)
    for connection in connectivity:
        neighbor_num[connection[0]] += 1
    count = 0
    for i in range(len(elements)):
        bond_vector = np.zeros([neighbor_num[i], BOND_CATEGORY_NUM], dtype=np.float32)
        neighbor_index = np.zeros([neighbor_num[i], 2], dtype=np.float32)
        for j in range(neighbor_num[i]):
            # neighbor_atom = elements[connectivity[count][1]%atom_num]
            # neighbor_atom = connectivity[count][1] % atom_num
            neighbor_index[j][0] = connectivity[count][0] % atom_num
            neighbor_index[j][1] = connectivity[count][1] % atom_num
            bond_vector[j] = bond_encoding(distances[count])
            count += 1
        bond_vectors.append(bond_vector)
        neighbor_indices.append(neighbor_index)

    return bond_vectors, neighbor_indices

def poscar_to_graph(name):
    """This function converts POSCAR file to graph structure."""
    
    ads_type, sys = name.split('-')[0:2]
    ads_type = int(ads_type)
    sys = int(sys)

    path = './surface/'

    with open(path+name, 'r') as poscarfile:
        elements = []
    
        line = poscarfile.readline()
        line = poscarfile.readline()
    
        a = list(map(float,poscarfile.readline().split()))
        b = list(map(float,poscarfile.readline().split()))
        c = list(map(float,poscarfile.readline().split()))
    
        trans_matrix = np.zeros([3,3], dtype=np.float32)
        trans_matrix[0] = a
        trans_matrix[1] = b
        trans_matrix[2] = c
    
        elements_type = poscarfile.readline().split()
    
        elements_num = poscarfile.readline().split()
        elements_num = list(map(int,elements_num))
        
        atom_num = []
        if sys == 1 or sys ==2 or sys==3:
            atom_num.append(int(elements_num[0]/3))
            atom_num.append(int(elements_num[1]))
            for i in range(2,len(elements_type)):
                atom_num.append(int(elements_num[i]))
        elif sys == 4 or sys==5:
            atom_num.append(int(elements_num[0]/2))
            atom_num.append(int(elements_num[1]/2))
            for i in range(2,len(elements_type)):
                atom_num.append(int(elements_num[i]))
        elif sys == 7:
            atom_num.append(int(elements_num[0] / 3))
            atom_num.append(int(elements_num[1] / 3))
            for i in range(2, len(elements_type)):
                atom_num.append(int(elements_num[i]))
        elif sys == 8:
            atom_num.append(int(elements_num[0] / 3))
            atom_num.append(int(elements_num[1] / 3))
            for i in range(2, len(elements_type)):
                atom_num.append(int(elements_num[i]))

        for i in range(len(elements_type)):
            for j in range(atom_num[i]):
                elements.append(elements_type[i])
                
        total_atom_num = sum(atom_num)
    
        poscarfile.readline()
        poscarfile.readline()
    
        atoms = np.zeros([total_atom_num, 3],dtype=np.float32)
    
        current_atom_num = 0
        while current_atom_num < total_atom_num:
            line = poscarfile.readline()
            if line.split()[4] == 'F':
                continue
            elif line.split()[4] =='T':
                atoms[current_atom_num] = list(map(float,line.split()[0:3]))
                current_atom_num += 1

    lattices = np.transpose(np.matmul(atoms, trans_matrix))
    expanded_lattices = unit_cell_expansion_slab(atoms, trans_matrix)

    connectivity, distances = find_neighbor(lattices, expanded_lattices, elements)

    atom_vectors = np.zeros([total_atom_num, CATEGORY_NUM], dtype=np.float32)
    bond_num = np.zeros([total_atom_num],dtype=np.int32)
    for i in range(total_atom_num):
        atom_vectors[i] = atom_encoding(elements[i])

    bond_vectors, neighbor_indices = bond_construction(elements,connectivity,distances, features, CATEGORY_NUM, TOTAL_CATEGORY_NUM, NEIGHBOR_CATEGORY_NUM)

    for i in range(len(bond_vectors)):
        bond_num[i] = bond_vectors[i].shape[0]

    return atom_vectors, bond_vectors, neighbor_indices