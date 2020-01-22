import numpy as np
from encoding import *

"""This is where bulk graph is constructed. """

def unit_cell_expansion_bulk(lattices, matrix):
    num = len(lattices)
    expanded_lattices = np.zeros([3,num*27],dtype=np.float32)
    coeff = [0, -1, 1]
    count = 0

    v1 = np.array([1,0,0])
    v2 = np.array([0,1,0])
    v3 = np.array([0,0,1])

    for i in coeff:
        for j in coeff:
            for k in coeff:
                new_lattices = np.copy(lattices)
                translation = v1 * i + v2 * j + v3 * k
                new_lattices = np.transpose(np.matmul(new_lattices + translation, matrix))
                expanded_lattices[:, count * num:(count + 1) * num] = new_lattices
                count = count + 1
    return expanded_lattices

def find_neighbor(lattices, expanded_lattices, elements):
    lattice_num = len(lattices[0])
    num = len(expanded_lattices[0])
    connectivity = []
    distances = []
    tolerance = 1.5
    for i in range(lattice_num):
        neighbor_num = 0
        for j in range(num):
            if i==j:
                continue
            cond1 = np.linalg.norm(lattices[:,i] - expanded_lattices[:,j]) < 6
            cond2 = np.linalg.norm(lattices[:,i] - expanded_lattices[:,j]) < (atom_features[elements[i]]['radius'] + atom_features[elements[j%len(elements)]]['radius'] + tolerance)
            if cond1 and cond2:
                connectivity.append([i,j])
                distance = np.linalg.norm(lattices[:,i] - expanded_lattices[:,j])
                distances.append(distance)
                neighbor_num = neighbor_num + 1
    return connectivity, distances

def bond_construction(elements,connectivity,distance):
    BOND_CATEGORY_NUM = TOTAL_CATEGORY_NUM - 2*CATEGORY_NUM

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
            # neighbor_atom = connectivity[count][1] % atom_num\
            neighbor_index[j][0] = connectivity[count][0] % atom_num
            neighbor_index[j][1] = connectivity[count][1] % atom_num
            bond_vector[j] = bond_encoding(distance[count])
            count += 1
        bond_vectors.append(bond_vector)
        neighbor_indices.append(neighbor_index)
    return bond_vectors, neighbor_indices

def poscar_to_graph(name):

    s = name.split('-')

    path = './bulk/'

    if s[1] == '1' or s[1] == '2' or s[1] == '3':
        name = s[1] + '-' + s[4] + '-' + s[4]
    elif s[1] =='7' and s[4]==s[5]:
        name = '1-' + s[4] +'-' +  s[5]
    else:
        name = s[4] + '-' + s[5]


    poscarfile = open(path + name, 'r')
    elements = []

    line = poscarfile.readline()
    line = poscarfile.readline()

    a = list(map(float, poscarfile.readline().split()))
    b = list(map(float, poscarfile.readline().split()))
    c = list(map(float, poscarfile.readline().split()))

    trans_matrix = np.zeros([3, 3], dtype=np.float32)
    trans_matrix[0] = a
    trans_matrix[1] = b
    trans_matrix[2] = c

    elements_ = poscarfile.readline().split()
    atoms = list(map(int, poscarfile.readline().split()))
    for i in range(len(elements_)):
        for j in range(atoms[i]):
            elements.append(elements_[i])

    atom_num = len(elements)

    # line = poscarfile.readline()
    mode = poscarfile.readline()[0]
    # print(mode)

    atoms = np.zeros([atom_num, 3], dtype=np.float32)

    for i in range(atom_num):
        temp = list(map(float, poscarfile.readline().split()[0:3]))
        atoms[i] = temp

    lattices = np.transpose(np.matmul(atoms, trans_matrix))

    expanded_lattices = unit_cell_expansion_bulk(atoms, trans_matrix)
    connectivity, distances = find_neighbor(lattices, expanded_lattices, elements)

    atom_vectors = np.zeros([atom_num, CATEGORY_NUM], dtype=np.float32)
    bond_num = np.zeros([atom_num], dtype=np.int32)
    for i in range(atom_num):
        atom_vectors[i] = atom_encoding(elements[i])

    bond_vectors, neighbor_indices = bond_construction(elements, connectivity, distances)

    for i in range(len(bond_vectors)):
        bond_num[i] = bond_vectors[i].shape[0]
    poscarfile.close()

    return atom_vectors, bond_vectors, neighbor_indices