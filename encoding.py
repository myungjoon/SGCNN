import numpy as np

atom_features = {}
with open('feature.csv','r') as f:  ## read CSV file and make dictionary
    index = f.readline().strip().split(sep=',')
    k=0
    for line in f.readlines():
        values = line.strip().split(sep=',')
        dict_int = {index[j] : int(values[j]) for j in range(1,4)}

        dict_float = {}

        for j in range(4,len(values)):
            if values[j] == '': ## empty value -> "empty"
                dict_float[index[j]] =  "empty"
            else:
                dict_float[index[j]] = float(values[j])

        dict_int.update(dict_float)

        atom_features[values[0]] = dict_int  ## values[0] == element


feature_list = {'group' : 18, 'period' : 9, 'electronegativity' : 10, 'ionization' : 10, 'affinity':10, 'volume' :  10, 'radius' : 10, 'atomic number' : 32, 'weight' : 10, 'melting' : 10, 'boiling' : 10,\
                'density' : 10, 'Zeff' : 10, 'polarizability' : 10, 'resistivity' : 10, 'fusion' : 10, 'vaporization' : 10, 'atomization' : 10, 'capacity' : 10, 'valence' : 11, 'd-electron':11}

features = ['group', 'electronegativity', 'volume', 'affinity', 'weight']

CATEGORY_NUM = 0 ## Total number of category
for i in range(len(features)):
    CATEGORY_NUM += feature_list[features[i]]

#BOND_CATEGORY_NUM = 20
BOND_CATEGORY_NUM = 1

NEIGHBOR_CATEGORY_NUM = CATEGORY_NUM + BOND_CATEGORY_NUM  ## The total catergoty number of neighbor atom
TOTAL_CATEGORY_NUM = CATEGORY_NUM + NEIGHBOR_CATEGORY_NUM  ## atom + neighbor atom

def atom_encoding(a):
    c = 0
    a_one_hot = np.zeros(CATEGORY_NUM, dtype=np.int32)  ## make [0 0 0 .. ] vector, length = CATEGORY_NUM
    #print(a_one_hot)
    for i in range(len(features)):
        index = 0  ## The category which elements belong to
        ## atom_features = The dictonary of feature values
        if features[i] =='group':
            index = atom_features[a]['group'] - 1 ## Ex. atom_features[a]['group'] = 1 -> index = 0
        elif features[i] == 'period':
            index = atom_features[a]['period'] - 1 ## Ex. atom_features[a]['period'] = 1 -> index = 0
        elif features[i] == 'radius': ## int -> round-up
            index = int((atom_features[a]['radius'] - 0.25) / 0.225)
            ## ((atom_features[a]['electronegativity'] - X) / Y) -> X : the min value of category, Y : the gap value of category
        elif features[i] == 'electronegativity':
            index = int((atom_features[a]['electronegativity'] - 0.5) / 0.35)
        elif features[i] == 'ionization':
            index = int((np.log(atom_features[a]['ionization']) - 1.3) / 0.2)
        elif features[i] == 'affinity':
            index = int((atom_features[a]['affinity'] + 2.4) / 0.6)
        elif features[i] == 'volume':
            index = int((np.log(atom_features[a]['volume']) - 1.5) / 0.28)
        elif features[i] == 'weight':
            index = int((atom_features[a]['weight'])/27)
        elif features[i] == 'melting': ## Different gap size
            index = int((atom_features[a]['melting']+300)/500)
        elif features[i] == 'boiling':
            index = int((atom_features[a]['boiling']+300)/600)
        elif features[i] == 'density':
            index = int((atom_features[a]['density'])/2.5)
        elif features[i] == 'Zeff':
            index = int((atom_features[a]['Zeff']-1)/0.45)
        elif features[i] == 'atomic number':
            index = atom_features[a]['atomic number'] - 1
        elif features[i] == 'd-electron':
            index = int(atom_features[a]['d-electron'])
        elif features[i] == 'polarizability':
            if atom_features[a]['polarizability'] == atom_features['H']['polarizability']:
                index = 0
            elif atom_features[a]['polarizability'] == atom_features['N']['polarizability']:
                index = 1
            else:
                index = int((atom_features[a]['polarizability']-4)/(19/8))
        elif features[i] == 'resistivity':
            if atom_features[a]['resistivity'] == atom_features['H']['resistivity']:
                index = 0
            elif atom_features[a]['resistivity'] == atom_features['N']['resistivity']:
                index = 1
            else:
                index = int(atom_features[a]['resistivity']/20)

        elif features[i] == 'fusion':
            if atom_features[a]['fusion'] == atom_features['H']['fusion']:
                index = 0
            elif atom_features[a]['fusion'] == atom_features['N']['fusion']:
                index = 1
            else:
                index = int((atom_features[a]['fusion']-6)/(30/8))

        elif features[i] == 'vaporization':
            if atom_features[a]['vaporization'] == atom_features['H']['vaporization']:
                index = 0
            elif atom_features[a]['vaporization'] == atom_features['N']['vaporization']:
                index = 1
            else:
                index = int((atom_features[a]['vaporization']-100)/(750/8))

        elif features[i] == 'heat capacity':
            if atom_features[a]['heat capacity'] == atom_features['H']['heat capacity']:
                index = 0
            elif atom_features[a]['heat capacity'] == atom_features['N']['heat capacity']:
                index = 1
            else:
                index = int((atom_features[a]['heat capacity']-0.12)/(0.45/8))

        elif features[i] == 'valence':
            index = int(atom_features[a]['valence'])
        index = index + c
        a_one_hot[index] = 1
        c = c + feature_list[features[i]]
    return a_one_hot

def bond_encoding(d):
    d_one_hot = np.zeros(BOND_CATEGORY_NUM, dtype=np.float32)
    for i in range(BOND_CATEGORY_NUM):
        d_one_hot[i] = 1
    return d_one_hot

    