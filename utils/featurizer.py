import numpy as np

from rdkit import Chem
from rdkit.Chem import rdchem, rdBase, rdPartialCharges


rdBase.DisableLog('rdApp.error')


def rdchem_enum_to_list(values):
    r"""
    values = {
        0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER
    }
    """
    return [values[i] for i in range(len(values))]


def safe_index(alist, elem):
    """
    返回元素 elem 在列表 alist 中的索引。如果 elem 不存在，则返回最后一个索引。
    """
    try:
        return alist.index(elem)
    except ValueError:
        return max(len(alist) - 1, 0)


# atom features dict
atom_features = {
    'atomic_num': list(range(1, 119)) + ['misc'],
    'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'num_radical_electrons': [0, 1, 2, 3, 4, 5, 'misc'],
    'total_num_H_neighbors': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'implicit_valence': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
    'hybridization': rdchem_enum_to_list(rdchem.HybridizationType.values),
    'chirality': rdchem_enum_to_list(rdchem.ChiralType.values),
    'valence_out_shell': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'is_aromatic': [0, 1],
    'is_in_ring': [0, 1],
}


# bond features dict
bond_features = {
    'bond_dir': rdchem_enum_to_list(rdchem.BondDir.values),
    'bond_type': rdchem_enum_to_list(rdchem.BondType.values),
    'stereo': rdchem_enum_to_list(rdchem.BondStereo.values),
    'is_conjugated': [0, 1],
    'is_in_ring': [0, 1],
}


# float atom features dict
float_atom_features = {
    'vdw_radius': None,
    'partial_charge': None,
    'mass': None,
}


# float bond features dict
float_bond_features = {
    'bond_length': None,
    'bond_angle': None,
}


period_table = Chem.GetPeriodicTable()


def get_atom_value(atom, feature_name):
    if feature_name == 'atomic_num':
        return atom.GetAtomicNum()
    elif feature_name == 'degree':
        return atom.GetDegree()
    elif feature_name == 'formal_charge':
        return atom.GetFormalCharge()
    elif feature_name == 'num_radical_electrons':
        return atom.GetNumRadicalElectrons()
    elif feature_name == 'total_num_H_neighbors':
        return atom.GetTotalNumHs()
    elif feature_name == 'implicit_valence':
        return atom.GetImplicitValence()
    elif feature_name == 'hybridization':
        return atom.GetHybridization()
    elif feature_name == 'chirality':
        return atom.GetChiralTag()
    elif feature_name == 'is_aromatic':
        return int(atom.GetIsAromatic())
    elif feature_name == 'is_in_ring':
        return int(atom.IsInRing())
    elif feature_name == 'valence_out_shell':
        return period_table.GetNOuterElecs(atom.GetAtomicNum())
    else:
        raise ValueError(f'Invalid feature name: {feature_name}')
    

def get_atom_feature_id(atom, feature_name):
    assert feature_name in atom_features, f'Invalid feature name: {feature_name}'
    return safe_index(atom_features[feature_name], get_atom_value(atom, feature_name))


def get_atom_feature_dim(feature_name):
    assert feature_name in atom_features, f'Invalid feature name: {feature_name}'
    return len(atom_features[feature_name])


def get_bond_value(bond, feature_name):
    if feature_name == 'bond_dir':
        return bond.GetBondDir()
    elif feature_name == 'bond_type':
        return bond.GetBondType()
    elif feature_name =='stereo':
        return bond.GetStereo()
    elif feature_name == 'is_conjugated':
        return int(bond.GetIsConjugated())
    elif feature_name == 'is_in_ring':
        return int(bond.IsInRing())
    else:
        raise ValueError(f'Invalid feature name: {feature_name}')
    
    
def get_bond_feature_id(bond, feature_name):
    assert feature_name in bond_features, f'Invalid feature name: {feature_name}'
    return safe_index(bond_features[feature_name], get_bond_value(bond, feature_name))


def get_bond_feature_dim(feature_name):
    assert feature_name in bond_features, f'Invalid feature name: {feature_name}'
    return len(bond_features[feature_name])


def check_partial_charge(atom):
    r"""Validate and sanitize Gasteiger partial charge value
    
    Handles cases: missing property -> 0.0
                   NaN/invalid value -> 0.0 
                   infinity -> 10.0
    """
    if not atom.HasProp('_GasteigerCharge'):
        return 0.0
        
    pc = atom.GetDoubleProp('_GasteigerCharge')
    
    if pc != pc:  # NaN check
        return 0.0
    if pc == float('inf'):
        return 10.0  # 保持与原始逻辑一致，仅处理正无穷
    
    return pc


def get_atom_feature_vector(atom):
    atom_name = {
        'atomic_num': safe_index(atom_features['atomic_num'], atom.GetAtomicNum()),
        'degree': safe_index(atom_features['degree'], atom.GetDegree()),
        'formal_charge': safe_index(atom_features['formal_charge'], atom.GetFormalCharge()),
        'num_radical_electrons': safe_index(atom_features['num_radical_electrons'], atom.GetNumRadicalElectrons()),
        'total_num_H_neighbors': safe_index(atom_features['total_num_H_neighbors'], atom.GetTotalNumHs()),
        'implicit_valence': safe_index(atom_features['implicit_valence'], atom.GetImplicitValence()),
        'hybridization': safe_index(atom_features['hybridization'], atom.GetHybridization()),
        'chirality': safe_index(atom_features['chirality'], atom.GetChiralTag()),
        'is_aromatic': safe_index(atom_features['is_aromatic'], int(atom.GetIsAromatic())),
        'is_in_ring': safe_index(atom_features['is_in_ring'], int(atom.IsInRing())),
        'valence_out_shell': safe_index(atom_features['valence_out_shell'], period_table.GetNOuterElecs(atom.GetAtomicNum())),
        'vdw_radius': float(period_table.GetRvdw(atom.GetAtomicNum())),
        'partial_charge': float(check_partial_charge(atom)),
        'mass': float(atom.GetMass()),
    }
    

    return atom_name


def get_bond_feature_vector(bond):
    bond_name = {
        'bond_dir': safe_index(bond_features['bond_dir'], bond.GetBondDir()),
        'bond_type': safe_index(bond_features['bond_type'], bond.GetBondType()),
        'stereo': safe_index(bond_features['stereo'], bond.GetStereo()),
        'is_conjugated': safe_index(bond_features['is_conjugated'], int(bond.GetIsConjugated())),
        'is_in_ring': safe_index(bond_features['is_in_ring'], int(bond.IsInRing())),
    }
    
    return bond_name


def get_atom_name(mol):
    atom_features = []
    
    rdPartialCharges.ComputeGasteigerCharges(mol)
    
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_feature_vector(atom))
        
    return atom_features


def get_bond_name(mol):
    bond_features = []
    
    for bond in mol.GetBonds():
        bond_features.append(get_bond_feature_vector(bond))
        
    return bond_features


def get_bond_length(edges, pos):
    bond_lengths = []
    for src, dst in edges:
        bond_lengths.append(np.linalg.norm(pos[dst] - pos[src]))
    
    bond_lengths = np.array(bond_lengths, 'float32')
    return bond_lengths


def get_superedge_angles(edges, pos):
    
    def _get_vector(pos, edge):
        return pos[edge[1]] - pos[edge[0]]
    
    def _get_angle(v1, v2):
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0
        v1 = v1 / (n1 + 1e-5)
        v2 = v2 / (n2 + 1e-5)
        return np.arccos(np.dot(v1, v2))
    
    E = len(edges)
    edge_indices = np.arange(E)
    super_edges = []
    bond_angles = []
    bond_angles_dirs = []

    for dst_i in range(E):
        dst_edge = edges[dst_i]
        src_edge_indices = edge_indices[edges[:, 1] == dst_edge[0]]
        for src_i in src_edge_indices:
            if src_i == dst_i:
                continue
            src_edge = edges[src_i]
            src_vec = _get_vector(pos, src_edge)
            dst_vec = _get_vector(pos, dst_edge)
            super_edges.append([src_i, dst_i])
            angle = _get_angle(src_vec, dst_vec)
            bond_angles.append(angle)
            bond_angles_dirs.append(src_edge[1] == dst_edge[0])

    if len(super_edges) == 0:
        super_edges = np.zeros([0, 2], 'int64')
        bond_angles = np.zeros([0,], 'float32')
    else:
        super_edges = np.array(super_edges, 'int64')
        bond_angles = np.array(bond_angles, 'float32')
        
    return super_edges, bond_angles, bond_angles_dirs    


def mol_to_graph_data(mol):
    if mol is None:
        return None
    
    atom_id_names = [
        'atomic_num', 'degree', 'formal_charge', 'num_radical_electrons', 'total_num_H_neighbors',
        'implicit_valence', 'hybridization', 'chirality', 'is_aromatic', 'is_in_ring',
    ]
    
    bond_id_names = [
        'bond_dir', 'bond_type','stereo', 'is_conjugated', 'is_in_ring',
    ]
    
    data = {}
    for name in atom_id_names:
        data[name] = []
    data['mass'] = []
    for name in bond_id_names:
        data[name] = []
        
    data['edges'] = []
    
    for atom in mol.GetAtoms():
        for name in atom_id_names:
            data[name].append(get_atom_feature_id(atom, name) + 1)
        data['mass'].append(get_atom_value(atom, 'mass') * 0.01)
        
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = get_bond_feature_id(bond, name) + 1
            data[name] += [bond_feature_id] * 2
        
    # self-loop (+2)
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'] += [(i, i)]
    
    for name in bond_id_names:
        bond_feature_id = get_bond_feature_dim( name) + 2
        data[name] += [bond_feature_id] * N
        
    # check whether edge exists
    if len(data['edges']) == 0:  # mol has no bonds
        for name in bond_id_names:
            data[name] = np.zeros((0,), dtype="int64")
        data['edges'] = np.zeros((0, 2), dtype="int64")
        
    # make ndarray and check length
    for name in atom_id_names:
        data[name] = np.array(data[name], 'int64')
    data['mass'] = np.array(data['mass'], 'float32')
    
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    data['edges'] = np.array(data['edges'], 'int64')
    
    return data
    

def mol_to_geognn_data(mol):
    if mol is None:
        return None
    
    data = mol_to_graph_data(mol)
    coord = np.array(mol.GetConformer().GetPositions(), 'float32')
    
    data['pos'] = np.array(coord, 'float32')
    data['bond_length'] = get_bond_length(mol, coord)
    
    bond_angle_graph_edges, bond_angles, _ = get_superedge_angles(mol, coord)
    data['bond_angle_graph_edges'] = bond_angle_graph_edges
    data['bond_angle'] = np.array(bond_angles, 'float32')
    
    return data



