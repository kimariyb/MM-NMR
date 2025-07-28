import math
import numpy as np
import torch
import algos

from rdkit import Chem
from rdkit.Chem import rdchem, AllChem


def is_valid_smiles(smiles: str) -> bool:
    """
    Check if the SMILES string is valid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error parsing SMILES: {e}")
        return False


def rdchem_enum_to_list(values) -> list:
    """
    Convert RDKit enum values to a list.

    values = {
        0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER
    }
    """
    return [values[i] for i in range(len(values))]


def safe_index(alist: list, value: str) -> int:
    """
    Get the index of a value in a list, returning the last index if not found.
    """
    return alist.index(value) if value in alist else len(alist) - 1


class CompoundKit:
    atom_feats_dict = {
        "atomic_num": list(range(1, 119)) + ['misc'],
        "chiral_tag": rdchem_enum_to_list(rdchem.ChiralType.values),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "hybridization": rdchem_enum_to_list(rdchem.HybridizationType.values),
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "is_aromatic": [0, 1],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        "num_radical_e": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'atom_is_in_ring': [0, 1],
    }

    atom_float_feats_names = [
        "mass",
        "van_der_waals_radius",
        "partial_charge"
    ]
    bond_float_feats_names = [
        "bond_length", 
        "bond_angle"
    ]    


    period_table = Chem.GetPeriodicTable()

    @staticmethod
    def get_atom_feature(atom, feature_name):
        """
        Get a specific feature of an atom.

        Args:
            atom: RDKit atom object.
            feature_name: Name of the feature to retrieve.

        Returns:
            The value of the requested feature.

        Raises:
            ValueError: If the feature name is invalid.
        """
        feature_map = {
            'atomic_num': atom.GetAtomicNum(),
            'chiral_tag': atom.GetChiralTag(),
            'degree': atom.GetDegree(),
            'explicit_valence': atom.GetExplicitValence(),
            'formal_charge': atom.GetFormalCharge(),
            'hybridization': atom.GetHybridization(),
            'implicit_valence': atom.GetImplicitValence(),
            'is_aromatic': int(atom.GetIsAromatic()),
            'mass': int(atom.GetMass()),
            'total_numHs': atom.GetTotalNumHs(),
            'num_radical_e': atom.GetNumRadicalElectrons(),
            'atom_is_in_ring': int(atom.IsInRing())
        }

        if feature_name not in feature_map:
            raise ValueError(f"Invalid feature name: {feature_name}")

        return feature_map[feature_name]
        
    @staticmethod
    def get_atom_feature_index(atom, feature_name):
        """
        Get the index of a specific atom feature in the predefined feature list.

        Args:
            atom: RDKit atom object.
            feature_name: Name of the feature to retrieve.

        Returns:
            The index of the feature value in the predefined list.

        Raises:
            AssertionError: If the feature name is not found in the feature dictionary.
        """
        assert feature_name in CompoundKit.atom_feats_dict, f"{feature_name} not found in atom_feats_dict"
        return safe_index(CompoundKit.atom_feats_dict[feature_name], CompoundKit.get_atom_feature(atom, feature_name))

    @staticmethod
    def get_atom_feature_size(feature_name):
        """
        Get the size of the predefined list for a specific atom feature.

        Args:
            feature_name: Name of the feature to retrieve.

        Returns:
            The size of the predefined feature list.

        Raises:
            AssertionError: If the feature name is not found in the feature dictionary.
        """
        assert feature_name in CompoundKit.atom_feats_dict, f"{feature_name} not found in atom_feats_dict"
        return len(CompoundKit.atom_feats_dict[feature_name])


    @staticmethod
    def get_bond_feature(bond, feature_name):
        """
        Get a specific feature of a bond.

        Args:
            bond: RDKit bond object.
            feature_name: Name of the feature to retrieve.

        Returns:
            The value of the requested feature.

        Raises:
            ValueError: If the feature name is invalid.
        """
        feature_map = {
            'bond_dir': bond.GetBondDir(),
            'bond_type': bond.GetBondType(),
            'is_in_ring': int(bond.IsInRing()),
            'is_conjugated': int(bond.GetIsConjugated()),
            'bond_stereo': bond.GetStereo()
        }

        if feature_name not in feature_map:
            raise ValueError(f"Invalid feature name: {feature_name}")

        return feature_map[feature_name]
        
    @staticmethod
    def get_bond_feature_index(bond, feature_name):
        """
        Get the index of a specific bond feature in the predefined feature list.

        Args:
            bond: RDKit bond object.
            feature_name: Name of the feature to retrieve.

        Returns:
            The index of the feature value in the predefined list.

        Raises:
            AssertionError: If the feature name is not found in the feature dictionary.
        """
        assert feature_name in CompoundKit.get_bond_feature, f"{feature_name} not found in get_bond_feature"
        return safe_index(CompoundKit.get_bond_feature[feature_name], CompoundKit.get_bond_feature(bond, feature_name))

    @staticmethod
    def get_ring_size(mol):
        """
        Get the ring size information for each atom in the molecule.

        Args:
            mol: RDKit molecule object.

        Returns:
            A list of lists, where each inner list contains the number of rings
            of size 3 to 8 that each atom is part of.
        """
        rings = mol.GetRingInfo()
        rings_info = list(rings.AtomRings())
        ring_list = []

        for atom in mol.GetAtoms():
            atom_result = []
            for ringsize in range(3, 9):
                num_of_ring_at_ringsize = sum(
                    1 for r in rings_info if len(r) == ringsize and atom.GetIdx() in r
                )
                atom_result.append(min(num_of_ring_at_ringsize, 9))

            ring_list.append(atom_result)

        return ring_list
    
    @staticmethod
    def atom_to_feature_vector(atom):
        """
        Convert an atom to a feature vector.

        Args:
            atom: RDKit atom object.

        Returns:
            A dictionary containing the atom's features.
        """
        feature_vector = {
            "atomic_num": safe_index(CompoundKit.atom_feats_dict["atomic_num"], atom.GetAtomicNum()),
            "chiral_tag": safe_index(CompoundKit.atom_feats_dict["chiral_tag"], atom.GetChiralTag()),
            "degree": safe_index(CompoundKit.atom_feats_dict["degree"], atom.GetTotalDegree()),
            "explicit_valence": safe_index(CompoundKit.atom_feats_dict["explicit_valence"], atom.GetExplicitValence()),
            "formal_charge": safe_index(CompoundKit.atom_feats_dict["formal_charge"], atom.GetFormalCharge()),
            "hybridization": safe_index(CompoundKit.atom_feats_dict["hybridization"], atom.GetHybridization()),
            "implicit_valence": safe_index(CompoundKit.atom_vocab_dict["implicit_valence"], atom.GetImplicitValence()),
            "is_aromatic": safe_index(CompoundKit.atom_feats_dict["is_aromatic"], int(atom.GetIsAromatic())),
            "total_numHs": safe_index(CompoundKit.atom_feats_dict["total_numHs"], atom.GetTotalNumHs()),
            'num_radical_e': safe_index(CompoundKit.atom_vocab_dict['num_radical_e'], atom.GetNumRadicalElectrons()),
            'atom_is_in_ring': safe_index(CompoundKit.atom_feats_dict['atom_is_in_ring'], int(atom.IsInRing())),
            'van_der_waals_radius': CompoundKit.period_table.GetRvdw(atom.GetAtomicNum()),
            'partial_charge': CompoundKit.check_partial_charge(atom),
            'mass': atom.GetMass(),
        }
        return feature_vector
    
    @staticmethod
    def get_atom_feature_vectors(mol):
        """
        Get a list of feature vectors for all atoms in the molecule.

        Args:
            mol: RDKit molecule object.

        Returns:
            A list of dictionaries, each containing the features of an atom.
        """
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        return [CompoundKit.atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    

    @staticmethod
    def check_partial_charge(atom):
        """
        Check and sanitize the partial charge of an atom.

        Args:
            atom: RDKit atom object.

        Returns:
            The sanitized partial charge.
        """
        pc = atom.GetDoubleProp('_GasteigerCharge')
        if pc != pc:  # Check for NaN
            pc = 0
        elif pc == float('inf'):
            pc = 10
        return pc
    

class Compound3DKit:
    @staticmethod
    def get_atom_positions(mol, conf):
        """
        Get the 3D positions of all atoms in the molecule.

        Args:
            mol: RDKit molecule object.
            conf: RDKit conformer object.

        Returns:
            A list of 3D coordinates for each atom.
        """
        if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())

        return [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                for i in range(len(mol.GetAtoms()))]
    
    @staticmethod
    def get_MMFF_atom_positions(mol, num_confs=None, return_energy=False):
        """
        Generate MMFF-optimized 3D positions for atoms in the molecule.

        Args:
            mol: RDKit molecule object.
            num_confs: Number of conformations to generate (default: None).
            return_energy: Whether to return the energy of the optimized conformation.

        Returns:
            A tuple containing the molecule, atom positions, and optionally the energy.
        """
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=num_confs)
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            index_ = np.argmin([x[1] for x in res])
            energy = res[index_][1]
            conf = new_mol.GetConformer(id=int(index_))
        except Exception:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()

        atom_positions = Compound3DKit.get_atom_positions(new_mol, conf)
        if return_energy:
            return new_mol, atom_positions, energy
        else:
            return new_mol, atom_positions
        
    @staticmethod
    def get_MMFF_atom_positions_nonminimum(mol, num_confs=None, return_energy=False, percent=75):
        """
        Generate MMFF-optimized 3D positions for atoms in the molecule, avoiding the minimum energy conformation.

        Args:
            mol: RDKit molecule object.
            num_confs: Number of conformations to generate (default: None).
            return_energy: Whether to return the energy of the optimized conformation.
            percent: Percentile of energy to select (default: 75).

        Returns:
            A tuple containing the molecule, atom positions, and optionally the energy.
        """
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=num_confs, randomSeed=42)
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            energies = [x[1] for x in res]
            energy_threshold = np.percentile(energies, percent)
            closest_index = np.argmin(np.abs(np.array(energies) - energy_threshold))
            conf = new_mol.GetConformer(id=int(closest_index))
            energy = energies[closest_index]
        except Exception:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()

        atom_positions = Compound3DKit.get_atom_positions(new_mol, conf)
        if return_energy:
            return new_mol, atom_positions, energy
        else:
            return new_mol, atom_positions
        
    @staticmethod
    def get_2d_atom_positions(mol):
        """
        Generate 2D coordinates for atoms in the molecule.

        Args:
            mol: RDKit molecule object.

        Returns:
            A list of 2D coordinates for each atom.
        """
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        return Compound3DKit.get_atom_positions(mol, conf)
    
    @staticmethod
    def get_pairwise_distances(atom_positions):
        """
        Calculate pairwise distances between all atoms in the molecule.

        Args:
            atom_positions: List of 3D coordinates for each atom.

        Returns:
            A matrix of pairwise distances between atoms.
        """
        atom_number = len(atom_positions)
        distances = np.zeros((atom_number, atom_number), dtype='float32')
        for i in range(atom_number):
            for j in range(atom_number):
                distances[i, j] = np.linalg.norm(atom_positions[i] - atom_positions[j])
        return distances
    
    @staticmethod
    def get_bond_angles(atom_positions, angles_atom_index):
        """
        Calculate bond angles for specified triplets of atoms.

        Args:
            atom_positions: List of 3D coordinates for each atom.
            angles_atom_index: List of triplets (i, j, k) representing the atoms forming the angles.

        Returns:
            A list of bond angles in radians.
        """
        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)  # Prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            return np.arccos(np.dot(vec1, vec2))
        
        angles_atom_index = np.array(angles_atom_index, 'int64')
        return np.array([
            _get_angle(
                atom_positions[angle[0]] - atom_positions[angle[1]],
                atom_positions[angle[2]] - atom_positions[angle[1]]
            ) for angle in angles_atom_index
        ], dtype='float32')

    @staticmethod
    def get_edge_distances(atom_positions, edges):
        """
        Calculate pairwise distances between midpoints of edges.

        Args:
            atom_positions: List of 3D coordinates for each atom.
            edges: List of edges (pairs of atom indices).

        Returns:
            A matrix of pairwise distances between edge midpoints.
        """
        edges_number = len(edges)
        bond_positions = [
            (atom_positions[edge[0]] + atom_positions[edge[1]]) / 2
            for edge in edges
        ]

        distances = np.zeros((edges_number, edges_number), dtype='float32')
        for i in range(edges_number):
            for j in range(edges_number):
                distances[i, j] = np.linalg.norm(bond_positions[i] - bond_positions[j])
        return distances
    
    @staticmethod
    def get_atom_bond_distances(atom_positions, edges):
        """
        Calculate distances between atoms and bond midpoints.

        Args:
            atom_positions: List of 3D coordinates for each atom.
            edges: List of edges (pairs of atom indices).

        Returns:
            A matrix of distances between atoms and bond midpoints.
        """
        atom_number = len(atom_positions)
        edges_number = len(edges)

        bond_positions = [
            (atom_positions[edge[0]] + atom_positions[edge[1]]) / 2
            for edge in edges
        ]

        distances = np.zeros((atom_number, edges_number), dtype='float32')
        for i in range(atom_number):
            for j in range(edges_number):
                distances[i, j] = np.linalg.norm(atom_positions[i] - bond_positions[j])
        return distances
    
    @staticmethod
    def get_angles_list(bond_angles, angles_bond_index):
        angles_bond_index = np.array(angles_bond_index)
        if len(angles_bond_index.shape) == 0:
            return np.array([0])
        else:
            angles_list = bond_angles[angles_bond_index[:, 0], angles_bond_index[:, 1]]
            return angles_list

    @staticmethod
    def get_bond_distances(pair_distances, edges):
        edges_number = len(edges)
        bond_distances = []
        for i in range(edges_number):
            bond_distances.append(pair_distances[edges[i][0]][edges[i][1]])
        bond_distances = np.array(bond_distances, 'float32')
        return bond_distances
    

def find_angel_index(edges):
    """
    Find the angel index.

    Parameters
    ----------
    edges: np.ndarray
        The edges.
    atom_num: int
        The number of atoms.

    Returns
    -------
    np.ndarray
        The angel index.
    """
    angles_atom_index = []
    angles_bond_index = []
    for ii in range(len(edges)):
        for jj in range(len(edges)):
            i = edges[ii]
            j = edges[jj]
            if ii != jj:
                if i[1] == j[0]:
                    if angles_atom_index.count([j[1], i[1], i[0]]) == 0:
                        angles_atom_index.append([i[0], i[1], j[1]])
                        angles_bond_index.append([ii, jj])
                elif i[0] == j[1]:
                    if angles_atom_index.count([j[0], i[0], i[1]]) == 0:
                        angles_atom_index.append([i[1], i[0], j[0]])
                        angles_bond_index.append([ii, jj])
                elif i[0] == j[0]:
                    if angles_atom_index.count([j[1], i[0], i[1]]) == 0:
                        angles_atom_index.append([i[1], i[0], j[1]])
                        angles_bond_index.append([ii, jj])
                elif i[1] == j[1]:
                    if angles_atom_index.count([j[0], i[1], i[0]]) == 0:
                        angles_atom_index.append([i[0], i[1], j[0]])
                        angles_bond_index.append([ii, jj])

    return angles_atom_index, angles_bond_index

def get_dist_bar(dist: np.ndarray, percentiles: list):
    # print(percentiles)
    try:
        result = np.percentile(dist, percentiles)
    except Exception as e:
        result = np.zeros(len(percentiles))

    return result

def set_up_spatial_pos(matrix_, up=10):
    matrix_[matrix_ > up] = up
    return matrix_


def get_spatial_pos(data_len, edges):
    adj = torch.zeros([data_len, data_len], dtype=torch.bool)
    adj[edges[:, 0], edges[:, 1]] = True
    adj[edges[:, 1], edges[:, 0]] = True
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    spatial_pos = shortest_path_result
    # spatial_pos = torch.from_numpy(shortest_path_result).long()
    spatial_pos = set_up_spatial_pos(spatial_pos, up=20)
    return spatial_pos


def add_spatial_pos(data):
    data_len = len(data['atomic_num'])
    adj = torch.zeros([data_len, data_len], dtype=torch.bool)
    adj[data['edges'][:, 0], data['edges'][:, 1]] = True
    adj[data['edges'][:, 1], data['edges'][:, 0]] = True
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    spatial_pos = shortest_path_result
    data['spatial_pos'] = spatial_pos
    return data


def binning_matrix(matrix, m, range_min, range_max):
    # 计算每个 bin 的范围
    bin_width = (range_max - range_min) / m

    # 将矩阵中的元素映射到 bin 中
    bin_indices = np.floor((matrix - range_min) / bin_width).astype(int)

    # 将超出范围的值限制在范围内
    bin_indices = np.clip(bin_indices, 0, m - 1)

    return bin_indices


def mol_to_data_pkl(mol, pre_calculated_compose=None):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if len(mol.GetAtoms()) == 0:
        return None
    if len(mol.GetAtoms()) <= 400:
        if pre_calculated_compose is not None:
            mol, atom_poses = mol, pre_calculated_compose
        else:
            mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
            # mol, atom_poses = Compound3DKit.get_MMFF_atom_poses_nonminimum(mol, numConfs=50, percent=0)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)


    atom_id_names = list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names

    data = {}

    ### atom features
    data = {name: [] for name in atom_id_names}

    raw_atom_feat_dicts = CompoundKit.get_atom_names(mol)
    for atom_feat in raw_atom_feat_dicts:
        for name in atom_id_names:
            data[name].append(atom_feat[name])

    data['edges'] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j
        data['edges'] += [(i, j)]

    #### self loop
    if len(data['edges']) == 0:
        N = len(data[atom_id_names[0]])
        for i in range(N):
            data['edges'] += [(i, i)]

    ### make ndarray and check length
    for name in list(CompoundKit.atom_vocab_dict.keys()):
        data[name] = np.array(data[name], 'int64')
    for name in CompoundKit.atom_float_names:
        data[name] = np.array(data[name], 'float32')
    data['edges'] = np.array(data['edges'], 'int64')

    angles_atom_index, angles_bond_index = find_angel_index(data['edges'])
    data['angles_atom_index'] = angles_atom_index
    data['angles_bond_index'] = angles_bond_index
    if len(angles_atom_index) == 0:
        # print("error")
        # print(item['smiles'])
        data['angles_atom_index'] = [[0, 0, 0]]
        data['angles_bond_index'] = [[0, 0]]

    atom_poses = np.array(atom_poses, 'float32')
    data['atom_pos'] = np.array(atom_poses, 'float32')
    data['pair_distances'] = Compound3DKit.get_pair_distances(atom_poses)
    data['bond_distances'] = Compound3DKit.get_bond_distances(data['pair_distances'], data['edges'])
    data['bond_angles'] = Compound3DKit.get_bond_angles(atom_poses, data['angles_atom_index'])
    data['edge_distances'] = Compound3DKit.get_edge_distances(atom_poses, data['edges'])

    data = add_spatial_pos(data)
    data['atom_bond_distances'] = Compound3DKit.get_atom_bond_distances(atom_poses, data['edges'])
    data['pair_distances_bin'] = binning_matrix(data['pair_distances'], 30, 0, 30)
    data['bond_angles_bin'] = binning_matrix(data['bond_angles'], 20, 0, math.pi)
    data['spatial_pos'] = get_spatial_pos(len(data['atom_pos']), data['edges'])

    return data


def get_edge_group_indices(mol):
    if len(mol.GetAtoms()) <= 400:
        mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names

    data = {}

    ### atom features
    data = {name: [] for name in atom_id_names}

    raw_atom_feat_dicts = CompoundKit.get_atom_names(mol)
    for atom_feat in raw_atom_feat_dicts:
        for name in atom_id_names:
            data[name].append(atom_feat[name])

    ### bond and bond features
    data['edges'] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j
        data['edges'] += [(i, j)]


    data['edges'] = np.array(data['edges'], 'int64')

    function_group_index, function_group_bond = CompoundKit.get_function_group_index(mol, data['edges'])
    return function_group_index, function_group_bond


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        # print(atom_scores[atom.GetIdx()])
        atom.SetAtomMapNum(atom.GetIdx())
    return mol



def mol_with_atom_index(mol, style='clear'):
    assert style in ['traditional', 'clear', 'all']
    if style == 'clear':
        for atom in mol.GetAtoms():
            atom.SetProp("atomNote", str(atom.GetIdx()))
    elif style == 'traditional':
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
    elif style == 'all':
        for atom in mol.GetAtoms():
            atom.SetProp("atomNote", str(atom.GetIdx()))
            atom.SetAtomMapNum(atom.GetIdx())
    return mol


def get_bond_type_str(bond: Chem.Bond) -> str:
    a1 = bond.GetBeginAtom().GetSymbol()
    a2 = bond.GetEndAtom().GetSymbol()
    # Sort symbols to account for symmetry (e.g., C=O is the same as O=C)
    a1, a2 = sorted([a1, a2])

    # Determine bond type
    bond_type = bond.GetBondType()
    if bond_type == Chem.rdchem.BondType.SINGLE:
        bond_str = f"{a1}-{a2}"
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        bond_str = f"{a1}={a2}"
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        bond_str = f"{a1}#{a2}"
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        bond_str = f"{a1}~={a2}"
    else:
        bond_str = f"{a1} <{bond_type}> {a2}"
    return bond_str
