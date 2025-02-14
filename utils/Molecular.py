import numpy as np

from rdkit import Chem
from rdkit.Chem import rdchem, AllChem, rdPartialCharges

from utils.Constants import SMARTS_LIST


def EnumerateFeatures(values):
    return [values[i] for i in range(len(values))]


def SafeIndex(lst, element):
    try:
        return lst.index(element)
    except ValueError:
        return -1


class CompoundKit:
    atom_vocab_dict = {
        "atomic_num": list(range(1, 119)) + ['misc'],
        "chiral_tag": EnumerateFeatures(rdchem.ChiralType.values),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "hybridization": EnumerateFeatures(rdchem.HybridizationType.values),
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "is_aromatic": [0, 1],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'num_radical_e': [0, 1, 2, 3, 4, 'misc'],
        'atom_is_in_ring': [0, 1],
        'valence_out_shell': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size5': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size6': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size7': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size8': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    }
    bond_vocab_dict = {
        "bond_dir": EnumerateFeatures(rdchem.BondDir.values),
        "bond_type": EnumerateFeatures(rdchem.BondType.values),
        "is_in_ring": [0, 1],

        'bond_stereo': EnumerateFeatures(rdchem.BondStereo.values),
        'is_conjugated': [0, 1],
    }

    atom_float_names =["van_der_waals_radis", "partial_charge", "mass"]
    bond_float_names = ["bond_length", "bond_angle"]

    smarts_list = SMARTS_LIST
    mol_list = [Chem.MolFromSmarts(smarts) for smarts in SMARTS_LIST]

    morgan_fp_N = 200
    morgan_2048_fp_N = 2048
    maccs_fp_N = 167

    period_table = Chem.GetPeriodicTable()

    @staticmethod
    def get_atom_value(atom, name):
        """get atom values"""
        if name == 'atomic_num':
            return atom.GetAtomicNum()
        elif name == 'chiral_tag':
            return atom.GetChiralTag()
        elif name == 'degree':
            return atom.GetDegree()
        elif name == 'explicit_valence':
            return atom.GetExplicitValence()
        elif name == 'formal_charge':
            return atom.GetFormalCharge()
        elif name == 'hybridization':
            return atom.GetHybridization()
        elif name == 'implicit_valence':
            return atom.GetImplicitValence()
        elif name == 'is_aromatic':
            return int(atom.GetIsAromatic())
        elif name == 'mass':
            return int(atom.GetMass())
        elif name == 'total_numHs':
            return atom.GetTotalNumHs()
        elif name == 'num_radical_e':
            return atom.GetNumRadicalElectrons()
        elif name == 'atom_is_in_ring':
            return int(atom.IsInRing())
        elif name == 'valence_out_shell':
            return CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
        else:
            raise ValueError(name)

    @staticmethod
    def get_atom_feature_id(atom, name):
        r"""get atom features id"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name

        return SafeIndex(CompoundKit.atom_vocab_dict[name], CompoundKit.get_atom_value(atom, name))

    @staticmethod
    def get_atom_feature_size(name):
        r"""get atom features size"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name

        return len(CompoundKit.atom_vocab_dict[name])

    @staticmethod
    def get_bond_value(bond, name):
        """get bond values"""
        if name == 'bond_dir':
            return bond.GetBondDir()
        elif name == 'bond_type':
            return bond.GetBondType()
        elif name == 'is_in_ring':
            return int(bond.IsInRing())
        elif name == 'is_conjugated':
            return int(bond.GetIsConjugated())
        elif name == 'bond_stereo':
            return bond.GetStereo()
        else:
            raise ValueError(name)

    @staticmethod
    def get_bond_feature_id(bond, name):
        """get bond features id"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name

        return SafeIndex(CompoundKit.bond_vocab_dict[name], CompoundKit.get_bond_value(bond, name))

    @staticmethod
    def get_bond_feature_size(name):
        """get bond features size"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name

        return len(CompoundKit.bond_vocab_dict[name])

    @staticmethod
    def get_morgan_fingerprint(mol, radius=2):
        """get morgan fingerprint"""
        nBits = CompoundKit.morgan_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_morgan2048_fingerprint(mol, radius=2):
        """get morgan2048 fingerprint"""
        nBits = CompoundKit.morgan_2048_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_maccs_fingerprint(mol):
        """get maccs fingerprint"""
        fp = AllChem.GetMACCSKeysFingerprint(mol)

        return [int(b) for b in fp.ToBitString()]


    @staticmethod
    def get_daylight_functional_group_counts(mol):
        """get daylight functional group counts"""
        fg_counts = []
        for fg_mol in CompoundKit.mol_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))

        return fg_counts

    @staticmethod
    def get_ring_size(mol):
        """return (N,6) list"""
        rings = mol.GetRingInfo()
        rings_info = []
        for r in rings.AtomRings():
            rings_info.append(r)
        ring_list = []
        for atom in mol.GetAtoms():
            atom_result = []
            for ringsize in range(3, 9):
                num_of_ring_at_ringsize = 0
                for r in rings_info:
                    if len(r) == ringsize and atom.GetIdx() in r:
                        num_of_ring_at_ringsize += 1
                if num_of_ring_at_ringsize > 8:
                    num_of_ring_at_ringsize = 9
                atom_result.append(num_of_ring_at_ringsize)

            ring_list.append(atom_result)

        return ring_list

    @staticmethod
    def atom_to_feat_vector(atom):
        atom_names = {
            "atomic_num": SafeIndex(CompoundKit.atom_vocab_dict["atomic_num"], atom.GetAtomicNum()),
            "chiral_tag": SafeIndex(CompoundKit.atom_vocab_dict["chiral_tag"], atom.GetChiralTag()),
            "degree": SafeIndex(CompoundKit.atom_vocab_dict["degree"], atom.GetTotalDegree()),
            "explicit_valence": SafeIndex(CompoundKit.atom_vocab_dict["explicit_valence"], atom.GetExplicitValence()),
            "formal_charge": SafeIndex(CompoundKit.atom_vocab_dict["formal_charge"], atom.GetFormalCharge()),
            "hybridization": SafeIndex(CompoundKit.atom_vocab_dict["hybridization"], atom.GetHybridization()),
            "implicit_valence": SafeIndex(CompoundKit.atom_vocab_dict["implicit_valence"], atom.GetImplicitValence()),
            "is_aromatic": SafeIndex(CompoundKit.atom_vocab_dict["is_aromatic"], int(atom.GetIsAromatic())),
            "total_numHs": SafeIndex(CompoundKit.atom_vocab_dict["total_numHs"], atom.GetTotalNumHs()),
            'num_radical_e': SafeIndex(CompoundKit.atom_vocab_dict['num_radical_e'], atom.GetNumRadicalElectrons()),
            'atom_is_in_ring': SafeIndex(CompoundKit.atom_vocab_dict['atom_is_in_ring'], int(atom.IsInRing())),
            'valence_out_shell': SafeIndex(CompoundKit.atom_vocab_dict['valence_out_shell'],
                                            CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())),
            'van_der_waals_radis': CompoundKit.period_table.GetRvdw(atom.GetAtomicNum()),
            'partial_charge': CompoundKit.check_partial_charge(atom),
            'mass': atom.GetMass(),
        }
        return atom_names

    @staticmethod
    def get_atom_names(mol):
        """get atom name list"""
        atom_features_dicts = []
        rdPartialCharges.ComputeGasteigerCharges(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts.append(CompoundKit.atom_to_feat_vector(atom))

        ring_list = CompoundKit.get_ring_size(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts[i]['in_num_ring_with_size3'] = SafeIndex(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size3'], ring_list[i][0])
            atom_features_dicts[i]['in_num_ring_with_size4'] = SafeIndex(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size4'], ring_list[i][1])
            atom_features_dicts[i]['in_num_ring_with_size5'] = SafeIndex(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size5'], ring_list[i][2])
            atom_features_dicts[i]['in_num_ring_with_size6'] = SafeIndex(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size6'], ring_list[i][3])
            atom_features_dicts[i]['in_num_ring_with_size7'] = SafeIndex(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size7'], ring_list[i][4])
            atom_features_dicts[i]['in_num_ring_with_size8'] = SafeIndex(
                CompoundKit.atom_vocab_dict['in_num_ring_with_size8'], ring_list[i][5])

        return atom_features_dicts

    @staticmethod
    def check_partial_charge(atom):
        """tbd"""
        pc = atom.GetDoubleProp('_GasteigerCharge')
        if pc != pc:
            # unsupported atom, replace nan with 0
            pc = 0
        if pc == float('inf'):
            # max 4 for other atoms, set to 10 here if inf is get
            pc = 10

        return pc


def MolToGraphData(mol):
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = [
        "atomic_num", "chiral_tag", "degree", "explicit_valence",
        "formal_charge", "hybridization", "implicit_valence",
        "is_aromatic", "total_numHs",
    ]
    bond_id_names = [
        "bond_dir", "bond_type", "is_in_ring",
    ]

    data = {}
    for name in atom_id_names:
        data[name] = []
    data['mass'] = []
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    # atom features
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return None
        for name in atom_id_names:
            data[name].append(CompoundKit.get_atom_feature_id(atom, name) + 1)  # 0: OOV
        data['mass'].append(CompoundKit.get_atom_value(atom, 'mass') * 0.01)

    # bond features
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j and j->i
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name) + 1  # 0: OOV
            data[name] += [bond_feature_id] * 2

    # self loop (+2)
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = CompoundKit.get_bond_feature_size(name) + 2  # N + 2: self loop
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

    # morgan fingerprint
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    # data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
    data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')

    return data


class Compound3DKit(object):
    """the 3Dkit of Compound"""
    @staticmethod
    def get_atom_poses(mol, conf):
        """tbd"""
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
        """the atoms of mol will be changed in some cases."""
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
            ### MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            index = np.argmin([x[1] for x in res])
            energy = res[index][1]
            conf = new_mol.GetConformer(id=int(index))
        except:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()

        atom_poses = Compound3DKit.get_atom_poses(new_mol, conf)
        if return_energy:
            return new_mol, atom_poses, energy
        else:
            return new_mol, atom_poses

    @staticmethod
    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return atom_poses

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, 'float32')
        return bond_lengths

    @staticmethod
    def get_superedge_angles(edges, atom_poses, dir_type='HT'):
        """get superedge angles"""
        def _get_vec(atom_poses, edge):
            return atom_poses[edge[1]] - atom_poses[edge[0]]
        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        E = len(edges)
        edge_indices = np.arange(E)
        super_edges = []
        bond_angles = []
        bond_angle_dirs = []
        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            if dir_type == 'HT':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            elif dir_type == 'HH':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
            else:
                raise ValueError(dir_type)
            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = _get_vec(atom_poses, src_edge)
                tar_vec = _get_vec(atom_poses, tar_edge)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(src_vec, tar_vec)
                bond_angles.append(angle)
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])  # H -> H or H -> T

        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], 'int64')
            bond_angles = np.zeros([0,], 'float32')
        else:
            super_edges = np.array(super_edges, 'int64')
            bond_angles = np.array(bond_angles, 'float32')

        return super_edges, bond_angles, bond_angle_dirs


def MolToGeoGNNGraphData(mol, atom_poses, dir_type):
    """
    mol: rdkit molecule
    dir_type: direction type for bond_angle grpah
    """
    if len(mol.GetAtoms()) == 0:
        return None

    data = MolToGraphData(mol)

    data['atom_pos'] = np.array(atom_poses, 'float32')
    data['bond_length'] = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos'])
    BondAngleGraph_edges, bond_angles, bond_angle_dirs = \
            Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'])
    data['BondAngleGraph_edges'] = BondAngleGraph_edges
    data['bond_angle'] = np.array(bond_angles, 'float32')

    return data


def GeoGNNGraphDataMMFF3d(mol):
    """tbd"""
    if len(mol.GetAtoms()) <= 400:
        mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)

    return MolToGeoGNNGraphData(mol, atom_poses, dir_type='HT')