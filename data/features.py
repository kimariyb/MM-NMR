import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem, AllChem


RDLogger.DisableLog('rdApp.*')


def one_encoding_unk(x, allowable_set):
    r"""
    One-hot encoding for a given value with an additional "unknown" value.

    Parameters
    ----------
    x : object
        The value to encode.
    allowable_set : list of object
        The set of allowable values.

    Returns
    -------
    list of float
        The one-hot encoding of the value.
    """
    if x not in allowable_set:
        return allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_features(atom: rdchem.Atom):
    r"""
    Get atom features for a given atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        The atom for which to get features.

    Returns
    -------
    list of float
        The atom features.
    """
    possible_set = [
        'H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 
        'S', 'Cl', 'Br', 'I'
    ]
    
    h = []
    h += one_encoding_unk(atom.GetSymbol(), possible_set) # 12
    h += one_encoding_unk(atom.GetDegree(), list(range(7))) # 7
    h += one_encoding_unk(atom.GetTotalNumHs(), list(range(5)))  # 5
    h.append(atom.GetFormalCharge())  # 1
    h.append(atom.GetNumRadicalElectrons())  # 1
    
    h += one_encoding_unk(atom.GetHybridization(), [
        rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2, rdchem.HybridizationType.UNSPECIFIED,
    ])  # 6
    
    h.append(atom.GetIsAromatic())  # 1
    h.append(atom.IsInRing())  # 1
    h += one_encoding_unk(atom.GetTotalNumHs(), list(range(5)))  # 5
    h.append(atom.HasProp('_ChiralityPossible'))  # 1
    
    if not atom.HasProp('_CIPCode'):
        h += [False, False]
    else:
        cip = atom.GetProp('_CIPCode')
        h += one_encoding_unk(cip, ['R', 'S'])
    
    return h


def get_bond_features(bond: rdchem.Bond):
    r"""
    Get bond features for a given bond.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        The bond for which to get features.

    Returns
    -------
    list of float
        The bond features.
    """    
    h = []
    
    bond_type = bond.GetBondType()
    h += one_encoding_unk(bond_type, [
        rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC
    ])  # 4
    
    h.append(bond.GetIsConjugated())  # 1
    h.append(bond.IsInRing())  # 1
    bond_stereo = bond.GetStereo()
    h += one_encoding_unk(bond_stereo, [
        rdchem.BondStereo.STEREONONE, rdchem.BondStereo.STEREOANY,
        rdchem.BondStereo.STEREOZ, rdchem.BondStereo.STEREOE,
    ])  # 4
    
    return h


def get_atom_features_dim():
    r"""
    Get the dimension of the atom features.

    Returns
    -------
    int
        The dimension of the atom features.
    """
    return len(get_atom_features(Chem.MolFromSmiles('CC').GetAtoms()[0]))


def get_bond_features_dim():
    r"""
    Get the dimension of the bond features.

    Returns
    -------
    int
        The dimension of the bond features.
    """
    return len(get_bond_features(Chem.MolFromSmiles('CC').GetBonds()[0]))


def mol2graph(mol: rdchem.Mol):
    r"""
    Convert a molecule to a graph representation.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The molecule to convert.

    Returns
    -------
    dict
        The graph representation of the molecule.
    """
    atom_features_list = []
    edges_list = []
    edge_features_list = []
    
    # mol has bonds
    if len(mol.GetBonds()) > 0: 
        for atom in mol.GetAtoms():
            atom_features_list.append(get_atom_features(atom))
    
        # atom features
        x = np.array(atom_features_list, dtype=np.int64)

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = get_bond_features(bond=bond)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.int64)
    else:
        return None
    
    graph = dict()
    graph['num_nodes'] = len(x)
    graph['node_feat'] = x
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr

    return graph


def mol2geometry(mol):
    r"""
    Convert a molecule to a geometry representation.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The molecule to convert.

    Returns
    -------
    np.array
        The geometry representation of the molecule.
    np.array
        The atomic numbers of the atoms in the molecule.
    """
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)

        params = Chem.rdDistGeom.ETKDGv3()
        params.numThreads = 12
        params.enforceChirality = True
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        params.pruneRmsThresh = 0.001
        params.maxAttempts = 10000
        
        # Embed the molecule
        AllChem.EmbedMolecule(mol, params)
        # MMFF optimization
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Remove Hs
        mol = Chem.RemoveHs(mol)
        pos = mol.GetConformer().GetPositions()
        z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

        return pos, z
    
    except:
        return None, None


if __name__ == '__main__':
    node_feats = get_atom_features_dim()
    edges_feats = get_bond_features_dim()
    print("node_feats: ", node_feats)
    print("edges_feats: ", edges_feats)