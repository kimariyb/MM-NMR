import ast
import numpy as np

from rdkit import RDLogger, Chem
from rdkit.Chem import AllChem

from utils.FingerPrint import GetPubChemFPs
from typing import List

from ogb.utils.features import atom_to_feature_vector

# Disable rdkit warnings
RDLogger.DisableLog('rdApp.*')

# 可用的分子特征
ALLOWABLE_FEATURES = {
    'atomic_num_list': ['H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I'],
    'formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'chirality_list': [
        Chem.ChiralType.CHI_TETRAHEDRAL_CW, 
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.ChiralType.CHI_OTHER, 
        Chem.ChiralType.CHI_UNSPECIFIED, 
        Chem.ChiralType.CHI_TETRAHEDRAL,
        Chem.ChiralType.CHI_ALLENE 	
    ],
    'hybridization_list': [
        Chem.HybridizationType.S, 
        Chem.HybridizationType.SP, 
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3, 
        Chem.HybridizationType.SP3D, 
        Chem.HybridizationType.SP3D2,
        Chem.HybridizationType.UNSPECIFIED 	
    ],
    'degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'number_H_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'number_radical_e_list': [0, 1, 2, 3, 4],
    'implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'is_aromatic_list': [False, True],
    'is_in_ring_list': [False, True],
    'bond_type_list': [
        Chem.rdchem.BondType.SINGLE, 
        Chem.rdchem.BondType.DOUBLE, 
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'bond_stereo_list': [
        Chem.rdchem.BondStereo.STEREONONE, 
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ, 
        Chem.rdchem.BondStereo.STEREOE, 
        Chem.rdchem.BondStereo.STEREOCIS ,
        Chem.rdchem.BondStereo.STEREOTRANS, 
        Chem.rdchem.BondStereo.STEREOATROPCW, 
        Chem.rdchem.BondStereo.STEREOATROPCCW 
    ],
    'is_conjugated_list': [False, True],
}


SMILES_DICT = {v: (i + 1) for i, v in enumerate("(.02468@BDFHLNPRTVZ/bdfhlnprt#*%)+-/13579=ACEGIKMOSUWY[]acegimosuy\\")}


def GernerateMask(mol, shift_dict):
    r"""
    生成 13C、19F 等杂核化学位移掩码
    """
    for j, atom in enumerate(mol.GetAtoms()):
        if j in shift_dict:
            atom.SetProp('shift', str(shift_dict[j]))
            atom.SetBoolProp('mask', True)
        else:
            atom.SetProp( 'shift', str(0))
            atom.SetBoolProp('mask', False)

    mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])
    shift = np.array([ast.literal_eval(atom.GetProp('shift')) for atom in mol.GetAtoms()])
    
    return mask, shift


def GernerateHydrogenMask(mol, shift_dict):
    r"""
    生成 1H 化学位移掩码
    """
    for j, atom in enumerate(mol.GetAtoms()):
        if j in shift_dict:
            atom.SetProp('shift', str(shift_dict[j]))
            atom.SetBoolProp('mask', True)
        else:
            atom.SetProp('shift', str([0]))             
            atom.SetBoolProp('mask', False)

    mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])
    shift = np.array([ast.literal_eval(atom.GetProp('shift')) for atom in mol.GetAtoms()])
    
    return mask, shift


def FeatureEncoding(value: int, allowable_set: List[int]) -> List[int]:
    r"""
    One-hot encoding of a value given a list of possible values.

    Parameters
    ----------
    value : int
        The value to be encoded.
    allowable_set : List[int]
        The list of possible values.

    Returns
    -------
    List[int]
        The one-hot encoded list.
    """
    if value not in allowable_set:
        raise ValueError(f"Value {value} not in allowable set {allowable_set}")
    
    return [int(x == value) for x in allowable_set]


def AtomToFeature(atom: Chem.rdchem.Atom):
    r"""
    将 Atom 对象转换为特征向量
    
    Parameters
    ----------
    atom : Chem.rdchem.Atom
        The atom to be converted.

    Returns
    -------
    List[int]
        The feature vector of the atom.
    
    Features
    --------
    Atomic Number: 
        One-hot encoding of the atomic number
    Formal Charge: 
        One-hot encoding of the formal charge
    Chirality: 
        One-hot encoding of the chirality
    Hybridization: 
        One-hot encoding of the hybridization
    Degree: 
        One-hot encoding of the degree
    Is Aromatic: 
        One-hot encoding of whether the atom is aromatic
    Is Rings:
        One-hot encoding of whether the atom is in a ring
    Implicit Valence: 
        One-hot encoding of the implicit valence
    No. Radical Electrons: 
        One-hot encoding of the number of radical electrons
    No. Hydrogens: 
        One-hot encoding of the number of hydrogens
    """
    # Atomic number one-hot encoding
    atomic_number_onehot = FeatureEncoding(value=atom.GetSymbol(), allowable_set=ALLOWABLE_FEATURES['atomic_num_list'])

    # Formal charge one-hot encoding
    formal_charge_onehot = FeatureEncoding(value=atom.GetFormalCharge(), allowable_set=ALLOWABLE_FEATURES['formal_charge_list'])

    # Chirality one-hot encoding
    chirality_onehot = FeatureEncoding(value=atom.GetChiralTag(), allowable_set=ALLOWABLE_FEATURES['chirality_list'])
    
    # Hybridization one-hot encoding
    hybridization_onehot = FeatureEncoding(value=atom.GetHybridization(), allowable_set=ALLOWABLE_FEATURES['hybridization_list'])
        
    # Is aromatic one-hot encoding
    is_aromatic_onehot = FeatureEncoding(value=atom.GetIsAromatic(), allowable_set=ALLOWABLE_FEATURES['is_aromatic_list'])
    
    # Is rings one-hot encoding
    is_rings_onehot = FeatureEncoding(value=atom.IsInRing(), allowable_set=ALLOWABLE_FEATURES['is_in_ring_list'])

    # Degree one-hot encoding
    degree_onehot = FeatureEncoding(value=atom.GetDegree(), allowable_set=ALLOWABLE_FEATURES['degree_list'])

    # Implicit valence one-hot encoding
    implicit_valence_onehot = FeatureEncoding(value=atom.GetImplicitValence(), allowable_set=ALLOWABLE_FEATURES['implicit_valence_list'])
    
    # No. radical electrons one-hot encoding
    no_radical_electrons_onehot = FeatureEncoding(value=atom.GetNumRadicalElectrons(), allowable_set=ALLOWABLE_FEATURES['number_radical_e_list'])

    # No. hydrogens one-hot encoding
    no_hydrogens_onehot = FeatureEncoding(value=atom.GetTotalNumHs(), allowable_set=ALLOWABLE_FEATURES['number_H_list'])
    
    features = atomic_number_onehot + formal_charge_onehot + chirality_onehot\
        + hybridization_onehot + is_aromatic_onehot + is_rings_onehot + degree_onehot\
        + implicit_valence_onehot + no_radical_electrons_onehot + no_hydrogens_onehot 
        
    return features


def BondToFeature(bond: Chem.rdchem.Bond):
    r"""
    将 Bond 对象转换为特征向量
    
    Parameters
    ----------
    bond : Chem.rdchem.Bond
        The bond to be converted.

    Returns
    -------
    List[int]
        The feature vector of the bond.
    
    Features
    --------
    Bond Type:
        One-hot encoding of the bond type
    Stereo:
        One-hot encoding of the stereo
    Is Conjugated:
        One-hot encoding of whether the bond is conjugated
    Is in Ring:
        One-hot encoding of whether the bond is in a ring
    """
    
    # Bond type one-hot encoding
    bond_type_onehot = FeatureEncoding(value=bond.GetBondType(), allowable_set=ALLOWABLE_FEATURES['bond_type_list'])
    
    # Stereo one-hot encoding
    stereo_onehot = FeatureEncoding(value=bond.GetStereo(), allowable_set=ALLOWABLE_FEATURES['bond_stereo_list'])
    
    # Is conjugated one-hot encoding
    is_conjugated_onehot = FeatureEncoding(value=bond.GetIsConjugated(), allowable_set=ALLOWABLE_FEATURES['is_conjugated_list'])
    
    # Is in ring one-hot encoding
    is_in_ring_onehot = FeatureEncoding(value=bond.IsInRing(), allowable_set=ALLOWABLE_FEATURES['is_in_ring_list'])
    
    features = bond_type_onehot + stereo_onehot + is_conjugated_onehot + is_in_ring_onehot
    
    return features

    
def MolToGraph(mol: Chem.rdchem.Mol):
    r"""
    将分子转换为图数据
    
    Parameters
    ----------
    mol : Chem.rdchem.Mol
        The molecule to be converted.

    Returns
    -------
    dict
        The graph data of the molecule.
    
    Features
    --------
    Node Features:
        The feature vector of each atom in the molecule.
    Edge Index:
        The index of each edge in the molecule.
    Edge Features:
        The feature vector of each edge in the molecule.
    """        
    # 获取分子的原子特征
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(AtomToFeature(atom))

    x = np.array(atom_features, dtype=np.int64)
    
    # 获取分子的边特征
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_feature = BondToFeature(bond)
            
            edges_list.append((i, j))
            edge_features.append(edge_feature)
            edges_list.append((j, i))
            edge_features.append(edge_feature)
        
        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features, dtype=np.int64)        
        
    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 4), dtype=np.int64)
        
    # 构建图数据         
    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)

    return graph


def SmilesToVector(smiles: str):
    r"""
    将 SMILES 字符串转换为特征向量
    
    Parameters
    ----------
    smiles : str
        The SMILES string to be converted.

    Returns
    -------
    np.array
        The feature vector of the SMILES string.
    """
    max_len = 100
    
    x = np.zeros(max_len)
    for i, ch in enumerate(smiles[: max_len]):
        x[i] = SMILES_DICT[ch]
    
    return np.array(x, dtype=np.int64)


def MolToFingerprints(mol):
    r"""
    将分子转换为指纹
    
    Returns
    -------
    np.array
        MACCS, ErG, PubChem 指纹
    """
    fp = []

    fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)  # maccs指纹
    fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)  # ErGFingerprint指纹
    fp_pubcfp = GetPubChemFPs(mol) # PubChem指纹
    
    fp.extend(fp_maccs)
    fp.extend(fp_phaErGfp)
    fp.extend(fp_pubcfp)

    return np.array(fp)


def GetAtomFeaturesDim():
    return list(map(len, [
        ALLOWABLE_FEATURES['atomic_num_list'],
        ALLOWABLE_FEATURES['formal_charge_list'],
        ALLOWABLE_FEATURES['chirality_list'],
        ALLOWABLE_FEATURES['hybridization_list'],
        ALLOWABLE_FEATURES['is_aromatic_list'],
        ALLOWABLE_FEATURES['is_in_ring_list'],
        ALLOWABLE_FEATURES['degree_list'],
        ALLOWABLE_FEATURES['implicit_valence_list'],
        ALLOWABLE_FEATURES['number_radical_e_list'],
        ALLOWABLE_FEATURES['number_H_list']
    ]))

def GetBondFeaturesDim():
    return list(map(len, [
        ALLOWABLE_FEATURES['bond_type_list'],
        ALLOWABLE_FEATURES['bond_stereo_list'],
        ALLOWABLE_FEATURES['is_conjugated_list'],
        ALLOWABLE_FEATURES['is_in_ring_list']
    ]))