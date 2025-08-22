import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdForceFieldHelpers
from ogb.utils.features import atom_to_feature_vector
from typing import List, Dict, Any, Optional, Tuple, Union, Callable



class MoleculeEncoder:
    """
    一个用于将分子编码为特征向量的类。
    
    这个类可以用于：
    1. 将分子编码为特征向量
    2. 将原子编码为特征向量
    3. 将键编码为特征向量
    """
    def __init__(self):
        # 原子数，对应 H, B, C, N, O, F, Si, P, S, Cl, Br
        self.atomic_numbers = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35]
        self.max_atomic_num = 100
        # 原子杂化
        self.hybridizations = [
            Chem.HybridizationType.UNSPECIFIED,
            Chem.HybridizationType.SP,
            Chem.HybridizationType.SP2,
            Chem.HybridizationType.SP3,
            Chem.HybridizationType.SP3D,
            Chem.HybridizationType.SP3D2
        ]
        # 原子手性
        self.chirality_tags = [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ]

        # 键类型
        self.bond_types = [
            Chem.BondType.UNSPECIFIED,
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC
        ]

        # 键立体属性
        self.bond_stereo = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS
        ]

    def one_hot_encode(self, value, enum_list, default_index=None):
        """
        将枚举值转换为one-hot编码向量
        
        参数:
            value: 需要编码的枚举值
            enum_list: 枚举值列表，定义了编码的顺序
            default_index: 当value不在enum_list中时使用的默认索引，如果为None则抛出异常
        
        返回:
            list: one-hot编码后的向量，长度与enum_list相同
        """
        # 创建初始化为零的向量
        one_hot_vector = [0] * len(enum_list)
        
        # 检查值是否在枚举列表中
        if value in enum_list:
            idx = enum_list.index(value)
            one_hot_vector[idx] = 1
        elif default_index is not None:
            # 如果提供了默认索引，使用默认索引
            if 0 <= default_index < len(enum_list):
                one_hot_vector[default_index] = 1
            else:
                raise ValueError(f"默认索引 {default_index} 超出范围 [0, {len(enum_list)-1}]")
        else:
            # 如果没有提供默认索引且值不在列表中，抛出异常
            raise ValueError(f"值 {value} 不在枚举列表中且未提供默认索引")
        
        return one_hot_vector

    def encode_atom(self, atom: Chem.Atom) -> np.ndarray:
        """
        将原子编码为特征向量
        
        参数:
            atom: RDKit原子对象
            
        返回:
            np.ndarray: 原子特征向量
        """
        features = []
        
        # 原子序数（one-hot编码）
        atomic_num_features = self.one_hot_encode(
            value=atom.GetAtomicNum(),
            enum_list=self.atomic_numbers,
        )
        features.extend(atomic_num_features)
        
        # 原子属性
        features.append(float(atom.GetDegree()))  # 连接度
        features.append(float(atom.GetFormalCharge()))  # 形式电荷
        features.append(float(atom.GetTotalNumHs()))  # 总氢原子数
        features.append(float(atom.GetTotalValence()))  # 总价电子数
        features.append(float(atom.GetNumRadicalElectrons())) # 自由基电子数
        features.append(1.0 if atom.GetIsAromatic() else 0.0)  # 芳香性
        features.append(1.0 if atom.IsInRing() else 0.0)  # 是否在环中
        
        # 杂化类型（one-hot编码）
        hybrid_features = self.one_hot_encode(
            value=atom.GetHybridization(),
            enum_list=self.hybridizations
        )
        # 原子手性（one-hot编码）
        chirality_features = self.one_hot_encode(
            value=atom.GetChiralTag(),
            enum_list=self.chirality_tags
        )
        features.extend(hybrid_features)
        
        return np.array(features, dtype=np.float32)

    def encode_bond(self, bond: Chem.Bond) -> np.ndarray:
        """
        将键编码为特征向量
        
        参数:
            bond: RDKit键对象
            
        返回:
            np.ndarray: 键特征向量
        """
        features = []
        
        # 键类型（one-hot编码）
        bond_type_features = self.one_hot_encode(
            value=bond.GetBondType(),
            enum_list=self.bond_types
        )
        features.extend(bond_type_features)
        
        # 键属性
        features.append(1.0 if bond.GetIsAromatic() else 0.0)  # 芳香键
        features.append(1.0 if bond.IsInRing() else 0.0)  # 环键
        features.append(1.0 if bond.GetIsConjugated() else 0.0)  # 共轭键
        
        return np.array(features, dtype=np.float32)
    
    def get_atom_features(self, mol: Chem.Mol) -> np.ndarray:
        """
        获取分子中所有原子的特征向量

        参数:
            mol: RDKit分子对象

        返回:
            np.ndarray: 原子特征矩阵，形状为 (num_atoms, num_features)
        """
        atom_features = [self.encode_atom(atom) for atom in mol.GetAtoms()]
        return np.stack(atom_features, axis=0)

    def get_bond_features(self, mol: Chem.Mol) -> np.ndarray:
        """
        获取分子中所有键的特征向量

        参数:
            mol: RDKit分子对象

        返回:
            np.ndarray: 键特征矩阵，形状为 (num_bonds, num_features)
        """
        bond_features = [self.encode_bond(bond) for bond in mol.GetBonds()]
        return np.stack(bond_features, axis=0)
    
    def get_atom_features_dim(self) -> int:
        """
        获取原子特征向量的维度

        返回:
            int: 原子特征向量的维度
        """
        atom_features = self.get_atom_features(
            Chem.MolFromSmiles('CC')
        )
        return len(atom_features[0])
    
    def get_bond_features_dim(self) -> int:
        """
        获取键特征向量的维度

        返回:
            int: 键特征向量的维度
        """
        bond_features = self.get_bond_features(
            Chem.MolFromSmiles('CC')
        )
        return len(bond_features[0])
    

class MoleculeFilter:
    """
    一个用于过滤非法分子的类，提供多种过滤方法来确保分子的合法性和可处理性。
    
    这个类可以用于：
    1. 检查分子是否包含非法原子
    2. 检查分子是否包含非法键
    3. 检查分子是否包含特定官能团
    """
    def __init__(
        self,
        allowed_atoms: Optional[List[int]] = None,
        forbidden_fragments: Optional[List[str]] = None,
        min_heavy_atoms: int = 1,
        max_heavy_atoms: int = 100,
        min_mol_weight: float = 10.0,
        max_mol_weight: float = 1000.0,
        custom_filters: Optional[List[Callable]] = None
    ):
        """
        初始化分子过滤器。
        
        参数:
            allowed_atoms: 允许的原子序数列表
            forbidden_fragments: 禁止的官能团SMILES列表
            min_heavy_atoms: 最小重原子数
            max_heavy_atoms: 最大重原子数
            min_mol_weight: 最小分子量
            max_mol_weight: 最大分子量
            custom_filters: 自定义过滤函数列表，每个函数接受一个RDKit分子对象并返回布尔值
        """
        # 默认允许的原子（H, B, C, N, O, F, Si, P, S, Cl, Br）
        self.allowed_atoms = allowed_atoms or [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35]
        
        # 默认禁止的官能团
        self.forbidden_fragments = forbidden_fragments or [
            '[N+](=O)[O-]',      # 硝基
            '[S](=O)(=O)O',      # 硫酸
            '[P](=O)(O)O',       # 磷酸
            '[N+;!-]'            # 季铵盐
        ]
        
        # 原子和分子量限制
        self.min_heavy_atoms = min_heavy_atoms
        self.max_heavy_atoms = max_heavy_atoms
        self.min_mol_weight = min_mol_weight
        self.max_mol_weight = max_mol_weight
        
        # 自定义过滤器
        self.custom_filters = custom_filters or []
    
    def is_valid_molecule(self, mol: Union[str, Chem.Mol]) -> bool:
        """
        检查分子是否合法。
        
        参数:
            mol: 分子对象或SMILES字符串
            
        返回:
            bool: 如果分子合法返回True，否则返回False
        """
        # 如果输入是SMILES字符串，转换为分子对象
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                return False
        
        # 应用所有过滤规则
        filters = [
            self._check_atoms,
            self._check_heavy_atoms,
            self._check_molecular_weight,
            self._check_forbidden_fragments
        ] + self.custom_filters
        
        # 应用所有过滤器
        for filter_func in filters:
            if not filter_func(mol):
                return False
        
        return True
    
    def filter_molecules(
        self, 
        molecules: List[Union[str, Chem.Mol]], 
        return_indices: bool = False
    ) -> Union[List[Union[str, Chem.Mol]], List[int]]:
        """
        过滤分子列表，返回合法分子或其索引。
        
        参数:
            molecules: 分子对象或SMILES字符串列表
            return_indices: 如果为True，返回合法分子的索引；否则返回合法分子
            
        返回:
            合法分子列表或其索引
        """
        valid_indices = []
        
        for i, mol in enumerate(molecules):
            if self.is_valid_molecule(mol):
                valid_indices.append(i)
        
        if return_indices:
            return valid_indices
        else:
            return [molecules[i] for i in valid_indices]
    
    def _check_atoms(self, mol: Chem.Mol) -> bool:
        """检查分子是否只包含允许的原子。"""
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in self.allowed_atoms:
                return False
        return True
    
    def _check_heavy_atoms(self, mol: Chem.Mol) -> bool:
        """检查分子的重原子数是否在允许范围内。"""
        num_heavy = Descriptors.HeavyAtomCount(mol)
        return self.min_heavy_atoms <= num_heavy <= self.max_heavy_atoms
    
    def _check_molecular_weight(self, mol: Chem.Mol) -> bool:
        """检查分子量是否在允许范围内。"""
        mol_weight = Descriptors.MolWt(mol)
        return self.min_mol_weight <= mol_weight <= self.max_mol_weight
    
    def _check_forbidden_fragments(self, mol: Chem.Mol) -> bool:
        """检查分子是否包含禁止的官能团。"""
        for fragment_smarts in self.forbidden_fragments:
            pattern = Chem.MolFromSmarts(fragment_smarts)
            if pattern is not None and mol.HasSubstructMatch(pattern):
                return False
        return True
    
    def add_custom_filter(self, filter_func: Callable[[Chem.Mol], bool]):
        """
        添加自定义过滤函数。
        
        参数:
            filter_func: 接受一个RDKit分子对象并返回布尔值的函数
        """
        self.custom_filters.append(filter_func)
    
    def remove_custom_filter(self, filter_func: Callable[[Chem.Mol], bool]):
        """
        移除自定义过滤函数。
        
        参数:
            filter_func: 要移除的过滤函数
        """
        if filter_func in self.custom_filters:
            self.custom_filters.remove(filter_func)
    
    def get_filter_stats(self, molecules: List[Union[str, Chem.Mol]]) -> Dict[str, Any]:
        """
        获取过滤统计信息。
        
        参数:
            molecules: 分子对象或SMILES字符串列表
            
        返回:
            包含过滤统计信息的字典
        """
        stats = {
            'total': len(molecules),
            'valid': 0,
            'invalid': 0,
            'invalid_reasons': {
                'atoms': 0,
                'heavy_atoms': 0,
                'molecular_weight': 0,
                'forbidden_fragments': 0,
                'custom': 0
            }
        }
        
        for mol in molecules:
            # 如果输入是SMILES字符串，转换为分子对象
            if isinstance(mol, str):
                mol_obj = Chem.MolFromSmiles(mol)
                if mol_obj is None:
                    stats['invalid'] += 1
                    continue
            else:
                mol_obj = mol
            
            
            failed_reasons = []

            if not self._check_atoms(mol_obj):
                failed_reasons.append('atoms')
            if not self._check_heavy_atoms(mol_obj):
                failed_reasons.append('heavy_atoms')
            if not self._check_molecular_weight(mol_obj):
                failed_reasons.append('molecular_weight')
            if not self._check_forbidden_fragments(mol_obj):
                failed_reasons.append('forbidden_fragments')
            
            custom_failed = False

            # 检查自定义过滤器
            for filter_func in self.custom_filters:
                if not filter_func(mol_obj):
                    custom_failed = True
                    break
            if custom_failed:
                failed_reasons.append('custom')            
            if not failed_reasons:
                stats['valid'] += 1
            else:
                stats['invalid'] += 1
                for reason in failed_reasons:
                    stats['invalid_reasons'][reason] += 1

        return stats
    

def generate_3d_conformation(
        mol: Chem.Mol, num_confs: int = 10
    ) -> Tuple[Optional[np.ndarray], Optional[float], Chem.Mol]:
    """
    生成分子的多个构象并进行优化

    Parameters
    ----------
    mol : Chem.Mol
        分子对象
    num_confs : int, optional
        生成的构象数量, by default 10

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[float], Chem.Mol]
        优化后的构象坐标, 最低能量, 分子对象
    """
    try:
        # 复制分子避免修改原分子
        mol_copy = Chem.Mol(mol)

        # 分子清理
        Chem.SanitizeMol(mol_copy)

        # 添加氢原子（用于构象生成）
        mol_with_h = Chem.AddHs(mol_copy, addCoords=True)

        # 设置构象生成参数
        params = Chem.rdDistGeom.ETKDGv3()
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        params.pruneRmsThresh = 0.001 # 适度的RMS阈值
        params.numThreads = 0 # 自动选择线程数
        params.enforceChirality = True
        params.maxAttempts = 10000
        params.randomSeed = 42  # 固定随机种子以确保可重现性

        # 生成多个7构象
        em = Chem.rdDistGeom.EmbedMultipleConfs(
            mol_with_h, numConfs=num_confs * 2, params=params
        )
        
        ps = AllChem.MMFFGetMoleculeProperties(
            mol_with_h, mmffVariant='MMFF94'
        )

        energies = []
        for conf in mol_with_h.GetConformers():  
            # 创建力场对象
            ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
                mol_with_h, ps, confId=conf.GetId()
            )
            
            if isinstance(ff, type(None)):
                continue

            # 计算能量
            energy = ff.CalcEnergy()
            energies.append(energy)

        # 删除氢原子
        mol_no_h = Chem.RemoveHs(mol_with_h)
        if em == -1:
            conformers = []
            for i, c in enumerate(mol_no_h.GetConformers()):
                pos = c.GetPositions()
                conformers.append(pos)
                if i > 9:
                    return conformers
        
        energies = np.array(energies)
        idx = energies.argsort()[:num_confs]
        energies = energies[idx]
        conformers = []
        for i, c in enumerate(mol_no_h.GetConformers()):
            if i not in idx:
                continue
            pos = c.GetPositions()
            conformers.append(pos)

        # 获取最低能量构象
        min_energy_idx = np.argmin(energies)
        min_energy_conformer = conformers[min_energy_idx]

        # 返回最低构象，以及对应的能量，还有 mol_no_h
        return min_energy_conformer, energies[min_energy_idx], mol_no_h

    except Exception as e:
        print(f"生成3D构象时出错: {str(e)}")
        return None, None, mol if 'mol' in locals() else Chem.Mol()
    

if __name__ == '__main__':
    # 创建一个分子
    mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')

    # 生成3D构象
    conformer, energy, mol = generate_3d_conformation(mol)

    # 打印最低能量
    print(f"最低能量: {energy}")
    print(f"最低能量构象: {conformer.shape}")

    # 编码分子
    mol_encoder = MoleculeEncoder()
    atom_feat = mol_encoder.get_atom_features(mol)
    print(mol_encoder.get_atom_features_dim())
    print(mol_encoder.get_bond_features_dim())