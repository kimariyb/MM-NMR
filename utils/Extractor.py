import ast
import numpy as np


def ExtractCarbonShift(mol):
    r"""
    提取 13C 化学位移信息
    """
    # 获取分子的属性字典
    mol_props = mol.GetPropsAsDict()
    atom_shifts = {}
    
    # 遍历所有属性键
    for key in mol_props.keys():
        # 找到以 'Spectrum 13C' 开头的属性
        if key.startswith('Spectrum 13C'):
            # 分割属性值，获取每个化学位移信息
            for shift in mol_props[key].split('|')[:-1]:
                # 分割化学位移值、未知字段和原子索引
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)
                # 如果原子索引不在字典中，初始化为空列表
                if shift_idx not in atom_shifts: 
                    atom_shifts[shift_idx] = []
                    
                # 将化学位移值添加到对应原子索引的列表中
                atom_shifts[shift_idx].append(shift_val)

    # 对每个原子索引，计算化学位移值的中位数
    for j in range(mol.GetNumAtoms()):
        if j in atom_shifts:
            atom_shifts[j] = np.median(atom_shifts[j])

    return atom_shifts


def ExtractHydrogenShift(mol):
    r"""
    提取 1H 化学位移信息
    """
    # 获取分子的属性字典
    mol_props = mol.GetPropsAsDict()
    atom_shifts = {}
    
    # 遍历所有属性键
    for key in mol_props.keys():
        # 找到以 'Spectrum 1H' 开头的属性
        if key.startswith('Spectrum 1H'):
            tmp_dict = {}
            # 分割属性值，获取每个化学位移信息
            for shift in mol_props[key].split('|')[:-1]:
                # 分割化学位移值、未知字段和原子索引
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)
                # 如果原子索引不在字典中，初始化为空列表
                if shift_idx not in atom_shifts: 
                    atom_shifts[shift_idx] = []
                # 如果原子索引不在临时字典中，初始化为空列表
                if shift_idx not in tmp_dict: 
                    tmp_dict[shift_idx] = []
                tmp_dict[shift_idx].append(shift_val)
                
            # 将临时字典中的值添加到原子位移字典中
            for shift_idx in tmp_dict.keys():
                atom_shifts[shift_idx].append(tmp_dict[shift_idx])
                
    # 对每个原子索引，处理化学位移值
    for shift_idx in atom_shifts.keys():
        # 找到每个原子索引的最大位移列表长度
        max_len = np.max([len(shifts) for shifts in atom_shifts[shift_idx]])
        
        for i in range(len(atom_shifts[shift_idx])):
            # 如果位移列表长度小于最大长度，进行填充
            if len(atom_shifts[shift_idx][i]) < max_len:
                # 如果列表长度为1，重复该值填充到最大长度
                if len(atom_shifts[shift_idx][i]) == 1:
                    atom_shifts[shift_idx][i] = [atom_shifts[shift_idx][i][0] for _ in range(max_len)]
                # 如果列表长度大于1，使用均值填充到最大长度
                elif len(atom_shifts[shift_idx][i]) > 1:
                    while len(atom_shifts[shift_idx][i]) < max_len:
                        atom_shifts[shift_idx][i].append(np.mean(atom_shifts[shift_idx][i]))

            # 对位移列表进行排序
            atom_shifts[shift_idx][i] = sorted(atom_shifts[shift_idx][i])
        # 计算每个原子索引的位移值的中位数
        atom_shifts[shift_idx] = np.median(atom_shifts[shift_idx], 0).tolist()
    
    return atom_shifts


def ExtractFluorineShift(mol):
    r"""
    提取 19F 化学位移信息
    """
    # 获取分子的属性字典
    mol_props = mol.GetPropsAsDict()
    atom_shifts = {}
    
    # 遍历所有属性键
    for key in mol_props.keys():
        # 找到以 'Spectrum 19F' 开头的属性
        if key.startswith('Spectrum 19F'):
            # 分割属性值，获取每个化学位移信息
            for shift in mol_props[key].split('|')[:-1]:
                # 分割化学位移值、未知字段和原子索引
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)
                # 如果原子索引不在字典中，初始化为空列表
                if shift_idx not in atom_shifts: 
                    atom_shifts[shift_idx] = []
                    
                # 将化学位移值添加到对应原子索引的列表中
                atom_shifts[shift_idx].append(shift_val)
                
    # 对每个原子索引，计算化学位移值的中位数
    for j in range(mol.GetNumAtoms()):
        if j in atom_shifts:
            atom_shifts[j] = np.median(atom_shifts[j])
    
    return atom_shifts


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