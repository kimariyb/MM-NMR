from typing_extensions import LiteralString


def split_smiles(sm) -> LiteralString:
    """Split SMILES into tokens with special handling for multi-character elements"""
    multi_chars = {
        # 双字符元素
        'Cl', 'Br', 'Si', 'Se', 'Na', 'Cu', 'Ca', 'Be', 'Ba', 'Bi',
        'Sr', 'Ni', 'Rb', 'Ra', 'Xe', 'Li', 'Al', 'As', 'Ag', 'Au',
        'Mg', 'Mn', 'Te', 'Zn', 'Kr', 'Fe', 'He',
        # 电荷表示
        '+2', '+3', '+4', '-2', '-3', '-4',
        # 小写元素（如有需要）
        'si', 'se', 'te'
    }
    
    arr = []
    i = 0
    while i < len(sm):
        # 处理百分号开头的三位数字（如%99）
        if sm[i] == '%' and i + 2 < len(sm):
            arr.append(sm[i:i+3])
            i += 3
        # 处理双字符元素
        elif i + 1 < len(sm) and sm[i:i+2] in multi_chars:
            arr.append(sm[i:i+2])
            i += 2
        # 单字符处理
        else:
            arr.append(sm[i])
            i += 1
            
    return ' '.join(arr)



