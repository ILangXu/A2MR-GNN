from rdkit import Chem
import torch
import numpy as np
from rdkit import RDLogger
import random
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def set_data_device(examples, device):
    if type(examples) == list or type(examples) == tuple:
        return [set_data_device(e, device) for e in examples]
    else:
        return examples.to(device)
def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
     #random.random()也是一个生成随机数的函数，但是每次调用后生成的随机数都是不同的
    # random.seed(x) 设置好参数（种子，即x）后每次调用后生成的结果都是一样的
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)#为特定GPU设置种子，生成随机数
    torch.cuda.manual_seed_all(seed)#为所有GPU设置种子，生成随机数
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def get_atom_features(atom, is_protein=False):
    ATOM_CODES = {}
    metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
              + list(range(37, 51)) + list(range(55, 84))
              + list(range(87, 104)))
    atom_classes = [(5, 'B'), (6, 'C'), (7, 'N'), (8, 'O'), (15, 'P'), (16, 'S'), (34, 'Se'),
                    ([9, 17, 35, 53], 'halogen'), (metals, 'metal')]
    for code, (atomidx, name) in enumerate(atom_classes): #code 是index，atomidx是原子序数，name是元素名称
        if type(atomidx) is list:
            for a in atomidx:#若序数是一个列表，即halogen或metal
                ATOM_CODES[a] = code
        else:
            ATOM_CODES[atomidx] = code
    try:
        classes = ATOM_CODES[atom.GetAtomicNum()]#根据当前原子的原子序数获取当前原子在种类列表中的索引位置
    except:
        classes = 9#如果原子序数不在其列，就设置为9

    possible_chirality_list = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ]
    chirality = possible_chirality_list.index(atom.GetChiralTag())#转化成索引数字

    possible_formal_charge_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    try:
        charge = possible_formal_charge_list.index(atom.GetFormalCharge())#转化成索引数字
    except:
        charge = 11

    possible_hybridization_list = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ]
    try:
        hyb = possible_hybridization_list.index(atom.GetHybridization())#杂化方式
    except:
        hyb = 6

    possible_numH_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    try:
        numH = possible_numH_list.index(atom.GetTotalNumHs()) #连接氢原子个数
    except:
        numH = 9

    possible_implicit_valence_list = [0, 1, 2, 3, 4, 5, 6, 7]
    try:
        valence = possible_implicit_valence_list.index(atom.GetTotalValence())#返回原子化合价
    except:
        valence = 8

    possible_degree_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    try:
        degree = possible_degree_list.index(atom.GetTotalDegree())#原子的度
    except:
        degree = 11

    is_aromatic = [False, True]
    aromatic = is_aromatic.index(atom.GetIsAromatic())#芳香的，芬芳的；芳香族的

    is_protein = int(is_protein)#是否蛋白质

    mass = atom.GetMass() / 100 #单个原子的质量

    return [classes, chirality, charge, hyb, numH, valence, degree, aromatic, is_protein, mass]


def get_bonds_features(bond, is_protein=False):
    possible_bonds_type = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                           Chem.rdchem.BondType.AROMATIC, Chem.rdchem.BondType.UNSPECIFIED, Chem.rdchem.BondType.ZERO,
                           Chem.rdchem.BondType.OTHER]
    try:
        bond_type = possible_bonds_type.index(bond.GetBondType())#化学键类型
    except:
        bond_type = 6

    possible_bond_dirs = [Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT, Chem.rdchem.BondDir.ENDDOWNRIGHT,
                          Chem.rdchem.BondDir.EITHERDOUBLE, Chem.rdchem.BondDir.UNKNOWN]
    try:
        bond_dirs = possible_bond_dirs.index(bond.GetBondDir()) #化学键的方向
    except:
        bond_dirs = 4

    stereo = int(bond.GetStereo())

    is_ring = int(bond.IsInRing())#判断是否在环中

    is_protein = int(is_protein)

    return [bond_type, bond_dirs, stereo, is_ring, is_protein]

def get_gnn_features(protein, ligand, threshhold=5):

    ligand_conf = ligand.GetConformer()
    ligand_positions = ligand_conf.GetPositions()
    protein_conf = protein.GetConformer()
    protein_positions = protein_conf.GetPositions()
    '''
    获取配体和蛋白质原子间的距离
    '''
    dis = ligand_positions[:, np.newaxis, :] - protein_positions[np.newaxis, :, :]#np.newaxis 放在哪个位置，就会给哪个位置增加维度
    dis = np.sqrt((dis * dis).sum(-1)) #距离矩阵，行下标是配体索引，列下标是蛋白质索引
    idx = np.where(dis < threshhold)#返回行标元组和列标元组
    idx = [[i, j] for i, j in zip(idx[0], idx[1])]#返回一个装着行列下标的list

    innerdis = ligand_positions[:, np.newaxis, :] - ligand_positions[np.newaxis, :, :]#计算配体内部的距离
    innerdis = np.sqrt((innerdis * innerdis).sum(-1))#配体内部距离矩阵

    atom_features_list = []#原子特征列表
    pidx = [i[1] for i in idx]#获取蛋白质和配体边的列下标，即有连边的蛋白质原子的索引
    nligand_atoms = len(ligand.GetAtoms())
    pidx = sorted(list(set(pidx)))#列下标排序去重，得到的是所有和配体有作用的蛋白质原子的索引值
#在蛋白质的序列中，氨基酸之间的氨基和羧基脱水成键，氨基酸由于其部分基团参与了肽键的形成，剩余的结构部分则称氨基酸残基。
    pidx = [i for i in pidx if protein.GetAtomWithIdx(int(i)).GetPDBResidueInfo().GetResidueName() != 'HOH']#判断当前原子的残基是否为水,把水去掉
    pidx2tidx = {pidx[i]: i + nligand_atoms for i in range(len(pidx))}#每个元素对应索引+配体原子个数
    for i in range(len(ligand.GetAtoms())):
        atom = ligand.GetAtomWithIdx(int(i))
        atom_feature = list(get_atom_features(atom, is_protein=False)) + ligand_positions[i].tolist() #13维
        atom_features_list.append(atom_feature)

    for i in pidx: #只保留与配体原子连接的蛋白质原子的特征
        atom = protein.GetAtomWithIdx(int(i))
        atom_feature = list(get_atom_features(atom, is_protein=True)) + protein_positions[i].tolist()
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float)#[n,13]原子特征矩阵


    edge_features_out = []
    edge_features_inner = []
    edges_out_list=[]
    edges_inner_list=[]

    if len(ligand.GetBonds()) > 0:  # mol has bonds
        edges_inner_list = []
        edge_features_inner = []

        for bond in ligand.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = list(get_bonds_features(bond, is_protein=False)) + [innerdis[i, j]]
            edges_inner_list.append((i, j))#边集合
            edge_features_inner.append(edge_feature)#边特征集合
            edges_inner_list.append((j, i))#边是双向的
            edge_features_inner.append(edge_feature)

    for b in idx:
        try:
            edges_out_list.append((b[0], pidx2tidx[b[1]]))  #相当于是只给配体原子和与配体原子有连接的蛋白质原子构造图和邻接矩阵，没有连接的蛋白质原子被排除在外
            edge_feature = [6, 4, 0, 0, 1] + [dis[b[0], b[1]]]
            edge_features_out.append(edge_feature)
            edges_out_list.append((pidx2tidx[b[1]], b[0]))
        except:
            continue
        edge_features_out.append(edge_feature)

    edge_index_inner = torch.tensor(np.array(edges_inner_list).T, dtype=torch.long)#边都是双向的，边的索引集合
    edge_index_out = torch.tensor(np.array(edges_out_list).T, dtype=torch.long)  # 边都是双向的，边的索引集合

    edge_attr_inner = torch.tensor(np.array(edge_features_inner),#边的特征矩阵
                             dtype=torch.float)
    edge_attr_out = torch.tensor(np.array(edge_features_out),  # 边的特征矩阵
                             dtype=torch.float)

    return x, edge_index_inner, edge_index_out, edge_attr_inner, edge_attr_out #[n,13],[2,m],[m,6]
