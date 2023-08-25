import networkx as nx
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
import pickle
from utils import get_gnn_features
import os


threshholds = [3,4,6,7,8,9,10]#距离阈值3, 4, 5, 6, 7, 8, 9, 10
coreset_path = "data/refined/core-set"#核心集文件夹
coreset_names = os.listdir(coreset_path)

for thr in threshholds:
    target_path = "data/mice_features/v2016"#特征路径
    root = 'data/refined/refined-set'#所有数据文件夹
    target_path = os.path.join(target_path, "all_"+str(thr)+"A")#矩阵信息
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    exitsted_datas = os.listdir(target_path)#列举已存在的数据
    exitsted_datas = [esd.split(".")[0] for esd in exitsted_datas]#获取已存在数据的文件名
    n_train = 0
    m_train = 0
    exeptions_all = []
    allDatas = os.listdir(root)#获取所有数据的列表
    for i_all in allDatas:
        if i_all in coreset_names:#判断数据是否存在于核心集，
            if os.path.exists(os.path.join(target_path, i_all+".pkl")):#判断特征矩阵文件夹里是否有对应的特征文件，存在就把对应的特征删除
                os.remove(os.path.join(target_path, i_all+".pkl"))
            continue
        if i_all in exitsted_datas:#判断数据是否在当前的矩阵中已经存在
            continue
        else:
            n_train += 1 #数据量+1
            pdb_path_all = os.path.join(root, i_all, i_all+'_pocket.pdb')#读取蛋白质文件
            mol2_path_all = os.path.join(root, i_all, i_all+'_ligand.mol2')#读取配体文件
            #如果protein.pdb读不出来就读pocket,mol2读不出来就换sdf
            protein = Chem.MolFromPDBFile(pdb_path_all)
            if protein is None:
                protein = Chem.MolFromPDBFile(os.path.join(root, i_all, i_all+"_pocket.pdb")) #protein文件读不出来就用pocket
            ligand = Chem.MolFromMol2File(mol2_path_all)
            if ligand is None:
                suppl = Chem.SDMolSupplier(os.path.join(root, i_all, i_all+'_ligand.sdf')) #mol2读不出来就用sdf
                mols = [mol for mol in suppl if mol] #获取sdf里所有的分子
                if len(mols)>0:
                    ligand = mols[0]
            if protein is None or ligand is None:
                m_train += 1 #miss数据+1
                print("protein or ligand is none")
                continue

            try:

                x_all, edge_index_inner, edge_index_out, edge_attr_inner,edge_attr_outer = get_gnn_features(protein, ligand, threshhold=thr)#生成特征
                # label_train = dic_train[i_train]
                if x_all is None or edge_index_inner is None or edge_index_out is None or edge_attr_inner is None or edge_attr_outer is None:
                    m_train += 1
                    print("feature is none")
                    continue
                with open(os.path.join(target_path, i_all+'.pkl'), 'wb') as f_train2: #序列化特征对象
                    pickle.dump((x_all, edge_index_inner, edge_index_out,edge_attr_inner,edge_attr_outer), f_train2)
            except Exception as e:
                # print(e)
                print("exception")
                m_train += 1
                exeptions_all.append(i_all)
            print('all data is ', m_train, '/', n_train)

    print(exeptions_all)
