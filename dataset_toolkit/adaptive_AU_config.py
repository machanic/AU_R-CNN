import config
from bidict import bidict
from collections import OrderedDict
from dataset_toolkit.compress_utils import get_zip_ROI_AU
import itertools


# 后两个参数应该配合使用,用于添加CRF单label妥协方案用到的新增的组合label（视为一个新label）
def adaptive_AU_database(database_name, use_paper_only=False):

    if database_name == "BP4D":
        if use_paper_only:
            database_use_AU = config.paper_use_BP4D
        else:
            database_use_AU = config.BP4D_use_AU
    elif database_name == "DISFA":
        if use_paper_only:
            database_use_AU = config.paper_use_DISFA
        else:
            database_use_AU = config.DISFA_use_AU
    elif database_name == "BP4D_DISFA":
        database_use_AU = config.DISFA_use_AU # BP4D_DISFA 仍然按照DISFA来训练
    new_AU_ROI = OrderedDict()
    for AU in config.AU_ROI.keys():
        if AU in database_use_AU:
            new_AU_ROI[AU] = config.AU_ROI[AU]
    orig_AU_ROI = config.AU_ROI
    orig_AU_SQUEEZE = config.AU_SQUEEZE
    config.AU_ROI = new_AU_ROI
    config.AU_SQUEEZE = bidict({idx: str(AU) for idx, AU in enumerate(sorted(map(int, list(config.AU_ROI.keys()))))})
    NEW_LABEL_INCORPORATE = {}
    for key,val_ls in config.LABEL_INCORPORATE.items():
        new_key = []
        for k in key:
            if k in config.AU_ROI:
                new_key.append(k)
        new_val_ls = []
        for val in val_ls:
            new_val = []
            for v in val:
                if v in config.AU_ROI:
                    new_val.append(v)
            if new_val:
                new_val_ls.append(tuple(new_val))

        if len(new_key) > 0 and len(new_val_ls) > 0:
            NEW_LABEL_INCORPORATE[tuple(new_key)] = new_val_ls
    config.LABEL_INCORPORATE = NEW_LABEL_INCORPORATE
    NEW_BOX_SHIFT = {}
    for key ,val in config.BOX_SHIFT.items():
        new_key = []
        for k in key:
            if k in config.AU_ROI:
                new_key.append(k)
        if len(new_key) > 0:
            NEW_BOX_SHIFT[tuple(new_key)] = val
    config.BOX_SHIFT = NEW_BOX_SHIFT

    return {"orig_AU_ROI":orig_AU_ROI, "orig_AU_SQUEEZE":orig_AU_SQUEEZE}

def adaptive_AU_relation(database_name):
    '''
    must called after adaptive_AU_database
    从config.AU_RELATION_BP4D中删去同一个区域的AU

    比如说config.AU_RELATION_BP4D 中有(10, 12) 但10和12已经是同一个区域了，因此删掉这种关系。

    另外需要注意另一种情况：再比如说config.AU_RELATION_BP4D 中有(10, 13)，但10和12再字典中组合成新的区域了，而13也和另一个AU组合成了新的区域。

    '''
    new_AU_relation = list()


    AU_couple = get_zip_ROI_AU()
    already_same_region_set = set()
    for AU, couple_tuple in AU_couple.items():
        for AU_a, AU_b in itertools.combinations(couple_tuple, 2):
            already_same_region_set.add(tuple(sorted([int(AU_a), int(AU_b)])))
    if database_name == "BP4D":
        for AU_tuple in config.AU_RELATION_BP4D:
            if tuple(sorted(AU_tuple)) not in already_same_region_set:
                new_AU_relation.append(tuple(sorted(AU_tuple)))
        config.AU_RELATION_BP4D = new_AU_relation
    elif database_name == "DISFA":
        for AU_tuple in config.AU_RELATION_DISFA:
            if tuple(sorted(AU_tuple)) not in already_same_region_set:
                new_AU_relation.append(tuple(sorted(AU_tuple)))
        config.AU_RELATION_DISFA = new_AU_relation



