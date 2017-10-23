from collections import defaultdict
import itertools
import config

old_AU_couple_dict = None

def run_once(f):
    def wrapper(*args, **kwargs):
        global old_AU_couple_dict
        if not wrapper.has_run:
            wrapper.has_run = True
            old_AU_couple_dict = f(*args, **kwargs)
            return old_AU_couple_dict
        else:
            return old_AU_couple_dict
    wrapper.has_run = False
    return wrapper



def get_zip_ROI_AU():
    regionlst_AU_dict = defaultdict(list)
    AU_couple_dict = {} # AU => couple_name
    for AU, region_lst in config.AU_ROI.items():
        region_tuple = tuple(sorted(region_lst))
        regionlst_AU_dict[region_tuple].append(AU)
    for au_lst in regionlst_AU_dict.values():
        for AU in au_lst:
            AU_couple_dict[AU] = tuple(map(str, sorted(map(int,au_lst))))

    return AU_couple_dict # AU -> (AU tuple with same region)

def get_AU_couple_child(AU_couple_dict):
    # must be called after def adaptive_AU_database
    AU_couple_child = defaultdict(set) # may have multiple child regions
    for AU_region_a, AU_region_b in itertools.combinations(config.AU_ROI.items(),2):
        AU_a, region_lst_a = AU_region_a
        AU_b, region_lst_b = AU_region_b

        region_set_a = set(region_lst_a)
        region_set_b = set(region_lst_b)
        contains_a = region_set_a.issubset(region_set_b)
        if contains_a and len(region_set_a) < len(region_set_b):
            AU_couple_child[AU_couple_dict[AU_b]].add(AU_couple_dict[AU_a])
        contains_b = region_set_b.issubset(region_set_a)
        if contains_b and len(region_set_a) > len(region_set_b):
            AU_couple_child[AU_couple_dict[AU_a]].add(AU_couple_dict[AU_b])
    # 增加一个逻辑，修正label更好 must be called after def adaptive_AU_database
    for AU_couple, AU_couple_incorporate_lst in config.LABEL_INCORPORATE.items():
        AU_couple_child[AU_couple].update(AU_couple_incorporate_lst)
    return AU_couple_child


if __name__ == "__main__":
    from dataset_toolkit.adaptive_AU_config import adaptive_AU_relation, adaptive_AU_database
    import config

    adaptive_AU_database("BP4D")
    adaptive_AU_relation()
    print(config.AU_SQUEEZE)
    AU_couple = get_zip_ROI_AU()
    print(list(AU_couple.values()))
