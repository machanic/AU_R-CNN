from collections import defaultdict


def AU_repeat_level(level_num, AU_count):
    '''

    :param level_num:  how many AU (repeat) level are there in AU class number
            AU_count: dict key=AU, value=count
    :return: AU_level is a dict, which value is level_index, lower level_index means lower repeat level, higher level means higher repeat level
    '''
    split_list = lambda A, n=level_num: [A[i:i + n] for i in range(0, len(A), n)]
    AU_level = dict()
    # print("sublist:{}".format(split_list(sorted(AU_count.items(), key=lambda e:e[1], reverse=True))))
    for idx, sub in enumerate(split_list(sorted(AU_count.items(), key=lambda e:e[1], reverse=True))):
        for AU, count in sub:
            AU_level[AU] = idx
    return AU_level


def load_AU_count(file_path):
    AU_count = defaultdict(int)
    with open(file_path, "r") as file_obj:
        for line in file_obj:
            AU, count = line.strip().split("=")
            AU_count[AU] = count
    return AU_count


if __name__ == "__main__":
    AU_count = load_AU_count("/home/machen/face_expr/resource/AU_occr_count.dict")
    AU_level = AU_repeat_level(5, AU_count)
    for AU, level in AU_level.items():
        print(AU, level)