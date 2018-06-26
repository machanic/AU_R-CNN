from collections import defaultdict

def read_file(file_path):
    subject_line = defaultdict(list)
    with open(file_path, "r") as file_obj:
        for line in file_obj:
            line = line.strip()
            img_path, AU_set_str, from_path, database = line.split()
            subject_name = img_path.split("/")[0]
            subject_line[subject_name].append(line)
    return subject_line


if __name__ == "__main__":
    train_set = set(["F001", "F003", "F005", "F007", "F009", "F011", "F013", "F015", "F017",
                     "F019", "F021", "F023", "M001", "M003", "M005", "M007", "M009", "M011", "M013", "M015", "M017"])
    val_set = set(["F002","F004","F006","F008","F010","F012","F014","F016","F018","F020","F022","M002","M004","M006",
                   "M008","M010","M012","M014","M016","M018"])

    d1 = read_file("/home/machen/dataset/BP4D/idx/3_fold/id_trainval_1.txt")
    d2 = read_file("/home/machen/dataset/BP4D/idx/3_fold/id_test_1.txt")
    d1.update(d2)
    train_lines = []
    for train_sub in train_set:
        train_lines.extend(d1[train_sub])
    test_lines = []
    for test_sub in val_set:
        test_lines.extend(d1[test_sub])

    train_filename = "/home/machen/dataset/BP4D/idx/FERA15_train.txt"
    with open(train_filename, "w") as file_obj:
        for line in train_lines:
            file_obj.write("{}\n".format(line))
        file_obj.flush()

    test_filename = "/home/machen/dataset/BP4D/idx/FERA15_validate.txt"
    with open(test_filename, "w") as file_obj:
        for line in test_lines:
            file_obj.write("{}\n".format(line))
        file_obj.flush()



