from collections import OrderedDict


def convert(file_path, out_file_path):
    feature_names = set()
    with open(file_path, 'r') as file_obj:
        for idx, line in enumerate(file_obj):
            tokens = line.rstrip().split()
            if not tokens[0].startswith("#edge"):
                feature_names.update([tok.split(":")[0] for tok in tokens[1:]])

    feature_name_idx = OrderedDict([(name,idx )for idx, name in enumerate(sorted(feature_names))])
    file_out = open(out_file_path, "w")
    with open(file_path, 'r') as file_obj:
        for idx, line in enumerate(file_obj):
            tokens = line.rstrip().split()
            if not tokens[0].startswith("#edge"):
                unknown_label = tokens[0][0] if tokens[0][0] == "?" else ""
                label = tokens[0][1]
                if label == "0":
                    label = "1,0"
                elif label == "1":
                    label = "0,1"
                nodeid = idx+1
                feature_dict = {tok.split(":")[0]: tok.split(":")[1] for tok in tokens[1:]}
                features = []
                for featurename in feature_name_idx.keys():
                    features.append(feature_dict.get(featurename, "0"))
                features = ",".join(features)
                newline = "{0}{1} ({2}) features:{3}\n".format(unknown_label, nodeid, label, features)
                file_out.write(newline)
            else:
                file_out.write(line)
        file_out.flush()
    file_out.close()

convert("C:/Users/machen/Desktop/OpenCRF_Multi_label/OpenCRF_multi_label/example.txt",
        "D:/toy/toy.txt")
