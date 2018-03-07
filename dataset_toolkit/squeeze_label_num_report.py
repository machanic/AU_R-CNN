from collections import OrderedDict
import config

def squeeze_label_num_report(database, use_paper_num_label):
    paper_report_label = OrderedDict()
    if database == "BP4D":
        paper_use_AU = config.paper_use_BP4D
    elif database == "DISFA":
        paper_use_AU = config.paper_use_DISFA
    if use_paper_num_label:
        for AU_idx, AU in sorted(config.AU_SQUEEZE.items(), key=lambda e: int(e[0])):
            if AU in paper_use_AU:
                paper_report_label[AU_idx] = AU
    if not paper_report_label:
        class_num = len(config.AU_SQUEEZE)
    else:
        class_num = len(paper_report_label)
    return paper_report_label, class_num