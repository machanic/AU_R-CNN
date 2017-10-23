class LabelTypeEnum(object):
    KNOWN_LABEL = 0
    UNKNOWN_LABEL = 1


class DiffMax(object):

    def __init__(self,diff_max:float):
        self.diff_max = diff_max