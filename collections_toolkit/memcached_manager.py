from dataset_toolkit.compress_utils import get_zip_ROI_AU
import_lib = True
try:
    import pylibmc as mc
except ImportError:
    import memcache as mc
    import_lib = False

class PyLibmcManager(object):
    def __init__(self, host):
        self.AU_couple = get_zip_ROI_AU()
        if import_lib:
            self.mc = mc.Client([host], binary=True, behaviors={'tcp_nodelay':True, "ketama":True})
        else:
            self.mc = mc.Client([host], debug=0)

    def __contains__(self, key):
        if import_lib:
            return key in self.mc
        else:
            return self.mc.get(key) is not None

    def set(self, key, value):
        self.mc.set(key , value)

    def get(self, key):
        return self.mc.get(key)

    def set_crop_mask(self, orig_img_path, new_face, AU_mask_dict):
        self.mc.set(orig_img_path, new_face)
        already_AU_couple = set()
        for AU, mask in AU_mask_dict.items():
            if self.AU_couple[AU] not in already_AU_couple:
                already_AU_couple.add(self.AU_couple[AU])
                self.mc.set("{0}_{1}".format(orig_img_path, ",".join(self.AU_couple[AU])), mask)

    def get_crop_mask(self, orig_img_path):
        cropped_face = self.mc.get(orig_img_path)
        AU_mask_dict = dict()
        for AU_couple in self.AU_couple.values():
            for AU in AU_couple:
                AU_mask_dict[str(AU)] = self.mc.get("{0}_{1}".format(orig_img_path, ",".join(self.AU_couple[AU])))
        return (cropped_face, AU_mask_dict)


