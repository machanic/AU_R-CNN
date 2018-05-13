import numpy as np
from PIL import Image, ImageEnhance
from functools import lru_cache
from dataset_toolkit.scripts.label_balance import AU_repeat_level
import cv2





# 爲了分類數據平衡，需要造一些數據進去，方法是根據repeat_level去分類
def image_generate(image, AU_labels):
    '''
    image is PIL read_image legacy return image, which is (C, H, W) format
    :param image: (C, H, W) format
    :param repeat_level: ratio is this image minimal class label propotional ratio to whole class_label number

    :return:
    '''
    AU_level = AU_repeat_level(level_num=5)

    repeat_level = max([AU_level[AU] for AU in AU_labels])
    convert_func_lst_1 = [lambda img: img.transpose(Image.FLIP_LEFT_RIGHT), lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)]
    convert_func_lst_2 = [lambda img: ImageEnhance.Sharpness(img).enhance(0.0), lambda img: ImageEnhance.Sharpness(img).enhance(2.0)]
    


@lru_cache(maxsize=128)
def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This legacy reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this legacy returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W) and
        return img.transpose(2, 0, 1)
