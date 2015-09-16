import os


def gen_img_paths(directory, filter_ext='.tif', level=1):
    """

    :param filepath:
    """
    directory = directory.rstrip(os.path.sep)
    assert os.path.isdir(directory)
    num_sep = directory.count(os.path.sep)
    for root, dirs, files in os.walk(directory):
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            continue
        for filepath in files:
            filename, extension = os.path.splitext(filepath)
            if extension == filter_ext:
                yield filename
