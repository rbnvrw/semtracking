from os import path, walk


def gen_img_paths(directory, filter_ext='.tif', level=1):
    """
    Generates image paths in directory (no recursion)
    :param filepath:
    """
    directory = directory.rstrip(path.sep)
    assert path.isdir(directory)
    num_sep = directory.count(path.sep)
    for root, dirs, files in walk(directory):
        num_sep_this = root.count(path.sep)
        if num_sep + level <= num_sep_this:
            continue
        for file_path in files:
            filename, extension = path.splitext(file_path)
            if extension == filter_ext:
                yield filename
