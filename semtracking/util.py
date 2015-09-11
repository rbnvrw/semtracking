import os


def gen_tif_paths(filepath):
    for subdir, dirs, files in os.walk(filepath):
        for filepath in files:
            filename, extension = os.path.splitext(filepath)
            if extension == '.tif':
                yield subdir, filename
