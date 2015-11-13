from os import path, walk
import getopt
import sys


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


def get_directory_from_command_line(argv, script_name):
    """

    :param argv:
    :param script_name:
    :return:
    """
    usage = 'Usage: ' + script_name + ' -d <directory>'
    directory = ''
    try:
        opts, args = getopt.getopt(argv, "hd:", ["dir="])
    except getopt.GetoptError:
        print usage
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print usage
            sys.exit()
        elif opt in ("-d", "--dir"):
            directory = arg

    if len(directory) == 0:
        print usage
        sys.exit(2)

    return directory
