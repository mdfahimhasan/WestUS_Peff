import os
import shutil
from glob import glob


def makedirs(directory_list):
    """
    Make directory (if not exists) from a list of directory.

    :param directory_list: A list of directories to create.

    :return: None.
    """
    for directory in directory_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


def copy_file(input_dir_file, copy_dir, search_by='*.tif', rename=None):
    """
    Copy a file to the specified directory.

    :param input_dir_file: File path of input directory/ Path of the file to copy.
    :param copy_dir: File path of copy directory.
    :param search_by: Default set to '*.tif'.
    :param rename: New name of file if required. Default set to None.

    :return: File path of copied file.
    """
    makedirs([copy_dir])
    if '.tif' not in input_dir_file:
        input_file = glob(os.path.join(input_dir_file, search_by))
        if rename is not None:
            copy_file = os.path.join(copy_dir, f'{rename}.tif')
        else:
            file_name = os.path.basename(input_file)
            copy_file = os.path.join(copy_dir, file_name)

        shutil.copyfile(input_file, copy_file)

    else:
        if rename is not None:
            copy_file = os.path.join(copy_dir, f'{rename}.tif')
        else:
            file_name = os.path.basename(input_dir_file)
            copy_file = os.path.join(copy_dir, file_name)

        shutil.copyfile(input_dir_file, copy_file)

    return copy_file
