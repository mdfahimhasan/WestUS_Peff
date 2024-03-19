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
        input_file = glob(os.path.join(input_dir_file, search_by))[0]
        if rename is not None:
            copied_file = os.path.join(copy_dir, f'{rename}.tif')
        else:
            file_name = os.path.basename(input_file)
            copied_file = os.path.join(copy_dir, file_name)

        shutil.copyfile(input_file, copied_file)

    else:
        if rename is not None:
            copied_file = os.path.join(copy_dir, f'{rename}.tif')
        else:
            file_name = os.path.basename(input_dir_file)
            copied_file = os.path.join(copy_dir, file_name)

        shutil.copyfile(input_dir_file, copied_file)

    return copied_file


def make_gdal_sys_call(gdal_command, args, verbose=True):
    """
    Make GDAL system call string.
    ** followed by code from Sayantan Majumdar.

    :param gdal_command: GDAL command string formatted as 'gdal_rasterize'.
    :param args: List of GDAL command.
    :param verbose: Set True to print system call info.

    :return: GDAL system call string.
    """
    if os.name == 'nt':
        # update it based on the pc's QGIS version and folderpath
        gdal_path = 'C:/Program Files/QGIS 3.22.7/OSGeo4W.bat'

        sys_call = [gdal_path] + [gdal_command] + args
        if verbose:
            print(sys_call)
        return sys_call

    else:
        print('gdal sys call not optimized for linux yet')
