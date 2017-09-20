import os


def find_filepath(filename):
    """Finds the file path for a single specified file or directory in the current working directory

        Parameters
    ----------
    filename
        string which is the name of the file to be found

    Returns
    -------
        Path like string for specified file"""
    for root, dirs, files in os.walk(os.getcwd(), False):
        for name in files:
            if filename == name :
                return os.path.join(root, name)
        for dir in dirs:
            if filename == dir:
                return os.path.join(root, dir)
    return FileNotFoundError

def find_all_filepaths(filename):
    """Finds file paths in the current working directory which contain the parameter filename as a substring

    Parameters
    ----------
    filename: string
        The name of the file to be found

    Returns
    -------
    ret: list
        List of all strings which contain the parameter filename as a substring"""
    ret = []
    for root, dirs, files in os.walk(os.getcwd(), False):
        for name in files:
            if filename in name :
                ret.append(os.path.join(root, name))
        for dir in dirs:
            if filename in dir:
                ret.append(os.path.join(root, dir))

    if not ret:
        return FileNotFoundError
    else:
        return ret

if __name__ == '__main__':
    print(find_filepath('networks'))
    print(find_filepath('basic_template'))
    print(find_all_filepaths('G_'))

