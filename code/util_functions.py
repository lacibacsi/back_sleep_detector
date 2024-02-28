import os


def load_image_names(path):
    '''
    Loads jpg files found in the input folder. Not recursive
    :param path: input path
    :return: list of file names with jpg extensions with the path
    '''
    jpgs = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.jpg')]
    #jpgs = [file for file in os.listdir(path) if file.endswith('.jpg')]

    return jpgs
