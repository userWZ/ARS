from restore import *


if __name__ == '__main__':

    # file = './templates/static/Restoration_Sample_Data.xlsx'
    file = 'IEEE30.xlsx'
    data, path = restore(file)
    print('data', data)
    print('path', path)
