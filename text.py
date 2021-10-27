from restore import *
import pandas as pd

if __name__ == '__main__':

    # file = 'Restoration_Sample_Data.xlsx'
    # file = 'IEEE30.xlsx'
    file = 'IEEE30_2.xlsx'
    data, path = restore(file)
    pd.set_option('display.max_columns', None)
    print('data', data)
    print('path', path)
