import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf


def txt2csv(p_txt,p_csv):
    pd.DataFrame(np.loadtxt(p_txt, dtype='str')).to_csv(p_csv, index=False)
    return


def load_csv(p):
    data = pd.read_csv(p).values
    # values.shape 1303038 * 6
    # length_dga = dga_orginal.shape[0]
    return data


# aggregate same sub families
def family_agg(p, p_out):
    data = load_csv(p)[:, :2]
    for i in range(data.shape[0] - 1):
        if (data[i][0] == 'pykspa_v1') | (data[i][0] == 'pykspa_v2_real') | (
                data[i][0] == 'pykspa_v2_fake'):
            data[i][0] = 'pykspa'
        elif (data[i][0] == 'fobber_v1') | (data[i][0] == 'fobber_v2'):
            data[i][0] = 'fobber'
    np.savetxt(p_out, data, delimiter=',', fmt='%s')
    return


def family_split(p, d_out):
    data = load_csv(p)[:, :]
    family_names = {}
    for i in range(data.shape[0]-1):
        if data[i][0] not in family_names:
            family_names[data[i][0]] = [list(data[i])]
        else:
            family_names[data[i][0]].append(list(data[i]))
    for name in family_names:
        p_out = '{}/{}.csv'.format(d_out, name)
        item = np.array(family_names[name])
        np.savetxt(p_out, item, delimiter=',', fmt='%s')
    return


if __name__ == '__main__':
    p_txt = 'data/dga-220907.txt'
    p_csv = 'data/dga-220907.csv'
    p_csv_agg = 'data/dga-agg-220907.csv'
    p_dga_families = 'data/dga-families-220907'
    p_nomal = ''

    # txt2csv(p_txt, p_csv)
    # family_agg(p_csv,p_csv_agg)
    family_split(p_csv_agg, p_dga_families)




