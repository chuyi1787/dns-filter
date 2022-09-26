import numpy as np
import pandas as pd


def get_ngram(p, n):
    res = []
    data = pd.read_csv(p).values
    for idd in range(data.shape[0]):
        domain = data[idd][1]
        ll = len(domain)
        if ll < n:
            continue
        ng_domain = []
        for i in range(ll-n+1):
            ng_domain.append(domain[i:i+n])
        # truncate
        if len(ng_domain) > 64:
            ng_domain = ng_domain[-64:]
            print(domain)
        str_ng_domain = ' '.join(ng_domain)
        res.append([data[idd][0], str_ng_domain])
    return res


def list2csv(ngram_list, p):
    np.savetxt(p, np.array(ngram_list), delimiter=',', fmt='%s')
    return


if __name__ == '__main__':


    p_dga = '../../data/dga/360lab/dga-agg-220907.csv'
    p_dga_ngram = '../../data/dga/360lab/dga-{}gram-220907.csv'

    ngram = 2
    Ngram_list = get_ngram(p_dga, ngram)
    list2csv(Ngram_list, p_dga_ngram.format(ngram))


