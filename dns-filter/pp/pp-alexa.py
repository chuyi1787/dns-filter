import numpy as np
import pandas as pd
from copy import deepcopy

# 读取top_1m.csv
top_1m = pd.read_csv('data/top_1m.csv')
top_1m = top_1m.values
top_1m[:, 0] = 'normal'
top_1w = top_1m[:10000, :2]

# 将top 1w存储到top_1w.csv
np.savetxt('data/top_1w.csv', top_1w, delimiter=',', fmt='%s')

# 读取top-1w.csv
top_1w = pd.read_csv('data/top_1w.csv')
top_1w = top_1w.values


#############################################################
# 预处理 去除顶级域、二级域等非决定性因素 保留有决定性的主体部分


# 去除顶级域、二级域等非决定性因素
def remove_top_level(dns):
    length_dns = len(dns)
    dns_remove = deepcopy(dns)
    for i in range(length_dns):
        dns_each = dns[i][1]
        length_dns_each = len(dns_each)
        for j in range(length_dns_each):
            if dns_each[j] == '.':
                dns_remove[i][1] = dns_each[0:j]
                break
    return dns_remove


# 处理并存储预处理后的正常域名
dns_normal = remove_top_level(top_1w)
np.savetxt('data/dns_normal.csv', dns_normal, delimiter=',', fmt='%s')
dns_normal = pd.read_csv('data/dns_normal.csv')
dns_normal = dns_normal.values
