import numpy as np


# C size: 38
C = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
     'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', '.']


def load_bigram_alphabet(c):  # N size: 1444 = 38*38
    n = []
    for i in range(len(c)):
        for j in range(len(c)):
            n.append(c[i] + c[j])
    return n


# 将各组域名用n-gram表示作为神经网络的输入(n = 2)
def n_gram_representation(dns_data, c, n=2, N_MAX=64):
    bi_alp = load_bigram_alphabet(c)
    # size: dns_data.shape[0]*1444*64
    data_input = np.zeros(
        [dns_data.shape[0], len(bi_alp), N_MAX], dtype='float32')

    for i in range(dns_data.shape[0]):
        domain = dns_data[i][0]

        # get ngrams and truncate
        domain_ngrams = []
        for j in range(len(domain)-n+1):
            domain_ngrams.append(domain[j:j+n])
        domain_ngrams = domain_ngrams[-N_MAX:]

        # to one hot code
        for j in range(len(domain_ngrams)):
            index_ = bi_alp.index(domain_ngrams[j])
            data_input[i][index_][j] = 1

    return data_input


if __name__ == '__main__':
    n_gram_representation()