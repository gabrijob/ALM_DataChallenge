import numpy as np
import laKernelFast


beta = 0.2
d = 1
e = 11
# Similarity matrix
S = np.identity(4) # 4 is number of genomes C,G,A,T

def transform_data(sequences):

    transformed_seqs = []
    for seq in sequences:
        char_seq = list(seq)[:-1]

        genomes = []
        for genome in char_seq:
            if genome == 'A':
                genomes.append(0)
            if genome == 'C':
                genomes.append(1)
            if genome == 'G':
                genomes.append(2)
            if genome == 'T':
                genomes.append(3)

        transformed_seqs.append(genomes)

    return np.array(transformed_seqs, np.int)


def la_kernel(x, y):
    dx = x.size
    dy = y.size

    # Initialization
    M = [[0 for j in range(dy)] for i in range(dx)]
    X_1 = [[0 for j in range(dy)] for i in range(dx)]
    Y_1 = [[0 for j in range(dy)] for i in range(dx)]
    X_2 = [[0 for j in range(dy)] for i in range(dx)]
    Y_2 = [[0 for j in range(dy)] for i in range(dx)]

    for i in range(1, dx):
        for j in range(1, dy):
            M[i][j] = np.exp(beta*S[x[i]][y[j]]) * (1 + X_1[i-1][j-1] + Y_1[i-1][j-1] + M[i-1][j-1])
            X_1[i][j] = np.exp(beta*d) * M[i-1][j] + np.exp(beta*e) * X_1[i-1][j]
            Y_1[i][j] = np.exp(beta*d) * (M[i][j- 1] + X_1[i][j-1]) + np.exp(beta*e) * Y_1[i][j-1]
            X_2[i][j] = M[i-1][j] + X_2[i-1][j]
            Y_2[i][j] = M[i][j-1] + X_2[i][j-1] + Y_2[i][j-1]

    return 1 + X_2[dx-1][dy-1] + Y_2[dx-1][dy-1] + M[dx-1][dy-1]


def log_la_kernel(x, y):
    return 1/beta * np.log(la_kernel(x, y))


def la_ekm(x, y, all_sequences):
    sum_la_ekm = 0

    for seq in all_sequences:
        sum_la_ekm = sum_la_ekm + log_la_kernel(x, seq) * log_la_kernel(y, seq)

    return sum_la_ekm


def gram_matrix_la(sequences):
    n = np.size(sequences, 0)
    gram_matrix = np.zeros((n,n))

    count = 0
    # the gram matrix is symmetric
    for i in range(n):
        for j in range(i+1):
            #gram_matrix[i,j] = la_ekm(sequences[i], sequences[j], sequences)
            try:
                lak = laKernelFast.calc(laKernelFast, (sequences[i], sequences[j]))
            except SystemError:
                lak = la_kernel(sequences[i], sequences[j])
            gram_matrix[i][j] = lak
            count = count + 1
            if not count%1000:
                print("Kernels calculated: {}".format(count))

    upTriIdxs = np.triu_indices(n)
    gram_matrix[upTriIdxs] = gram_matrix.T[upTriIdxs]

    return gram_matrix