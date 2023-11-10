import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
import timeit


# ==================================================================================================================== #


def load_xyz(path, N, number_of_time_steps):
    all_configurations = []
    file = open(path, 'r')

    all_lines = file.readlines()
    number_of_atoms = int(all_lines[0])

    for k in range(number_of_time_steps):

        s = int(k * number_of_atoms + (k + 1) * 2)

        coordinates = np.zeros((N, 3))  # XYZ
        el = np.zeros(N)

        for i in range(N):
            index = int(s + i)

            line = all_lines[index].split(' ')

            coordinates[i, 0] = float(line[1])
            coordinates[i, 1] = float(line[2])
            coordinates[i, 2] = float(line[3])

            if line[0] == 'NA':
                el[i] = 1

        all_configurations.append([coordinates, el])

    return all_configurations


def generate_q_vec(k_max,q_max,Lx,Ly):

    q_vec = np.zeros((q_max,2))
    index = 0

    for i in range(-k_max,k_max+1):
        for j in range(-k_max, k_max+1):

            if i == j == 0: continue

            #  print(f'i = {i} j = {j}')

            q_vec[index][0] = (2*np.pi/Lx) * i
            q_vec[index][1] = (2*np.pi/Ly) * j

            index += 1

    return q_vec


def rebuild_coordinates(coordinates, Lx, Ly, Lz, number_of_chains, chain_sizes, AB_ratio, N):
    Lxh = Lx / 2
    Lyh = Ly / 2
    Lzh = Lz / 2

    for i in range(N):

        if coordinates[i, 0] > Lxh:
            coordinates[i, 0] -= Lx
        if coordinates[i, 0] < -Lxh:
            coordinates[i, 0] += Lx

        if coordinates[i, 1] > Lyh:
            coordinates[i, 1] -= Ly
        if coordinates[i, 1] < -Lyh:
            coordinates[i, 1] += Ly



    for i in range(number_of_chains):

        AS = 1
        AF = AB_ratio[i][0]

        BS = AB_ratio[i][0] + 1
        BF = chain_sizes[i]

        shift = sum(chain_sizes[:i])

        for j in range(AS, AF):
            index = shift + j
            # print(f'{i} * {len_chain} + {j} = {index}')

            dx = coordinates[index, 0] - coordinates[index - 1, 0]
            dy = coordinates[index, 1] - coordinates[index - 1, 1]
            dz = coordinates[index, 2] - coordinates[index - 1, 2]

            dx = dx - Lx * np.rint(dx / Lx)
            dy = dy - Ly * np.rint(dy / Ly)
            dz = dz - Lz * np.rint(dz / Lz)

            coordinates[index, 0] = dx + coordinates[index - 1, 0]
            coordinates[index, 1] = dy + coordinates[index - 1, 1]
            coordinates[index, 2] = dz + coordinates[index - 1, 2]

        # kontrola prvního segmentu v bloku B

        index_A = shift
        index_B = shift + AB_ratio[i][0]

        dx = coordinates[index_B, 0] - coordinates[index_A, 0]
        dy = coordinates[index_B, 1] - coordinates[index_A, 1]
        dz = coordinates[index_B, 2] - coordinates[index_A, 2]

        dx = dx - Lx * np.rint(dx / Lx)
        dy = dy - Ly * np.rint(dy / Ly)
        dz = dz - Lz * np.rint(dz / Lz)

        coordinates[index_B, 0] = dx + coordinates[index_A, 0]
        coordinates[index_B, 1] = dy + coordinates[index_A, 1]
        coordinates[index_B, 2] = dz + coordinates[index_A, 2]

        for j in range(BS, BF):
            index = shift + j
            # print(f'{i} * {len_chain} + {j} = {index}')

            dx = coordinates[index, 0] - coordinates[index - 1, 0]
            dy = coordinates[index, 1] - coordinates[index - 1, 1]
            dz = coordinates[index, 2] - coordinates[index - 1, 2]

            dx = dx - Lx * np.rint(dx / Lx)
            dy = dy - Ly * np.rint(dy / Ly)
            dz = dz - Lz * np.rint(dz / Lz)

            coordinates[index, 0] = dx + coordinates[index - 1, 0]
            coordinates[index, 1] = dy + coordinates[index - 1, 1]
            coordinates[index, 2] = dz + coordinates[index - 1, 2]

    coordinates[:, 0] += Lxh
    coordinates[:, 1] += Lyh
    coordinates[:, 2] += Lzh

    return coordinates


# ==================================================================================================================== #


def analysis(input_path, output_path):

    A_lengths = np.loadtxt(input_path + 'AchainLengths.txt')
    B_lengths = np.loadtxt(input_path + 'BchainLengths.txt')

    AB_ratio = np.zeros((B_lengths.shape[0], 2), dtype=int)
    AB_ratio[:, 0] = A_lengths
    AB_ratio[:, 1] = B_lengths


    # ================================================================================================================ #
    chain_sizes = np.asarray((AB_ratio[:, 0] + AB_ratio[:, 1]).astype(int))  # length of chain
    number_of_chains = AB_ratio.shape[0]  # number of chains
    N = int(sum(chain_sizes))
    number_of_configuration = 100  # number of time steps in xyz file

    all_configuration = load_xyz(f'{input_path}DPDFluidProd50.xyz', N, number_of_configuration)
    # ================================================================================================================ #

    shifts = np.zeros(number_of_chains + 1, dtype=int)

    for i in range(number_of_chains + 1):
        shifts[i] = sum(chain_sizes[:i])

    # Size of box
    Lx = int(60)
    Ly = int(60)
    Lz = int(np.amax(np.amax(AB_ratio)) * 1.1)

    k_max = 12
    q_max = int((2 * k_max + 1) ** 2 - 1)

    q_vec = np.asarray(generate_q_vec(k_max, q_max, Lx, Ly))

    q_size = (q_vec[:, 0] ** 2 + q_vec[:, 1] ** 2) ** (1 / 2)

    S_q_A = np.zeros(q_max)
    S_q_B = np.zeros(q_max)

    # ================================================================================================================ #

    for i in range(number_of_configuration):

        if i % 10 == 0: print(i)

        coordinates = all_configuration[i][0]

        # rebuilding chains
        coordinates = rebuild_coordinates(coordinates, Lx, Ly, Lz, number_of_chains, chain_sizes, AB_ratio, N)
        coordinates = np.asarray(coordinates)

        current_S_q_A,current_S_q_B = compute_sf(q_vec,q_max,coordinates,number_of_chains,chain_sizes,AB_ratio,shifts)

        S_q_A = S_q_A + current_S_q_A
        S_q_B = S_q_B + current_S_q_B

    # ================================================================================================================ #
    '''
    S_q_A = S_q_A / number_of_configuration
    S_q_B = S_q_B / number_of_configuration

    plot_s_q(q_size, S_q_A, 'A', output_path)
    plot_s_q(q_size, S_q_B, 'B', output_path)

    x_uniA, y_uniA = sort_and_get_unique(q_size, S_q_A)
    x_uniB, y_uniB = sort_and_get_unique(q_size, S_q_B)

    plot_s_q_uni(x_uniA, y_uniA, 'A', output_path)
    plot_s_q_uni(x_uniB, y_uniB, 'B', output_path)

    write_S_q(q_size, S_q_A, S_q_B, output_path)
    write_S_q_uni(x_uniA, y_uniA, 'A', output_path)
    write_S_q_uni(x_uniB, y_uniB, 'B', output_path)
    '''

    return None


# ==================================================================================================================== #


@jit(nopython=True, parallel=True)
def compute_sf(q_vec,q_max,coordinates,number_of_chains, chain_sizes, AB_ratio,shifts):

    S_q_A = np.zeros(q_max)
    S_q_B = np.zeros(q_max)

    for q in prange(0,q_max):

        cos_A = 0
        sin_A = 0

        cos_B = 0
        sin_B = 0

        for i in range(number_of_chains):

            AS = 0
            AF = AB_ratio[i][0]

            BS = AB_ratio[i][0]
            BF = chain_sizes[i]

            for j in range(AS, AF):
                index = shifts[i] + j

                sp = (q_vec[q,0] * coordinates[index,0]) + (q_vec[q,1] * coordinates[index,1])
                cos_A = cos_A + np.cos(sp)
                sin_A = sin_A + np.sin(sp)

            for j in range(BS, BF):
                index = shifts[i] + j

                sp = (q_vec[q, 0] * coordinates[index, 0]) + (q_vec[q, 1] * coordinates[index, 1])
                cos_B = cos_B + np.cos(sp)
                sin_B = sin_B + np.sin(sp)

        S_q_A[q] = (cos_A ** 2 + sin_A ** 2) / np.sum(AB_ratio[:,0])
        S_q_B[q] = (cos_B ** 2 + sin_B ** 2) / np.sum(AB_ratio[:,1])

    return S_q_A,S_q_B


# ==================================================================================================================== #


def plot_s_q(x,y,label,output_path):

    plt.scatter(x,y,label=label)

    plt.xlabel(r'$q$')
    plt.ylabel(r'S(q) ' + label)
    plt.legend()

    plt.savefig(f'{output_path}structure_factor{label}.png')

    plt.clf()
    plt.close()

    return None


def sort_and_get_unique(x,y):

    x, y = zip(*sorted(zip(x, y)))

    x = np.array(x)
    y = np.array(y)

    x_uni = np.unique(x)

    y_uni = np.zeros(x_uni.shape[0])
    y_counter = np.zeros(x_uni.shape[0])

    for i in range(x.shape[0]):

        index = np.where(x_uni == x[i])[0][0]
        y_uni[index] += y[i]
        y_counter[index] += 1

    y_uni = y_uni / y_counter

    return x_uni,y_uni


def plot_s_q_uni(x,y,label,output_path):

    plt.plot(x, y, label=label)

    plt.xlabel(r'$q$')
    plt.ylabel(r'S(q) ' + label)
    plt.legend()

    plt.savefig(f'{output_path}structure_factor_unique{label}.png')

    plt.clf()
    plt.close()

    return None


def write_S_q(x, A,B, output_path):
    col_width = 16

    file = open(f'{output_path}S(q).txt', 'w')

    for i in range(len(A)):
        bins_str = format(x[i], '.7f')
        a_str = format(A[i], '.7f')
        b_str = format(B[i], '.7f')

        file.write((col_width - len(bins_str)) * ' ' + bins_str)
        file.write((col_width - len(a_str)) * ' ' + a_str)
        file.write((col_width - len(b_str)) * ' ' + b_str + '\n')

    file.close()

    return None


def write_S_q_uni(x, y,label, output_path):
    col_width = 16

    file = open(f'{output_path}S(q){label}.txt', 'w')

    for i in range(len(y)):
        bins_str = format(x[i], '.7f')
        y_str = format(y[i], '.7f')

        file.write((col_width - len(bins_str)) * ' ' + bins_str)
        file.write((col_width - len(y_str)) * ' ' + y_str + '\n')

    file.close()

    return None


# ==================================================================================================================== #


if __name__ == '__main__':

    print('Hello, home!')

    folders = ['s0.03', 's0.1', 's0.5', 's1.0']
    times = []

    for ind, folder in enumerate(folders):
        #'''
        start_time = timeit.default_timer()

        DATA_PATH = f'Data/{folder}/'
        OUTPUT_PATH = 'Results/'

        analysis(DATA_PATH, OUTPUT_PATH)

        finish_time = timeit.default_timer()
        print(f'Výpočet strukturního faktoru trval : {finish_time - start_time}')
        times.append(finish_time - start_time)
        # '''

    np.savetxt(f'Results/times_CPU.txt',np.array(times))