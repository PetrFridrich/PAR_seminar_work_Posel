import numpy as np
import matplotlib.pyplot as plt


def main():

    x = [0,1,2,3]

    y1 = np.loadtxt(f'Results/times_CPU.txt')
    y2 = np.loadtxt(f'Results/times_GPU_1.txt')
    y3 = np.loadtxt(f'Results/times_GPU_2.txt')

    fig = plt.figure(figsize=(10,5))

    plt.plot(x, y1, label='CPU', marker='*', c='blue')
    plt.plot(x, y2, label='GPU_1', marker='*', c='green')
    plt.plot(x, y3, label='GPU_2', marker='*', c='red')

    plt.xticks(x,labels=['s0.03', 's0.1', 's0.5', 's1.0'])

    plt.title('Časová náročnost výpočtu strukturního faktoru')
    plt.ylabel('Čas (sec)')
    plt.xlabel('Velikost úlohy')

    plt.legend()

    plt.savefig(f'Results/Graph.png')
    plt.clf()
    plt.close()


if __name__ == '__main__':

    print('Hello, home!')

    main()