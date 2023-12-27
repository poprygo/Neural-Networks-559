import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

def u(x):
    return 1 if x >= 0 else 0

def generate_data(w, n=100):
    S = [np.array([1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) for i in range(n)]
    d = [1 if w @ x >= 0 else 0 for x in S]
    S0 = [S[i] for i in range(len(S)) if d[i] == 0]
    S1 = [S[i] for i in range(len(S)) if d[i] == 1]
    return S0, S1, S, d

def pta(eta, w0, S, d):
    w = np.array(w0)
    epochs = 0
    mclass = [sum([1 if u(w @ S[i]) != d[i] else 0 for i in range(len(S))])]
    
    while mclass[-1] != 0:
        epochs += 1
        for i in range(len(S)):
            w = w + eta * S[i] * (d[i] - u(w @ S[i]))
        mclass.append(sum([1 if u(w @ S[i]) != d[i] else 0 for i in range(len(S))]))
    
    return w, epochs, mclass


def plot_data_and_sep(S0, S1, w):
    plt.figure()
    plt.plot([x[1] for x in S0], [x[2] for x in S0], marker="s", linestyle="none", fillstyle="none")
    plt.plot([x[1] for x in S1], [x[2] for x in S1], marker="o", linestyle="none", fillstyle="none")
    plt.plot([-1, 1], [-(w[0] + w[1] * x) / w[2] for x in [-1, 1]], linestyle="--")
    plt.title("Initial lass separation. $n=100$ points, w=({:.2f}, {:.2f}, {:.2f})".format(w[0], w[1], w[2]))
    plt.legend(["$S_0$", "$S_1$", "Boundary"], loc=1)
    plt.axis([-1, 1, -1, 1])
    plt.show()

def plot_results(S0, S1, wr, weights, eta):
    plt.figure()
    plt.plot([x[1] for x in S0], [x[2] for x in S0], marker="s", linestyle="none", fillstyle="none")
    plt.plot([x[1] for x in S1], [x[2] for x in S1], marker="o", linestyle="none", fillstyle="none")
    plt.plot([-1, 1], [-(wr[0] + wr[1] * x) / wr[2] for x in [-1, 1]], linestyle="--")

    for e in eta:
        plt.plot([-1, 1], [-(weights[e][0] + weights[e][1] * x) / weights[e][2] for x in [-1, 1]], linestyle="--")

    plt.title(f"$n={len(S0)+len(S1)}$ Class separation for different values of $\eta$")
    plt.legend(["$S_0$", "$S_1$", "Boundary (exact)"] + ["Boundary ($\eta={}$)".format(e) for e in eta], loc=1)
    plt.axis([-1, 1, -1, 1])
    plt.show()


def plot_misclassifications(eta, epochs, mclass, n):
    plt.figure()
    plt.plot(range(epochs+1), mclass, c='black')
    plt.xlabel('Epoch', labelpad=10, fontsize=12)
    plt.ylabel('Number of Misclassifications', labelpad=10, fontsize=12)
    plt.title(f'{n=} Epoch vs Misclassifications for η={eta}', y=1.05, fontsize=14, loc='left')
    plt.show()

def main():
    np.random.seed(42)
    wr = np.array([np.random.uniform(-1/4, 1/4), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
    n = 100
    S0, S1, S, d = generate_data(wr, n)
    plot_data_and_sep(S0, S1, wr)
    
    w0 = np.random.uniform(-1, 1, 3)
    for n in [100, 1000]:
        S0, S1, S, d = generate_data(wr, n)
        weights = {}
        epochs = {}
        mclass = {}

        for eta in [0.1, 1, 10]:
            weights[eta], epochs[eta], mclass[eta] = pta(eta, w0, S, d)
            print(f'{n=} {eta=} {epochs[eta]=}')
            print(f"w = {wr}")
            print(f"w\' = {w0}")
            print(f"w\" = {weights[eta]}")        
        plot_results(S0, S1, wr, weights, [0.1, 1, 10])

        for eta in [0.1, 1, 10]:
            plot_misclassifications(eta, epochs[eta], mclass[eta], n)

if __name__ == "__main__":
    main()
