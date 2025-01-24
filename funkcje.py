from numpy.random import normal
from math import exp, log, sqrt
from scipy.stats import norm, chi2
import numpy as np


def BM(n):
    Z = [normal(0, 1) for _ in range(n+1)]
    points = [0]*(n + 1)
    for i in range(1, n+1):
        points[i] = points[i-1] + sqrt(1/n)*Z[i]
    return points


def GBM(n, points, mi=0.01875, sigma=0.25, S0=100):
    S = [0]*(n + 1)
    S[0] = S0
    t = [i / n for i in range(n+1)]

    for i in range(1, n+1):
        S[i] = S0 * exp(mi * t[i] + sigma * points[i])

    return S


def payoff_euro(N, r=0.05, K=100, S0=100, sigma=0.25):
    mi = r - sigma**2/2
    S = S0 * exp(mi + sigma * N)

    return exp(-r) * max(S-K, 0)


def payoff_asian(m, points, r=0.05, K=100):
    S = GBM(m, points)
    A = sum(S[1:])/m

    return exp(-r) * max(A-K, 0)


def black_scholes(sigma=0.25, r=0.05, S0=100, K=100):
    d1 = 1/sigma * (log(S0/K) + r + sigma**2/2)
    d2 = d1 - sigma

    return S0 * norm.cdf(d1) - K * exp(-r) * norm.cdf(d2)


def cmc_eu(R):
    I_eu = 0
    for _ in range(R):
        N = normal(0, 1)
        I_eu += payoff_euro(N)

    return I_eu/R


def cmc_as(R, m):
    I_as = 0
    for _ in range(R):
        I_as += payoff_asian(m, BM(m))

    return I_as/R


def anty_eu(R):
    I_anty = 0
    for _ in range(int(R/2)):
        Z = normal(0, 1)
        I_anty += payoff_euro(N=Z) + payoff_euro(N=-Z)

    return I_anty/R


def kontrol_eu(R, R2):
    X_list = [normal(0, 1) for _ in range(R2)]
    Y_list = [payoff_euro(x) for x in X_list]
    Y_avg = sum(Y_list)/R2
    X_avg = sum(X_list)/R2
    S_yx = sum((Y_list[j] - Y_avg)*(X_list[j] - X_avg) for j in range(R2))/(R2-1)
    c = -S_yx

    X = [normal(0, 1) for _ in range(R)]
    X_cmc = sum(X)/R
    Y_cmc = sum(payoff_euro(x) for x in X)/R

    return Y_cmc + c*X_cmc


# Stratified estimators
def stratified_sampling_bi(n, m, k):
    A = np.tril(np.ones((n, n)) / np.sqrt(n))

    Ksi = np.random.normal(0, 1, n)
    s = np.linalg.norm(Ksi)
    Y = Ksi / s

    U = np.random.uniform(0, 1)
    D = chi2.ppf((k - 1) / m + U / m, df=n)

    Z = np.sqrt(D) * Y
    B_i = A @ Z

    return np.concatenate(([0], B_i))


def proportional_allocation(m, R, n):
    r = int(R/m)
    pay = [0]*m
    for _ in range(r):
        for i in range(m):
            pay[i] += payoff_asian(n, stratified_sampling_bi(n, m, i+1))

    return sum(pay)/R


# Optimal allocation asia
def optimal_allocation(m, R, n):
    Rj2 = 100
    S = [0]*m
    for i in range(m):
        Y_vec = [payoff_asian(n, stratified_sampling_bi(n, m, i+1)) for _ in range(Rj2)]
        Y_avg = sum(Y_vec)/Rj2
        S[i] = sqrt(sum([(Y_vec[j] - Y_avg)**2 for j in range(Rj2)])/(Rj2-1))

    proporcje = [S[i]/sum(S) for i in range(m)]
    pay2 = [0]*m
    for i in range(m):
        r = int(proporcje[i]*R) if int(proporcje[i]*R) > 0 else 1
        for _ in range(r):
            pay2[i] += payoff_asian(n, stratified_sampling_bi(n, m, i+1))
        pay2[i] = pay2[i]/r

    return sum(pay2)/m


# Optimal allocation euro
def optimal_allocation2(m, R, n=1):
    Rj = 1000
    S = [0]*m
    for i in range(m):
        Y_vec = [stratified_sampling_bi(n, m, i+1)[1] for _ in range(Rj)]
        Y_avg = sum(Y_vec)/Rj
        S[i] = sqrt(sum([(Y_vec[j] - Y_avg)**2 for j in range(Rj)])/(Rj-1))

    proporcje = [S[i]/sum(S) for i in range(m)]
    pay2 = [0]*m
    for i in range(m):
        r = int(proporcje[i]*R) if int(proporcje[i]*R) > 0 else 1
        for _ in range(r):
            pay2[i] += payoff_asian(n, stratified_sampling_bi(n, m, i+1))
        pay2[i] = pay2[i]/r

    return sum(pay2)/m


# Wariancje próbkowe
BS = black_scholes()


# opcja europejska
def var_cmc_eu(N, R):
    var = 0
    for _ in range(N):
        var += (cmc_eu(R) - BS) ** 2

    return var / (N-1)


def var_anty_eu(N, R):
    var = 0
    for _ in range(N):
        var += (anty_eu(R) - BS) ** 2

    return var / (N - 1)


def var_kontrol_eu(N, R, R2):
    var = 0
    for _ in range(N):
        var += (kontrol_eu(R, R2) - BS) ** 2

    return var / (N - 1)


def var_proportional_eu(N, R, m, n=1):
    var = 0
    for _ in range(N):
        var += (proportional_allocation(m, R, n) - BS) ** 2

    return var / (N - 1)


def var_optimal_eu(N, R, m):
    var = 0
    for _ in range(N):
        var += (optimal_allocation2(m, R) - BS) ** 2

    return var / (N - 1)


# opcja azjatycka
def var_cmc_as(N, R, n):
    proba = [0]*N
    for i in range(N):
        proba[i] = cmc_as(R, n)

    srednia = sum(proba)/N
    var = sum((proba[i] - srednia)**2 for i in range(N))

    return var / (N - 1)


def var_proprtional_as(N, R, n, m):
    proba = [0] * N
    for i in range(N):
        proba[i] = proportional_allocation(m, R, n)

    srednia = sum(proba) / N
    var = sum((proba[i] - srednia) ** 2 for i in range(N))

    return var / (N - 1)


def var_optimal_as(N, R, n, m):
    proba = [0] * N
    for i in range(N):
        proba[i] = optimal_allocation(m, R, n)

    srednia = sum(proba) / N
    var = sum((proba[i] - srednia) ** 2 for i in range(N))

    return var / (N - 1)


if __name__ == "__main__":
    from time import time
    start = time()
    N, R = 100, 10000
    var_prop_eu2 = var_proportional_eu(N, R, 2)
    stop = time()
    print(var_prop_eu2, "czas:", stop-start)
