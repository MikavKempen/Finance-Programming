import numpy as np
from scipy.stats import norm

# ------------------------
# Inputs: Class 1 = highest fare ... Class 5 = lowest fare
# ------------------------
C = 200
fares  = np.array([5000, 3000, 1500, 1000,  500])   # p1..p5
means  = np.array([  10,   20,   60,   80,  100])   # mu1..mu5
stds   = np.array([    5,    8,   20,   15,   10])  # sigma1..sigma5
n = len(fares)

# ------------------------
# Discretize demands into pmfs over 0..dmax
# ------------------------
dmax = int(max(means + 5 * stds))  # wider tail for stability
d = np.arange(dmax + 1)            # 0..dmax

pmfs = []
for mu, sigma in zip(means, stds):
    # discretize normal to integer demand via CDF differences
    z_hi = (d + 0.5 - mu) / sigma
    z_lo = (d - 0.5 - mu) / sigma
    p = norm.cdf(z_hi) - norm.cdf(z_lo)
    p = np.clip(p, 0.0, 1.0)
    s = p.sum()
    if s <= 0:
        p = np.zeros_like(d, dtype=float)
        p[int(round(mu))] = 1.0
    else:
        p /= s
    pmfs.append(p)
pmfs = np.array(pmfs)  # shape (n, dmax+1)

# Precompute cumulative sums for speed
cum_p  = np.cumsum(pmfs, axis=1)            # cumulative prob up to index k
tail_p = 1.0 - cum_p                         # tail prob from k+1 onward
exp_d  = np.cumsum(pmfs * d, axis=1)        # cumulative E[D * 1{D<=k}]

def E_min(q, cls_idx):
    """
    E[min(q, D_cls)] using discrete pmf for class `cls_idx`.
    E[min(q, D)] = sum_{d=0}^{q-1} d*p(d) + q * P(D >= q).
    Handles q beyond pmf support safely.
    """
    if q <= 0:
        return 0.0
    m = pmfs.shape[1]  # length of pmf support = dmax+1
    if q >= m:
        # All mass is in left sum; tail prob ~ 0
        return exp_d[cls_idx, m-1]
    # left sum over d = 0..q-1
    left = exp_d[cls_idx, q-1]
    # tail prob P(D >= q)
    tail = 1.0 - cum_p[cls_idx, q-1]
    return left + q * tail

# ------------------------
# Dynamic Programming
# V[j][x] = max expected revenue when opening class j (then j-1..1) with x seats left
# j = 1..n ; store as 0..n in Python lists for convenience
# ------------------------
V = [np.zeros(C + 1) for _ in range(n + 1)]
Y = [np.zeros(C + 1, dtype=int) for _ in range(n + 1)]

# Base: highest fare class 1
for x in range(C + 1):
    V[1][x] = fares[0] * E_min(x, 0)

# Recur: add classes 2..n (going to lower fares)
for j in range(2, n + 1):   # class index j (1-based): 2..5
    p = fares[j - 1]
    pmf = pmfs[j - 1]
    m = pmf.size
    for x in range(C + 1):
        best_val, best_y = -1e100, 0
        for y in range(x + 1):
            q = x - y  # seats allocated to class j
            # 1) expected revenue from class j
            rev = p * E_min(q, j - 1)
            # 2) continuation expectation: E[V[j-1](max{y, x - D_j})]
            if q <= 0:
                cont = V[j - 1][x]  # sell nothing to j; all seats go to higher classes
            else:
                # left part: d = 0..q-1 (limited by pmf length)
                q_eff = min(q, m)
                d_vals = np.arange(q_eff)            # 0..q_eff-1
                probs_left = pmf[:q_eff]             # P(D=d) for d<q (or up to support)
                # x - d_vals is in [x-(q_eff-1), ..., x], all valid since q_eff <= x+1
                cont_left = np.dot(V[j - 1][x - d_vals], probs_left)
                # tail: d >= q → continuation at V[j-1](y)
                tail = pmf[q:].sum() if q < m else 0.0
                cont = cont_left + V[j - 1][y] * tail
            val = rev + cont
            if val > best_val:
                best_val, best_y = val, y
        V[j][x] = best_val
        Y[j][x] = best_y

# ------------------------
# Report protections in the *opening* order: 5,4,3,2,1
# ------------------------
print("Open Class | Protect for higher classes | Booking limit for this class")
remaining = C
for open_cls in [5, 4, 3, 2]:  # decisions happen at classes 5..2
    y = Y[open_cls][remaining]     # protection to keep for higher classes
    q = remaining - y              # seats you allow to sell now
    print(f"{open_cls:^10} | {y:^27} | {q:^26}")
    remaining = y                  # seats left for higher classes
# When you open class 1, you sell everything remaining
print(f"{1:^10} | {0:^27} | {remaining:^26}")

print(f"\nExpected total revenue (optimal): €{V[n][C]:,.0f}")
