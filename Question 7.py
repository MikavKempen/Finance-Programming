# --- Parameters we start with ---
# p0 = initial full price (upper bound, we can only mark down)
# K = total inventory (max number of coats)
# T = number of selling days (90 days season)
# lam = arrival probability each day (0.8)
# mu, sigma = mean and std of willingness to pay (Normal dist)
p0 = 1000.0
K  = 20
T  = 90
lam   = 0.8
mu    = 300.0
sigma = 10.0

# --- Libraries we need ---
import numpy as np
from math import sqrt, erf

# --- Demand model ---
# Here we define the price-response function b(p)
# b(p) = lambda * P(w >= p), where w ~ N(mu, sigma^2)
# So basically: chance that a customer arrives * chance that their WTP ≥ price
def Phi(z):
    # standard normal cumulative distribution function
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))

def b(p):
    # probability of selling at price p
    return lam * (1.0 - Phi((p - mu) / sigma))

# --- Golden section search ---
# this is a numerical search to find the price p that maximizes our expected revenue function
# we use this instead of taking derivatives, because we have to do it 90*20 times and it's simpler + robust
def golden_section_max(f, a, b, tol=1e-6, max_iter=200):
    phi = (1 + sqrt(5)) / 2.0
    invphi = 1.0 / phi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        # check which side of the interval gives higher value, then zoom in there
        if fc < fd:
            a, c = c, d
            fc = fd
            d = a + invphi * (b - a)
            fd = f(d)
        else:
            b, d = d, c
            fd = fc
            c = b - invphi * (b - a)
            fc = f(c)
    return (c, fc) if fc >= fd else (d, fd)

# --- Dynamic programming solver ---
# this is the core backward induction loop (Bellman recursion)
# we fill in vt[t,x] = expected future revenue from day t onward with x units in stock
# and pt[t,x] = optimal price to ask that day
def solve_markdown_backward_induction(p0, K, b, T=90):
    # we make an array for all value functions (T+2 because we also store v_{T+1}(x)=0)
    vt = np.zeros((T + 2, K + 1), dtype=float)
    # and one for all prices for each day and inventory level
    pt = np.zeros((T + 1, K + 1), dtype=float)

    # start going backwards from the end of the season (day 90) to the start (day 1)
    for t in range(T, 0, -1):
        # if no inventory left, revenue stays the same (always 0)
        vt[t, 0] = vt[t + 1, 0]
        pt[t, 0] = 0.0

        # loop over all possible inventory levels
        for x in range(1, K + 1):
            # delta = marginal value of one extra unit
            delta = vt[t + 1, x - 1] - vt[t + 1, x]

            # define the function we want to maximize
            # gain(p) = probability of sale * (immediate profit + effect on future value)
            gain = lambda p: b(p) * (p + delta)

            # find the price that maximizes expected gain for this day and inventory
            p_star, best_gain = golden_section_max(gain, 0.0, p0)

            # update the Bellman recursion
            vt[t, x] = vt[t + 1, x] + best_gain
            pt[t, x] = p_star

    # remove the unused extra row and return the clean tables
    vt_star = vt[1:T + 1, :]
    pt_star = pt[1:T + 1, :]
    return vt_star, pt_star

# --- Run the solver ---
# this calculates all v*_t(x) and p*_t(x) according to the dynamic program
vt_star, pt_star = solve_markdown_backward_induction(p0, K, b, T)

# make output readable
np.set_printoptions(precision=2, suppress=True)

# --- Print all results ---
print("\n=== Optimal Markdown Pricing by Backward Induction ===")
print(f"Parameters: p0={p0}, K={K}, T={T}, λ={lam}, μ={mu}, σ={sigma}\n")

# show the expected revenue (value function) for every day and inventory level
print("v*_t(x): expected cumulative revenue at start of period t")
for t in range(T):
    print(f"Day {t+1:2d}: ", vt_star[t, :])

# show the optimal price decisions for each day and inventory level
print("\nOptimal price p*_t(x) for each day t and inventory level x:")
for t in range(T):
    print(f"Day {t+1:2d}: ", pt_star[t, :])

# small summary with key values to interpret the result
print("\nSummary:")
print(f"  - v*_1(K) = {vt_star[0, K]:.2f}  (expected total revenue at start)")
print(f"  - p*_90(1) = {pt_star[89, 1]:.2f} (optimal last-day price, same for all x≥1)")
print(f"  - Example: p*_1(K) = {pt_star[0, K]:.2f}")


#
# END OF QUESTION C
#
# --- Part (d): Static price vs. Markdown policy (100-run simulations) ---
# In this part we compare a fixed (static) price strategy vs. the dynamic markdown policy.
# The goal is to find the best static price p*_static and see how both perform over 100 seasons.

# --- Step 1: Analytical expected revenue for a static price ---
#
# Function that calculates E[min(S,K)] for S ~ Bin(T, q)
# This is the expected number of items sold when daily sale probability = q
# If q = 0 or q = 1, we handle edge cases directly
def expected_min_sales(T, K, q):
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return float(min(T, K))
    # Use recurrence to compute Binomial probabilities
    pmf0 = (1.0 - q) ** T  # P(S=0)
    pmf = pmf0
    cdf_to_Km1 = pmf0
    exp_to_Km1 = 0.0
    for s in range(0, K - 1):
        exp_to_Km1 += s * pmf
        # recurrence: P(S=s+1) from P(S=s)
        pmf = pmf * ((T - s) / (s + 1.0)) * (q / (1.0 - q))
        cdf_to_Km1 += pmf
    exp_to_Km1 += (K - 1) * pmf
    p_le_Km1 = cdf_to_Km1
    # expected sales = sold units below K + forced sales up to K
    return exp_to_Km1 + K * (1.0 - p_le_Km1)

# Revenue function for a static price
# Expected revenue = price * expected units sold at that price
def R_static(p):
    z = (p - mu) / sigma
    q = lam * (1.0 - Phi(z))  # daily sale probability
    E_units = expected_min_sales(T, K, q)
    return p * E_units

# Find the static price p*_static that maximizes expected revenue
# Use the same golden-section search as before (robust, no derivative needed)
p_static_star, R_static_star = golden_section_max(R_static, 0.0, p0)

# --- Step 2: Monte Carlo simulation setup ---

# Random generator (fixed seed for reproducibility)
rng = np.random.default_rng(42)

# One run of static pricing simulation:
# price never changes, at most 1 sale per day if customer arrives and WTP >= price
def simulate_static_once(p):
    x, rev = K, 0.0
    for t in range(T):
        if x == 0:
            break
        if rng.random() < lam:           # arrival?
            w = rng.normal(mu, sigma)    # willingness to pay
            if w >= p:
                rev += p
                x -= 1
    return rev

# One run of markdown simulation:
# price depends on remaining stock and day t from the optimal markdown policy
def simulate_markdown_once():
    x, rev = K, 0.0
    for t in range(T):
        if x == 0:
            break
        p_today = pt_star[t, x]
        if rng.random() < lam:
            w = rng.normal(mu, sigma)
            if w >= p_today:
                rev += p_today
                x -= 1
    return rev

# --- Step 3: Run 100 Monte Carlo simulations for both strategies ---
N = 100
static_runs = np.array([simulate_static_once(p_static_star) for _ in range(N)])
markdown_runs = np.array([simulate_markdown_once() for _ in range(N)])

# --- Step 4: Print results ---
# Compare mean revenues and standard deviations between both approaches
print("\n=== Part (d): Static vs. Markdown — 100-run Monte Carlo ===")
print(f"Optimal static price p*_static ≈ {p_static_star:.2f}")
print(f"Expected static revenue (analytical) R_static(p*_static) ≈ {R_static_star:.2f}")

print("\nSimulation (100 runs):")
print(f"  Static policy:   mean = {static_runs.mean():.2f}, std = {static_runs.std(ddof=1):.2f}")
print(f"  Markdown policy: mean = {markdown_runs.mean():.2f}, std = {markdown_runs.std(ddof=1):.2f}")

print(f"\nAverage improvement of markdown over static (simulation): "
      f"{markdown_runs.mean() - static_runs.mean():.2f}")
#
# END QUESTION D
#
# ===== Part (e): Markdown with disposal cost c per leftover unit =====
# New twist: any unit left after day 90 is waste and costs c on day 100, but i think they mean 90.
# Implementation detail: in DP we handle this by changing the terminal value
# from v_{T+1}(x) = 0 to v_{T+1}(x) = -c * x (penalize leftovers).

c = 80.0  # disposal cost per unsold unit charged after the season

def solve_markdown_with_disposal(p0, K, b, c, T=90):
    """
    Same DP as in (c) but with terminal penalty for leftovers.
    Returns:
      vt_waste[t,x] = optimal expected net revenue from day t with x units (includes waste penalty),
      pt_waste[t,x] = optimal price at day t with x units.
    """
    import numpy as np

    # allocate value and price tables (include extra row for t = T+1)
    vt = np.zeros((T + 2, K + 1), dtype=float)
    pt = np.zeros((T + 1, K + 1), dtype=float)

    # terminal condition: leftover x units cost c each after the season
    for x in range(K + 1):
        vt[T + 1, x] = -c * x

    # backward sweep: t = T,...,1
    for t in range(T, 0, -1):
        # with 0 inventory nothing to sell
        vt[t, 0] = vt[t + 1, 0]
        pt[t, 0] = 0.0

        for x in range(1, K + 1):
            # marginal value of one extra unit going into tomorrow
            # NOTE: on the last selling day this equals c (because tomorrow you pay c if unsold)
            delta = vt[t + 1, x - 1] - vt[t + 1, x]

            # choose today’s price to maximize expected gain:
            # sale prob * (today's price + saved future value from having one less unit)
            gain = lambda p: b(p) * (p + delta)

            # same robust 1-D optimizer as before
            p_star, best_gain = golden_section_max(gain, 0.0, p0)

            # Bellman update
            vt[t, x] = vt[t + 1, x] + best_gain
            pt[t, x] = p_star

    # strip the extra terminal row
    return vt[1:T + 1, :], pt[1:T + 1, :]

# --- Run and summarize ---
vt_waste, pt_waste = solve_markdown_with_disposal(p0, K, b, c, T)

np.set_printoptions(precision=2, suppress=True)

print("\n=== Part (e): Markdown with disposal cost ===")
print(f"Parameters: p0={p0}, K={K}, T={T}, λ={lam}, μ={mu}, σ={sigma}, c={c}\n")

# show a few representative days (start, mid, end)
print("Sample rows of v*_t(x) with disposal (net of waste cost):")
for t in [90, 60, 30, 1]:
    print(f"Day {t:2d}: ", vt_waste[t-1, :])

print("\nSample rows of p*_t(x) with disposal:")
for t in [90, 60, 30, 1]:
    print(f"Day {t:2d}: ", pt_waste[t-1, :])

# key sanity checks
print("\nKey facts:")
print(f"  - Last-day price p*_90(1) = {pt_waste[89, 1]:.2f} (same for all x≥1).")
print(f"  - Net expected value v*_1(K) (includes waste penalties) = {vt_waste[0, K]:.2f}.")
print(f"  - Example: p*_1(K) = {pt_waste[0, K]:.2f}.")
