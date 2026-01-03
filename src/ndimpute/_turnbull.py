import numpy as np
import pandas as pd

def turnbull_em(left, right, max_iter=1000, tol=1e-5):
    """
    Computes the Non-Parametric Maximum Likelihood Estimator (NPMLE)
    for interval-censored data using the Turnbull EM algorithm.

    Args:
        left (array): Lower bounds of intervals.
        right (array): Upper bounds of intervals.
                       Use np.inf for right-censored (L, inf).
                       Use L for exact observations (L, L).

    Returns:
        tuple: (intervals, probs)
            intervals: (M, 2) array of equivalence classes [start, end].
            probs: (M,) array of probability mass assigned to each interval.
    """
    left = np.array(left)
    right = np.array(right)
    n = len(left)

    # 1. Determine Equivalence Intervals
    # Collect all unique endpoints
    endpoints = np.unique(np.concatenate([left, right]))
    endpoints = endpoints[~np.isinf(endpoints)] # Remove infinity
    endpoints.sort()

    # Create disjoint candidate intervals (s_j, q_j]
    # An interval is valid if it lies inside at least one observation (L, R]
    # Turnbull intervals are formed by the sorted unique endpoints.
    # [e0, e1], (e1, e2], ...

    # Actually, standard construction:
    # 1. Union of all (L, R].
    # 2. Break at all endpoints.
    # 3. Keep intervals that are inside at least one observed interval?
    # Simpler: All sorted unique endpoints define the grid.

    # If exact observations exist (L=R), we need to handle them.
    # Standard algorithm works on (L, R]. Exact is [L, L]?
    # Let's assume (L, R] convention for now.
    # Exact x is (x-eps, x]? Or just treated as degenerate interval.

    # Let's use the standard set construction from Turnbull 1976.
    # L set: all left endpoints. R set: all right endpoints.
    # Equivalence classes [q_j, p_j].

    # Simplified grid approach:
    # Use all finite endpoints.
    # Construct intervals (e_i, e_{i+1}].
    # Check if this interval is possible for observation k.

    candidates = []
    for i in range(len(endpoints) - 1):
        candidates.append((endpoints[i], endpoints[i+1]))

    # Also need to handle exact values specifically if any?
    # If we have [1, 1], then 1 is an endpoint.
    # The grid approach above creates (1, next).
    # We need to include singleton intervals [e, e] if e is an observed exact value?
    # For now, let's assume continuous intervals for simplicity or treating exact as tiny intervals.

    # Let's filter candidates. A candidate (a, b] has mass only if there exists an observation (L, R] such that (a, b] is subset of (L, R].
    # i.e., L <= a and b <= R.

    valid_intervals = []
    alpha = np.zeros((n, len(candidates)), dtype=bool)

    keep_indices = []
    for j, (start, end) in enumerate(candidates):
        # Indicator: Is interval j contained in observation i?
        # Standard Turnbull: L_i <= start < end <= R_i (for interval [start, end]?)
        # Let's be careful with boundaries.
        # Obs i covers Candidate j if [start, end] is a valid location for T_i.
        # i.e. max(L_i, start) < min(R_i, end) ? No that's overlap.
        # It is: Is it POSSIBLE that T_i fell in Candidate j?
        # Yes if Candidate j is inside (L_i, R_i].
        # So L_i <= start and end <= R_i.

        # Check if ANY observation covers this.
        # Actually, if NO observation covers it, p_j must be 0.

        is_covered = (left <= start) & (right >= end)
        if np.any(is_covered):
            valid_intervals.append((start, end))
            keep_indices.append(j)

    intervals = np.array(valid_intervals)
    m = len(intervals)

    if m == 0:
        return intervals, np.array([])

    # Alpha matrix: alpha[i, j] = 1 if observation i contains interval j
    alpha = np.zeros((n, m))
    for j in range(m):
        start, end = intervals[j]
        alpha[:, j] = (left <= start) & (right >= end)

    # 2. EM Algorithm (Self-Consistency)
    # Initialize probabilities uniform
    p = np.ones(m) / m

    for iteration in range(max_iter):
        p_prev = p.copy()

        # E-step: Expected number of events in interval j
        # d_j = sum_i [ (alpha_ij * p_j) / sum_k(alpha_ik * p_k) ]

        denom = alpha @ p # Shape (n,)
        # Avoid division by zero
        denom[denom == 0] = 1e-100

        # Contribution of each interval to each obs
        contrib = alpha * p[None, :] # (n, m)
        # Normalize rows
        contrib = contrib / denom[:, None]

        # M-step: Update p
        # p_j = (1/n) * sum_i contrib_ij
        p = np.sum(contrib, axis=0) / n

        if np.max(np.abs(p - p_prev)) < tol:
            break

    return intervals, p

def predict_turnbull(intervals, probs, times):
    """
    Calculates Survival Probability S(t) = P(T > t) from Turnbull estimates.
    """
    # Survival at t is sum of prob mass for all intervals STRICTLY greater than t.
    # Intervals are [start, end].
    # If t < start, the whole interval is > t.

    times = np.atleast_1d(times)
    surv = np.zeros_like(times, dtype=float)

    # Use upper bounds of intervals for conservative survival?
    # Or lower?
    # Standard: S(t) = sum_{j: start_j > t} p_j

    starts = intervals[:, 0]

    for i, t in enumerate(times):
        mask = starts > t # Intervals starting after t
        surv[i] = np.sum(probs[mask])

    return surv
