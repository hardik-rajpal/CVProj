
from itertools import chain, groupby, product
import random

import numpy as np
from scipy.optimize import linprog

from scipy.spatial.distance import mahalanobis

# Number of constraints per oriented pairwise match

NUM_CONSTRAINTS = 2

# Threshold after which to reject a match from the LP solution

MATCH_REJECTION_THRESHOLD = 10e-5

# Delta functions for inequality constraints

DELTA_X = [0, -1, 0, 1]
DELTA_Y = [1, 0, -1, 0]

# Number of rotations to align pieces for MGC calculation

MGC_NUM_ROTATIONS = [3, 0, 1, 2]

# Number of possible orientations for two pieces

MGC_NUM_ORIENTATIONS = len(MGC_NUM_ROTATIONS)


def compute_weights(pairwise_matches, mgc_distances):
    num_images = max((i for i, _, _ in pairwise_matches)) + 1
    index_set = frozenset(range(num_images))
    weights = {}
    for i, j, o in pairwise_matches:
        min_row = min(mgc_distances[k, j, o] for k in index_set - {i})
        min_col = min(mgc_distances[i, k, o] for k in index_set - {j})
        weights[i, j, o] = min(min_row, min_col) / mgc_distances[i, j, o]
    return weights


def i_o_key(l1):
    return l1[0], l1[-1]
def compute_active_selection(pairwise_matches, mgc_distances):

    active_selection = []
    for _, group in groupby(sorted(pairwise_matches, key=i_o_key), i_o_key):
        entries = list(group)
        distances = np.array([mgc_distances[entry] for entry in entries])
        lowest_index = np.argmin(distances)
        entry = entries[lowest_index]
        active_selection.append(entry)
    return active_selection


def compute_solution(active_selection, weights, maxiter=None):
    def sorted_by_i_and_o(active_selection):
        return sorted(active_selection, key=i_o_key)

    def row_index(i, o):
        return (MGC_NUM_ORIENTATIONS * NUM_CONSTRAINTS * i) + \
               (NUM_CONSTRAINTS * o)

    # Sort active selection by i and o. The resulting order allows for
    # simplifications on the inequality constraint matrix A_ub.

    n = int(len(active_selection) / MGC_NUM_ORIENTATIONS)
    sorted_a = sorted_by_i_and_o(active_selection)

    # Construct inequality constraints matrix A_ub, given as follows:
    #    A_ub = | H1 | 0  | X |
    #           | 0  | H2 | Y |,
    # and where X = Y and H1 = H2 (constraints are identical for X and Y).

    # Recall that the inequality constraints are given as:
    #
    #   h_ijo >= x_i  - x_j - delta_o^x
    #   h_ijo >= -x_i + x_j + delta_o^x
    #
    # -delta_o^x and delta_o^x are the only constants and will be assigned to
    # the upper bounds of the inequality constraints. Therefore, the
    # constraints above are rewritten as follows:
    #
    #  -x_i + x_j + h_ijo >= -delta_o^x
    #   x_i - x_j + h_ijo >=  delta_o^x
    #
    # Rewriting than greater-than-or-equal signs to smaller-than-or-equal gives:
    #
    #   x_i - x_j - h_ijo <=  delta_o^x
    #  -x_i + x_j - h_ijo <= -delta_o^x
    #
    # Given these constraints, submatrices H1 and H2 are composed of two -1's
    # in each column:

    h_base = np.array([-1] * NUM_CONSTRAINTS + [0] * (MGC_NUM_ORIENTATIONS *
                                                      NUM_CONSTRAINTS *
                                                      n - NUM_CONSTRAINTS))
    H = np.array([np.roll(h_base, k) for k in range(
            0, MGC_NUM_ORIENTATIONS * NUM_CONSTRAINTS * n, NUM_CONSTRAINTS)]).T

    xi_base = np.array([1, -1] * MGC_NUM_ORIENTATIONS + [0] *
                       (MGC_NUM_ORIENTATIONS * NUM_CONSTRAINTS) *
                       (n - 1))
    Xi = np.array([np.roll(xi_base, k) for k in
                   range(0, MGC_NUM_ORIENTATIONS * NUM_CONSTRAINTS * n,
                         NUM_CONSTRAINTS * MGC_NUM_ORIENTATIONS)]).T
    Xj = np.zeros(Xi.shape, dtype=np.int32)
    for i, j, o in sorted_a:
        r = row_index(i, o)
        Xj[r:r + 2, j] = [-1, 1]
    X = Xi + Xj
    h, w = H.shape
    Z_h = np.zeros((h, w), dtype=np.int32)
    Z_x = np.zeros((h, n), dtype=np.int32)
    A_ub = np.vstack([H, Z_h])
    A_ub = np.hstack([A_ub, np.vstack([Z_h, H])])
    A_ub = np.hstack([A_ub, np.vstack([X, Z_x])])
    A_ub = np.hstack([A_ub, np.vstack([Z_x, X])])
    
    b_x = list(chain.from_iterable([[DELTA_X[o], -DELTA_X[o]] for (_, _, o) in sorted_a]))
    b_y = list(chain.from_iterable([[DELTA_Y[o], -DELTA_Y[o]] for (_, _, o) in sorted_a]))
    b_ub = np.array(b_x + b_y)

    c_base = [weights[_] for _ in sorted_a]
    c = np.array(c_base * NUM_CONSTRAINTS + ([0] * NUM_CONSTRAINTS * n))


    options = {'maxiter': maxiter} if maxiter else {}
    solution = linprog(c, A_ub, b_ub, options=options)

    if not solution.success:
        if solution.message == 'Iteration limit reached.':
            raise ValueError('iteration limit reached, try increasing the ' +
                             'number of max iterations')
        else:
            raise ValueError('unable to find solution to LP: {}'.format(
                solution.message))

    xy = solution.x[-n * 2:]
    return xy[:n], xy[n:]


def compute_rejected_matches(active_selection, x, y):
    rejected_matches = set()
    for i, j, o in active_selection:
        if abs(x[i] - x[j] - DELTA_X[o]) > MATCH_REJECTION_THRESHOLD:
            rejected_matches.add((i, j, o))
        if abs(y[i] - y[j] - DELTA_Y[o]) > MATCH_REJECTION_THRESHOLD:
            rejected_matches.add((i, j, o))
    return rejected_matches


def solve(images, maxiter=None, random_seed=None):
    if random_seed:
        random.seed(random_seed)

    pairwise_matches = initial_pairwise_matches(len(images))
    mgc_distances = compute_mgc_distances(images, pairwise_matches)
    weights = compute_weights(pairwise_matches, mgc_distances)
    active_selection = compute_active_selection(pairwise_matches, mgc_distances)
    x, y = compute_solution(active_selection, weights, maxiter)

    old_x, old_y = None, None

    while (old_x is None and old_y is None) or not \
            (np.array_equal(old_x, x) and np.array_equal(old_y, y)):
        rejected_matches = compute_rejected_matches(active_selection, x, y)
        pairwise_matches = list(set(pairwise_matches) - rejected_matches)
        active_selection = compute_active_selection(pairwise_matches,
                                                    mgc_distances)

        old_x, old_y = x, y
        x, y = compute_solution(active_selection, weights, maxiter)

    return x, y


def compute_mgc_distances(images, pairwise_matches):
    return {(i, j, o): mgc(images[i], images[j], o) for
            i, j, o in pairwise_matches}


def mgc(image1, image2, orientation):

    num_rotations = MGC_NUM_ROTATIONS[orientation]

    image1_signed = np.rot90(image1, num_rotations).astype(np.int16)
    image2_signed = np.rot90(image2, num_rotations).astype(np.int16)

    g_i_l = image1_signed[:, -1] - image1_signed[:, -2]
    mu = g_i_l.mean(axis=0)

    s = np.cov(g_i_l.T) + np.eye(3) * 10e-6

    g_ij_lr = image2_signed[:, 1] - image1_signed[:, -1]

    return sum(mahalanobis(row, mu, np.linalg.inv(s)) for row in g_ij_lr)

def initial_pairwise_matches(num_images):
    return list(product(range(num_images), range(num_images),range(MGC_NUM_ORIENTATIONS)))
