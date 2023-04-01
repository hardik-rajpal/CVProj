
from itertools import chain, groupby, product
import random
from scipy.spatial.distance import mahalanobis
from scipy.optimize import linprog
import numpy as np
CONSTRAINTS = 2
THRES_REJ = 10e-5
DX = [0, -1, 0, 1];DY = [1, 0, -1, 0]
MGC_NUM_ROTATIONS = [3, 0, 1, 2]
MGC_NUM_ORIENTATIONS = len(MGC_NUM_ROTATIONS)
CONSTR_ORIENT_PROD = MGC_NUM_ORIENTATIONS*CONSTRAINTS
class Solver:
    def computeWeights(self,pairwiseMatches, mgcDistances, numImages):
        indexSet = frozenset(range(numImages))
        weights = {}
        for i, j, o in pairwiseMatches:
            minRow = min(mgcDistances[k, j, o] for k in indexSet - {i})
            minCol = min(mgcDistances[i, k, o] for k in indexSet - {j})
            weights[i, j, o] = min(minRow, minCol) / mgcDistances[i, j, o]
        return weights
    def iOKey(l1):
        return l1[0], l1[-1]
    def sortedByIAndO(activeSelection):
        return sorted(activeSelection, key=Solver.iOKey)
    def computeActiveSelection(self,pairwiseMatches, mgcDistances):
        activeSelection = []
        for _, group in groupby(Solver.sortedByIAndO(pairwiseMatches), Solver.iOKey):
            entries = list(group)
            distances = np.array([mgcDistances[entry] for entry in entries])
            lowestIndex = np.argmin(distances)
            entry = entries[lowestIndex]
            activeSelection.append(entry)
        return activeSelection
    def computeSolution(self,activeSelection, weights, maxiter=None):
        def rowIndex(i, o):
            return (CONSTR_ORIENT_PROD * i) + (CONSTRAINTS * o)
        n = int(len(activeSelection) / MGC_NUM_ORIENTATIONS)
        sortedA = Solver.sortedByIAndO(activeSelection)
        hBase = np.array([-1] * CONSTRAINTS + [0] * (CONSTR_ORIENT_PROD*n - CONSTRAINTS))
        H = np.array([np.roll(hBase, k) for k in range(0, CONSTR_ORIENT_PROD * n, CONSTRAINTS)]).T
        xiBase = np.array([1, -1] * MGC_NUM_ORIENTATIONS + [0] *(CONSTR_ORIENT_PROD)*(n - 1))
        Xi = np.array([np.roll(xiBase, k) for k in range(0, CONSTR_ORIENT_PROD * n,CONSTRAINTS * MGC_NUM_ORIENTATIONS)]).T
        Xj = np.zeros(Xi.shape, dtype=np.int32)
        for i, j, o in sortedA:
            r = rowIndex(i, o)
            Xj[r:r + 2, j] = [-1, 1]
        X = Xi + Xj
        h, w = H.shape
        ZH = np.zeros((h, w), dtype=np.int32)
        ZX = np.zeros((h, n), dtype=np.int32)
        AUb = np.vstack([H, ZH])
        AUb = np.hstack([AUb, np.vstack([ZH, H])])
        AUb = np.hstack([AUb, np.vstack([X, ZX])])
        AUb = np.hstack([AUb, np.vstack([ZX, X])])
        
        bX = list(chain.from_iterable([[DX[o], -DX[o]] for (_, _, o) in sortedA]))
        bY = list(chain.from_iterable([[DY[o], -DY[o]] for (_, _, o) in sortedA]))
        bUb = np.array(bX + bY)
        cBase = [weights[_] for _ in sortedA]
        c = np.array(cBase * CONSTRAINTS + ([0] * CONSTRAINTS * n))
        options = {'maxiter': maxiter} if maxiter else {}
        solution = linprog(c, AUb, bUb, options=options)
        if not solution.success:
            if solution.message == 'Iteration limit reached.':
                raise ValueError('maxiters reached')
            else:
                raise ValueError('no solution: {}'.format(
                    solution.message))
        xy = solution.x[-n * 2:]
        return xy[:n], xy[n:]
    def computeRejectedMatches(self,activeSelection, x, y):
        rejectedMatches = set()
        for i, j, o in activeSelection:
            if abs(x[i] - x[j] - DX[o]) > THRES_REJ:
                rejectedMatches.add((i, j, o))
            if abs(y[i] - y[j] - DY[o]) > THRES_REJ:
                rejectedMatches.add((i, j, o))
        return rejectedMatches
    def computeMgcDistances(self,images, pairwiseMatches):
        return {(i, j, o): Solver.mgc(images[i], images[j], o) for
                i, j, o in pairwiseMatches}
    def mgc(image1, image2, orientation):
        numRotations = MGC_NUM_ROTATIONS[orientation]
        image1Signed = np.rot90(image1, numRotations).astype(np.int16)
        image2Signed = np.rot90(image2, numRotations).astype(np.int16)
        gIL = image1Signed[:, -1] - image1Signed[:, -2]
        mu = gIL.mean(axis=0)
        s = np.cov(gIL.T) + np.eye(3) * 10e-6
        gIjLr = image2Signed[:, 1] - image1Signed[:, -1]
        return sum(mahalanobis(row, mu, np.linalg.inv(s)) for row in gIjLr)
    def initialPairwiseMatches(self,numImages):
        x,y,z = np.meshgrid(np.arange(numImages),np.arange(numImages),np.arange(MGC_NUM_ORIENTATIONS))
        ans = np.stack([x.flatten(),y.flatten(),z.flatten()],axis=1)
        return [tuple(x) for x in ans.tolist()]
    def solve(self,images, maxiter=None, randomSeed=None):
        if randomSeed:
            random.seed(randomSeed)
        pairwiseMatches = self.initialPairwiseMatches(len(images))
        #initialize space of all possible matchings.👆
        mgcDistances = self.computeMgcDistances(images, pairwiseMatches)
        #dictionary from match tuple to distance.
        weights = self.computeWeights(pairwiseMatches, mgcDistances, len(images))
        #weights for ...?
        activeSelection = self.computeActiveSelection(pairwiseMatches, mgcDistances)
        x, y = self.computeSolution(activeSelection, weights, maxiter)
        oldX, oldY = None, None
        while (oldX is None and oldY is None) or not (np.array_equal(oldX, x) and np.array_equal(oldY, y)):
            rejectedMatches = self.computeRejectedMatches(activeSelection, x, y)
            pairwiseMatches = list(set(pairwiseMatches) - rejectedMatches)
            activeSelection = self.computeActiveSelection(pairwiseMatches,mgcDistances)
            oldX, oldY = x, y
            x, y = self.computeSolution(activeSelection, weights, maxiter)
        return x, y