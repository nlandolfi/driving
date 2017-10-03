import math
import numpy as np
piby4 = math.pi/4

# pair is a struct of (state, features)

def feats(s, t):
    delta = t - s
    delta_norm = delta / np.linalg.norm(delta)

    return np.array([
        delta[0],
        delta[1],
        delta[2],
        delta_norm[0],
        delta_norm[1],
        delta_norm[2],
    ])

def isNearby(myState, delta, state):
    p = np.array([myState[0], myState[1]])
    q = np.array([state[0], state[1]])
    return np.linalg.norm(p - q) < delta

def isLane(myState, state):
    xL = myState[0] - 0.075
    xR = myState[0] + 0.075
    x = state[0]
    return x > xL and x < xR

def nearby(myState, delta, pairs):
    found = []
    p = np.array([myState[0], myState[1]])
    for pair in pairs:
        state = pair[1]
        q = np.array([state[0], state[1]])
        if np.linalg.norm(p - q) < delta:
            found.append(pair[1])

    return found

def lane(myState, pairs):
    found = []
    xL = myState[0] - 0.075
    xR = myState[0] + 0.075
    for pair in pairs:
        state = pairs[0]
        x = state[0]
        if x > xL and x < xR:
            found.append(pair[1])
    return np.array(found)

def all(pairs):
    return pairs

# takes an N by D matrix
# where the rows are the samples
# and the columns are the feature space
#
# returns a 2 by D matrix
def lambdaMu(samples):
    if len(samples) == 0 or len(samples) == 1 or len(samples) == 2:
        return np.stack([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])

    std = np.std(samples, axis=0)


    #lam = -np.exp(-std)
    lam = -1/std

    for (i, l) in enumerate(lam):
        if np.isinf(l):
            lam[i] = 0

    return np.stack([lam, np.mean(samples, axis=0)])

def samples(current, others):
    posSamples, laneSamples, roadSamples = [], [], []

    for o in others:
        for i in range(0, len(o)-2):
            s, t = o[i], o[i+1]
            fs = feats(s, t)

            if isNearby(current, 0.05, s):
                posSamples.append(fs)

            if isLane(current, s):
                laneSamples.append(fs)

            roadSamples.append(fs)

    pos, lane, road = np.array([]), np.array([]), np.array([])

    if len(posSamples) > 0:
        pos = np.stack(posSamples)
    if len(laneSamples) > 0:
        lane = np.stack(laneSamples)
    if len(roadSamples) > 0:
        road = np.stack(roadSamples)

    return pos, lane, road

def variances(current, hist):
    others = hist[1:len(hist)-1]
    print("%s others" % (len(others)))

    # partition the samples
    pos, lane, road = samples(current, others)

    print("Position: %s" % (len(pos)))
    print("Lane    : %s" % (len(lane)))
    print("Road    : %s" % (len(road)))

    print(pos)

    return lambdaMu(pos), lambdaMu(lane), lambdaMu(road)

# Running Stats {{{

# Compute streaming empirical first and second moments.
class RunningStats(object):
    def __init__(self):
        self.xs = []

    # process a new value
    def ingest(self, x):
        self.xs.append(x)

    # get sample mean
    def mean(self):
        if len(self.xs) == 0.:
            return 0.

        return np.mean(self.xs)

    # get sample variance
    def var(self):
        if len(self.xs) == 0. or len(self.xs) == 1.:
            return 0.

        return np.var(self.xs)

    def std(self):
        return np.sqrt(self.var())

    def lam(self):
        s = self.std()
        if s == 0.:
            return 0.0

        return -1./s

    def num(self):
        return len(self.xs)

# }}}

# | 1 | 2 | 3 |
def laneNumber(state):
    x = state[0]
    if x < -0.065:
        return 1 # left

    if x > 0.065:
        return 3 # right

    return 2 # middle

gridSize = 1

def gridSquare(state):
    x, y = state[0], state[1]

    x = int(x * 100)
    y = int(y * 100)

    return (x, y)

noStats = [(0., 0.), (0., 0.), (0., 0.)]

class CohesiveStats(object):
    def reset(self):
        self.carsLastState = {}

        # a map from grid squares to stats for that square
        self.posFeatures  = {}

        # Lane features is a list of RunningStats
        # for each of the three lanes
        self.laneFeatures = {
            1: [RunningStats(), RunningStats(), RunningStats()],
            2: [RunningStats(), RunningStats(), RunningStats()],
            3: [RunningStats(), RunningStats(), RunningStats()],
        }

        self.roadFeatures = [RunningStats(), RunningStats(), RunningStats()]

    def __init__(self, *args, **vargs):
        self.reset()

    def processNewState(self, carid, state):
        if carid not in self.carsLastState:
            self.carsLastState[carid] = state
            return

        lastState = self.carsLastState[carid]
        self.carsLastState[carid] = state

        self.processFeatures(lastState, state)

    def relevantGridSquares(self, state):
        center = gridSquare(state)
        x, y = center[0], center[1]

        squares = []

        for i in range(-3, 4):
            for j in range(-3, 4):
                squares.append((x + i*gridSize, y + i*gridSize))

        print(squares)
        return squares

    def processFeatures(self, start, end):
        delta = np.asarray(end) - np.asarray(start)

        # Positions
        for square in self.relevantGridSquares(start):
            if square not in self.posFeatures:
                self.posFeatures[square] = [
                    RunningStats(), RunningStats(), RunningStats(),
                ]

            pos = self.posFeatures[square]
            pos[0].ingest(delta[0])
            pos[1].ingest(delta[1])
            pos[2].ingest(delta[2])

        # Lanes
        lane = self.laneFeatures[laneNumber(start)]
        lane[0].ingest(delta[0])
        lane[1].ingest(delta[1])
        lane[2].ingest(delta[2])

        # Road
        self.roadFeatures[0].ingest(delta[0])
        self.roadFeatures[1].ingest(delta[1])
        self.roadFeatures[2].ingest(delta[2])

    def posStats(self, state):
        square = gridSquare(state)
        if square not in self.posFeatures:
            return noStats

        pos = self.posFeatures[square]

        return [(f.mean(), f.lam()) for f in pos]

    def laneStats(self, state):
        laneNum = laneNumber(state)
        lane = self.laneFeatures[laneNum]
        return [(f.mean(), f.lam()) for f in lane]

    def roadStats(self, state):
        return [(f.mean(),f.lam()) for f in self.roadFeatures]

if __name__ == '__main__':
    rs = RunningStats()
    rs.ingest(1)
    rs.ingest(2)
    rs.ingest(3)

    print(rs.mean())
    print(rs.std())
    print(np.mean([1, 2, 3]))
    print(np.std([1, 2, 3]))

    print(gridSquare([0, 0, 0, 0]))
    print(gridSquare([-.13, 0.1, 0, 0]))

    cs = CohesiveStats()
    cs.processNewState(0, [0, 0, 0, 0])
    cs.processNewState(0, [0, .05, 0, 0])
    cs.processNewState(0, [0, .10, 0, 0])
    cs.processNewState(0, [0, .15, 0, 0])

    cs.processNewState(1, [1, 0, 0, 0])
    cs.processNewState(1, [1, .05, 0, 0])
    cs.processNewState(1, [1, .10, 0, 0])
    cs.processNewState(1, [1, .15, 0, 0])

    print(cs.roadStats([0, 0, 0, 0]))
    print(cs.roadFeatures[0].num())
