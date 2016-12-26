import copy
import heapq
import sys
from datetime import datetime

INF = 1000000
output_label = ''

# Dummy function to test the algorithm without heuristic. #
def get_heuristic_cost(action):
    return 0

# Misplaced tiles heuristic: Admissible and consistent. #
def hamming_dh(position):
    n2 = position.size
    c = 0
    for kk in range(n2):
        if position.states[kk] != kk+1 and position.states[kk] != 0:
            c += 1
    return c

# Manhattan distance heuristic: Admissible and consistent. #
def manhattan_dh(position):
    n = position.width
    def row(x): return x / n
    def col(x): return x % n
    score = 0
    for idx, x in enumerate(position.states):
        if x == 0: continue
        ir,ic = row(idx), col(idx)
        xr,xc = row(x-1), col(x-1)
        score += abs(ir-xr) + abs(ic-xc)
    return score

#################################################################################
################# Generic class that gives next moves/actions ###################
#################################################################################

class nPuzzleSolver(object):
    def __init__(self, n, states=None):
        self.width = n
        self.size = n * n
        if states is None:
            self.states = [(i + 1) % self.size for i in range(self.size)]
        else:
            self.states = list(states)
        self.hsh = None
        self.prev_action = []

    def __hash__(self):
        if self.hsh is None:
            self.hsh = hash(tuple(self.states))
        return self.hsh

    def __repr__(self):
        return "nPuzzleSolver(%d, %s)" % (self.width, self.states)

    def __eq__(self, other):
        return self.states == other.states

    def __lt__(self, other):
        return ("%s" % (self.states) < "%s" % (other.states))

    # Generator used for generating the next possible moves. #
    def get_actions(self):
        zero_pos = self.states.index(0)
        def next_possible_action(i):
            j = zero_pos
            tmp = list(self.states)
            prev_action = tmp[i]
            tmp[i], tmp[j] = tmp[j], tmp[i]
            result = nPuzzleSolver(self.width, tmp)
            result.last_move = prev_action
            return result

        if zero_pos - self.width >= 0:
            yield next_possible_action(zero_pos-self.width)
        if zero_pos +self.width < self.size:
            yield next_possible_action(zero_pos+self.width)
        if zero_pos % self.width > 0:
            yield next_possible_action(zero_pos-1)
        if zero_pos % self.width < self.width-1:
            yield next_possible_action(zero_pos+1)
    
    # Labels the actions according to next move. #
    def label_action(self, parent):
        global output_label
        zero_pos = self.states.index(0)
        pzero_pos = parent.states.index(0)
        if zero_pos - pzero_pos == self.width:
            output_label += "D,"
        if zero_pos - pzero_pos == -1:
            output_label += "L,"
        if zero_pos - pzero_pos == 1:
            output_label += "R,"
        if zero_pos - pzero_pos == -1*self.width:
            output_label += "U,"
        return output_label

# Build the path from initial state to goal state. #
def construct_path(start, finish, parent):
    global output_label
    output_label = ''
    x = finish
    xs = [x]
    while x != start:
        x.label_action(parent[x])
        x = parent[x]
        xs.append(x)
    output_label = output_label[:-1]
    output_label = output_label[::-1]
    return (output_label)

#################################################################################
############################## RBFS Algorithm ###################################
#################################################################################

def sbuAiNPuzzleRBFS(istate, gstate):
    g = {}
    h = {}
    p = {}
    f = {}
    g[istate] = 0
    h[istate] = 0
    f[istate] = manhattan_dh(istate)
    p[istate] = None
    nse = 0

    # Recursive call for best first search. #
    def rbfs(cstate, f_limit):
        nonlocal nse
        if cstate == gstate:
            #print ("Depth: ",g[cstate])
            return 1, f[cstate]
        actions = cstate.get_actions()
        if not actions:
            return -1, INF 
        for action in actions:
            g[action] = g[cstate]+1
            h[action] = manhattan_dh(action)
            f[action] = max(g[action]+h[action], f[cstate])
        while(1):
            minimum = INF
            actions = cstate.get_actions()
            next_best=list(f.keys())[0]
            best=list(f.keys())[0]
            for key in actions:
                if f[key] <= minimum:
                    next_best = copy.deepcopy(best)
                    best = copy.deepcopy(key)
                    minimum = f[key]
            if f[best] > f_limit:
                return -1, f[best]
            p[best] = cstate
            nse += 1
            new_limit = min(f_limit, f[next_best])
            if best == next_best:
                new_limit = f_limit
            result, f[best] = rbfs(best, new_limit)
            if result != -1:
                return result, f[best]
    result,fcost=rbfs(istate, INF)
    if result==1:
        #print(fcost)
        construct_path(istate, gstate, p)

#################################################################################
############################### A* Algorithm ####################################
#################################################################################

def sbuAiNpuzzleAstar(istate, gstate):
    p = {}
    g = {}
    h = {}
    g[istate] = 0
    p[istate] = None
    h[istate] = 0
    frontier = []
    nse = 0
   
    heapq.heappush(frontier, (0, istate))
    while frontier:
        nse += 1
        f, cstate = heapq.heappop(frontier)
        if cstate == gstate:
            #print ("depth:", g[cstate])
            #print ("No of explored nodes:",nse)
            construct_path(istate, gstate, p)
            return 0
        actions = cstate.get_actions()
        cost = g[cstate]
        for action in actions:
            if action not in g or g[action] > cost + 1:
                g[action] = cost + 1
                if action not in h:
                    h[action] = manhattan_dh(action)
                p[action] = cstate
                heapq.heappush(frontier, (g[action] + h[action], action))
    return -1

def main(alg, puz, input_file, output_file):
    tstart = datetime.now()
    input_list = []
    fh1 = open(input_file)
    fh2 = open(output_file, 'w')
    lines = fh1.readlines()
    for line in lines:
        t = line.strip().split(',')
        for i in t:
            input_list.append(i)
    istate = [(int(x) if x else 0) for x in input_list]
    pos = nPuzzleSolver(puz, istate)
    ret=-1
    if alg == 1:
        ret = sbuAiNpuzzleAstar(pos, nPuzzleSolver(puz))
    elif alg == 2:
        ret = sbuAiNPuzzleRBFS(pos, nPuzzleSolver(puz))
    tend = datetime.now()
    
    if ret == -1:
        print ("Solution not found")
    else:
        fh2.write(output_label)
        #print ((tend-tstart).microseconds/1000)
    fh1.close()
    fh2.close()
    return ()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print ("The number of arguments must be at least 4")
        sys.exit()
    alg = int(sys.argv[1])
    puz = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]
    main(alg, puz, input_file, output_file)
