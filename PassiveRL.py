#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 15:10:59 2019

@author: hrishekesh.shinde
"""

import operator
import random
from collections import defaultdict
import matplotlib.pyplot as plt


class MDP:

    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text. Instead of P(s' | s, a) being a probability number for each
    state/state/action triplet, we instead have T(s, a) return a
    list of (p, s') pairs. We also keep track of the possible states,
    terminal states, and actions for each state. [page 646]"""

    def __init__(self, init, actlist, terminals, transitions=None, reward=None, states=None, gamma=0.9):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        # collect states from transitions table if not passed.
        self.states = states or self.get_states_from_transitions(transitions)
            
        self.init = init
        
        if isinstance(actlist, list):
            # if actlist is a list, all states have the same actions
            self.actlist = actlist

        elif isinstance(actlist, dict):
            # if actlist is a dict, different actions for each state
            self.actlist = actlist
        
        self.terminals = terminals
        self.transitions = transitions or {}
        if not self.transitions:
            print("Warning: Transition table is empty.")

        self.gamma = gamma

        self.reward = reward or {s: 0 for s in self.states}

        # self.check_consistency()

    def R(self, state):
        """Return a numeric reward for this state."""

        return self.reward[state]
    
orientations = EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
turns = LEFT, RIGHT = (+1, -1)


def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]


def turn_right(heading):
    return turn_heading(heading, RIGHT)


def turn_left(heading):
    return turn_heading(heading, LEFT)

def vector_add(a, b):
    """Component-wise addition of two vectors."""
    return tuple(map(operator.add, a, b))

def policy_evaluation(pi, U, mdp, k=20):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""

    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            U[s] = R(s) + gamma*sum(p*U[s1] for (p, s1) in T(s, pi[s]))
    return U


class GridMDP(MDP):

    """A two-dimensional grid MDP, as in [Figure 17.1]. All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state). Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""

    def __init__(self, grid, terminals, init=(0, 0), gamma=.9):
        grid.reverse()     # because we want row 0 on bottom, not on top
        reward = {}
        states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x]:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]
        self.states = states
        actlist = orientations
        transitions = {}
        for s in states:
            transitions[s] = {}
            for a in actlist:
                transitions[s][a] = self.calculate_T(s, a)
        MDP.__init__(self, init, actlist=actlist,
                     terminals=terminals, transitions=transitions, 
                     reward=reward, states=states, gamma=gamma)

    def T(self, state, action):
        return self.transitions[state][action] if action else [(0.0, state)]
    
    def calculate_T(self, state, action):
        if action:
            return [(0.8, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action)))]
        else:
            return [(0.0, state)]
        
    def go(self, state, direction):
        """Return the state that results from going in this direction."""

        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state
    
    
class PassiveDUEAgent:
    def __init__(self, pi, mdp):
        self.pi = pi
        self.mdp = mdp
        self.U = {}
        self.s = None
        self.a = None
        self.s_history = []
        self.r_history = []
        self.init = mdp.init
        
    def __call__(self, percept):
        s1, r1 = percept
        self.s_history.append(s1)
        self.r_history.append(r1)
        ##
        ##
        if s1 in self.mdp.terminals:
            self.s = self.a = None
        else:
            self.s, self.a = s1, self.pi[s1]
        return self.a
    
    def estimate_U(self):
        # this function can be called only if the MDP has reached a terminal state
        # it will also reset the mdp history
        assert self.a is None, 'MDP is not in terminal state'
        assert len(self.s_history) == len(self.r_history)
        # calculating the utilities based on the current iteration
        U2 = {s : [] for s in set(self.s_history)}
        for i in range(len(self.s_history)):
            s = self.s_history[i]
            U2[s] += [sum(self.r_history[i:])]
        U2 = {k : sum(v)/max(len(v), 1) for k, v in U2.items()}
        # resetting history
        self.s_history, self.r_history = [], []
        # setting the new utilities to the average of the previous 
        # iteration and this one
        for k in U2.keys():
            if k in self.U.keys():
                self.U[k] = (self.U[k] + U2[k]) /2
            else:
                self.U[k] = U2[k]
        return self.U
    
    
class PassiveADPAgent:
    class ModelMDP(MDP):
        """ Class for implementing modified Version of input MDP with
        an editable transition model P and a custom function T. """
        def __init__(self, init, actlist, terminals, gamma, states):
            super().__init__(init, actlist, terminals, states=states, gamma=gamma)
            nested_dict = lambda: defaultdict(nested_dict)
            # StackOverflow:whats-the-best-way-to-initialize-a-dict-of-dicts-in-python
            self.P = nested_dict()

        def T(self, s, a):
            """Return a list of tuples with probabilities for states
            based on the learnt model P."""
            return [(prob, res) for (res, prob) in self.P[(s, a)].items()]

    def __init__(self, pi, mdp):
        self.pi = pi
        self.mdp = PassiveADPAgent.ModelMDP(mdp.init, mdp.actlist,
                                            mdp.terminals, mdp.gamma, mdp.states)
        self.U = {}
        self.Nsa = defaultdict(int)
        self.Ns1_sa = defaultdict(int)
        self.s = None
        self.a = None
        self.visited = set()        # keeping track of visited states

    def __call__(self, percept):
        s1, r1 = percept
        mdp = self.mdp
        R, P, terminals, pi = mdp.reward, mdp.P, mdp.terminals, self.pi
        s, a, Nsa, Ns1_sa, U = self.s, self.a, self.Nsa, self.Ns1_sa, self.U

        if s1 not in self.visited:  # Reward is only known for visited state.
            U[s1] = R[s1] = r1
            self.visited.add(s1)
        if s is not None:
            Nsa[(s, a)] += 1
            Ns1_sa[(s1, s, a)] += 1
            # for each t such that Ns′|sa [t, s, a] is nonzero
            for t in [res for (res, state, act), freq in Ns1_sa.items()
                      if (state, act) == (s, a) and freq != 0]:
                P[(s, a)][t] = Ns1_sa[(t, s, a)] / Nsa[(s, a)]

        self.U = policy_evaluation(pi, U, mdp)
        self.Nsa, self.Ns1_sa = Nsa, Ns1_sa
        if s1 in terminals:
            self.s = self.a = None
        else:
            self.s, self.a = s1, self.pi[s1]
        return self.a


class PassiveTDAgent:
    def __init__(self, pi, mdp, alpha=None):

        self.pi = pi
        self.U = {s: 0. for s in mdp.states}
        self.Ns = {s: 0 for s in mdp.states}
        self.s = None
        self.a = None
        self.r = None
        self.gamma = mdp.gamma
        self.terminals = mdp.terminals

        if alpha:
            self.alpha = alpha
        else:
            self.alpha = lambda n: 1/(1+n)  # udacity video

    def __call__(self, percept):
        s1, r1 = self.update_state(percept)
        pi, U, Ns, s, r = self.pi, self.U, self.Ns, self.s, self.r
        alpha, gamma, terminals = self.alpha, self.gamma, self.terminals
        if not Ns[s1]:
            U[s1] = r1
        if s is not None:
            Ns[s] += 1
            U[s] += alpha(Ns[s]) * (r + gamma * U[s1] - U[s])
        if s1 in terminals:
            self.s = self.a = self.r = None
        else:
            self.s, self.a, self.r = s1, pi[s1], r1
        return self.a

    def update_state(self, percept):
        """To be overridden in most cases. The default case
        assumes the percept to be of type (state, reward)."""
        return percept
    
    
def run_single_trial(agent_program, mdp):
    """Execute trial for given agent_program
    and mdp. mdp should be an instance of subclass
    of mdp.MDP """

    def take_single_action(mdp, s, a):
        """
        Select outcome of taking action a
        in state s. Weighted Sampling.
        """
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for probability_state in mdp.T(s, a):
            probability, state = probability_state
            cumulative_probability += probability
            if x < cumulative_probability:
                break
        return state

    current_state = mdp.init
    while True:
        current_reward = mdp.R(current_state)
        percept = (current_state, current_reward)
        next_action = agent_program(percept)
        if next_action is None:
            break
        current_state = take_single_action(mdp, current_state, next_action)
        

def getUtilityEstimations(sequential_decision_environment, policy, point1, point2, size):
    DUEutilities_0_1 = []
    DUEutilities_2_2 = []
    ADPutilities_0_1 = []
    ADPutilities_2_2 = []
    TDutilities_0_1 = []
    TDutilities_2_2 = []
    DUEagent = PassiveDUEAgent(policy, sequential_decision_environment)
    for i in range(200):
        run_single_trial(DUEagent, sequential_decision_environment)
        DUEutilities_0_1.append(DUEagent.estimate_U()[point1])
        DUEutilities_2_2.append(DUEagent.estimate_U()[point2])
    
    
    ADPagent = PassiveADPAgent(policy, sequential_decision_environment)
    for i in range(200):
        run_single_trial(ADPagent, sequential_decision_environment)
        ADPutilities_0_1.append(ADPagent.U[point1])
        ADPutilities_2_2.append(ADPagent.U[point2])
        
    TDagent = PassiveTDAgent(policy, sequential_decision_environment, alpha = lambda n: 60./(59+n))
    for i in range(200):
        run_single_trial(TDagent,sequential_decision_environment)
        TDutilities_0_1.append(TDagent.U[point1])
        TDutilities_2_2.append(TDagent.U[point2])
        
    plt.plot(range(1, 201), DUEutilities_0_1, label="DUE agent")
    plt.plot(range(1, 201), ADPutilities_0_1, label = "ADP agent")
    plt.plot(range(1, 201), TDutilities_0_1, label = "TD agent")
    plt.axis([0, 200, 0, 1])
    plt.xlabel('Iteration number')
    plt.ylabel('Utility Value')
    plt.legend(loc='lower right')
    plt.title('Environment Size '+ size + 'Point '+str(point1))
    plt.savefig('Environment Size '+ size + 'Point '+str(point1)+'.png')
    plt.show()
    
    plt.plot(range(1, 201), DUEutilities_2_2, label="DUE agent")
    plt.plot(range(1, 201), ADPutilities_2_2, label = "ADP agent")
    plt.plot(range(1, 201), TDutilities_2_2, label = "TD agent")
    plt.axis([0, 200, 0, 1])
    plt.xlabel('Iteration number')
    plt.ylabel('Utility Value')
    plt.legend(loc='lower right')
    plt.title('Environment Size '+ size + 'Point '+str(point2))
    plt.savefig('Environment Size '+ size + 'Point '+str(point2)+'.png')
    plt.show()
    
    
    
sequential_decision_environment_1 = GridMDP([[-0.04, -0.04, -0.04, +1],
                                           [-0.04, None, -0.04, -1],
                                           [-0.04, -0.04, -0.04, -0.04]],
                                          terminals=[(3, 2), (3, 1)])

sequential_decision_environment_2 = GridMDP([[-0.04, -0.04, -0.04, +1],
                                           [-0.04, -0.04, -0.04, -1],
                                           [-0.04, -0.04, -0.04, -0.04]],
                                          terminals=[(3, 2), (3, 1)])

sequential_decision_environment_3 = GridMDP([[-0.04, -0.04, -0.04, -0.04, +1],
                                           [-0.04, None, -0.04, -0.04, -1],
                                           [-0.04, -0.04, -0.04,-0.04, -0.04],
                                           [-0.04, None, -0.04, None, -0.04],
                                           [-0.04, -0.04, -0.04, None, -0.04]],
                                          terminals=[(4, 4), (4, 3)])



# Action Directions
north = (0, 1)
south = (0,-1)
west = (-1, 0)
east = (1, 0)

policy_1_1 = {
    (0, 2): east,  (1, 2): east,  (2, 2): east,   (3, 2): None,
    (0, 1): north,                (2, 1): north,  (3, 1): None,
    (0, 0): north, (1, 0): west,  (2, 0): west,   (3, 0): west, 
}

policy_1_2 = {
    (0, 2): south,  (1, 2): east,  (2, 2): east,   (3, 2): None,
    (0, 1): south,                (2, 1): north,  (3, 1): None,
    (0, 0): east, (1, 0): east,   (2, 0): north,   (3, 0): west, 
}

policy_2_1 = {
    (0, 2): east,  (1, 2): east,  (2, 2): east,   (3, 2): None,
    (0, 1): north, (1, 1): north, (2, 1): north,  (3, 1): None,
    (0, 0): north, (1, 0): north,  (2, 0): west,   (3, 0): west, 
}

policy_3_1 = {
    (0, 4): east, (1, 4): east,  (2, 4): north,  (3, 4): east,  (4, 4): None, 
    (0, 3): north,               (2, 3): north,  (3, 3): north, (4, 3): None, 
    (0, 2): east, (1, 2): east,  (2, 2): north,  (3, 2): west,  (4, 2): west, 
    (0, 1): north,               (2, 1): north,                 (4, 1): north, 
    (0, 0): east, (1, 0): east,  (2, 0): north,                 (4, 0): north,  
}

getUtilityEstimations(sequential_decision_environment_1, policy_1_1, (0, 1), (2, 2), '4X3P1')
getUtilityEstimations(sequential_decision_environment_1, policy_1_2, (0, 0), (2, 2), '4X3P2')
getUtilityEstimations(sequential_decision_environment_2, policy_2_1, (0, 1), (2, 2), '4X3NoBlocksP1')
getUtilityEstimations(sequential_decision_environment_3, policy_3_1, (0, 0), (2, 2), '5X5P1')