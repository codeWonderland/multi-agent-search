"""Multi-agent search agents.

Author: Alice Easter
Class: CSI-480-01
Assignment: PA 3 -- Multi-Agent Search
Due Date: October 15, 2018 11:59 PM

Certification of Authenticity:
I certify that this is entirely my own work, except where I have given
fully-documented references to the work of others. I understand the definition
and consequences of plagiarism and acknowledge that the assessor of this
assignment may, for the purpose of assessing this assignment:
- Reproduce this assignment and provide a copy to another member of academic
- staff; and/or Communicate a copy of this assignment to a plagiarism checking
- service (which may then retain a copy of this assignment on its database for
- the purpose of future plagiarism checking)

----------------------
Champlain College CSI-480, Fall 2018
The following code was adapted by Joshua Auerbach (jauerbach@champlain.edu)
from the UC Berkeley Pacman Projects (see license and attribution below).

----------------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""
import random
from util import manhattan_distance, lookup
from functools import reduce
from game import Agent, Directions
from collections import defaultdict


class ReflexAgent(Agent):
    """An agent that chooses an action reflexively based on a state evaluation.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def __init__(self):
        """Init variables and call super __init__."""
        super().__init__()

        self.last_dir = None
        self.same_dir_count = 0

    def get_action(self, game_state):
        """Choose among the best options according to the evaluation function.

        You do not need to change this method, but you're welcome to.

        Just like in the previous project, get_action takes a GameState and
        returns some Directions.X for some X in the set
        {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = []
        distances = []

        for action in legal_moves:
            tmp_eval = self.evaluation_function(game_state, action)

            scores.append(tmp_eval[0])
            distances.append(tmp_eval[1])

        # determine options with best score
        best_score = max(scores)
        best_score_indices = [index for index in range(len(scores))
                              if scores[index] == best_score]

        # choose option with best score and closest distance to food
        best_distance = float("inf")
        best_distance_indices = best_score_indices
        for index in best_score_indices:
            # if the distance is shorter, we update the
            # best_distance and select that option
            if distances[index] < best_distance:
                best_distance = distances[index]

                best_distance_indices = [index]

            elif distances[index] is best_distance:
                best_distance_indices.append(index)

        chosen_index = None

        for index in best_distance_indices:
            if legal_moves[index] is self.last_dir and \
                    self.same_dir_count < 5:
                chosen_index = index

        # if we still have multiple, we randomly choose
        while chosen_index is None or \
                legal_moves[chosen_index] is Directions.STOP:
            chosen_index = random.choice(best_distance_indices)

            self.same_dir_count = 0

        # update our direction for next time
        chosen_dir = legal_moves[chosen_index]
        self.last_dir = chosen_dir
        self.same_dir_count += 1

        return chosen_dir

    @staticmethod
    def evaluation_function(current_game_state, action):
        """Return evaluation (number) based on game state and proposed action.

        *** Design a better evaluation function here. ***

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are
        better.

        The code samples below extracts some useful information from the state,
        like the remaining food (new_food) and Pacman position after moving
        (new_pos). new_scared_times holds the number of moves that each ghost
        will remain scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = \
            current_game_state.generate_pacman_successor(action)

        new_pos = successor_game_state.get_pacman_position()

        new_ghost_states = successor_game_state.get_ghost_states()
        ghosts_pos = [ghost_state.get_position()
                      for ghost_state in new_ghost_states]

        new_capsules = successor_game_state.get_capsules()

        score = 0

        # excessive distance
        food_dist = float("inf")

        if new_pos in ghosts_pos:
            # score = 0
            # food_dist = 1000
            pass

        elif successor_game_state.get_num_food() is 0:
            # space is last piece of food
            score = 5
            food_dist = 0

        else:
            for ghost in ghosts_pos:
                dist = manhattan_distance(new_pos, ghost)

                if dist == 1.0:
                    if new_pos in new_capsules:
                        # space is a power capsule and ghost is nearby
                        score = 4

                    else:
                        # space is next to ghost
                        score = 1

                elif current_game_state.has_food(new_pos[0], new_pos[1]):
                    # food in next space
                    score = 3
                else:
                    # nothing in space
                    score = 2

            for food_pos in current_game_state.get_food().as_list():
                tmp_dist = manhattan_distance(new_pos, food_pos)
                food_dist = min(food_dist, tmp_dist)

        return score, food_dist


def score_evaluation_function(current_game_state):
    """Return the score of the current game state.

    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()


class MultiAgentSearchAgent(Agent):
    """Common elements to all multi-agent searchers.

    Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        """Create agent given an evaluation function and search depth."""
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluation_function = lookup(eval_fn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """Your minimax agent (question 2)."""

    def get_action(self, game_state):
        """Return the minimax action from the given game_state.

        Run minimax to max depth self.depth and evaluate "leaf" nodes using
        self.evaluation_function.
        """
        max_value = float("-inf")
        action = Directions.STOP

        # we know that the 0 depth is pacman
        # so we check to see which action has the most value
        for pac_action in game_state.get_legal_actions(0):
            new_state = game_state.generate_successor(0, pac_action)

            # check value of entire branch
            value = self.value(new_state, 1, 1)

            # whichever action is most valuable gets selected for return
            if value > max_value:
                max_value = value
                action = pac_action

        return action

    def value(self, game_state, current_depth, agent_index):
        """Return value of node based off of subtree.

        determines if the current depth is friend or foe then
        gets appropriate value for either
        """
        # make sure that we haven't gone through all our agents
        if agent_index < game_state.get_num_agents():
            # grab all eligible actions
            actions = game_state.get_legal_actions(agent_index)

            # if terminal node, we return its value
            if not actions:
                return self.evaluation_function(game_state)

            optimal_value = float("-inf") if agent_index is 0 else float("inf")

            for action in actions:
                # grab states for current action
                new_state = game_state.generate_successor(agent_index, action)

                # if the agent is foe we min
                if agent_index:
                    optimal_value = min(
                        optimal_value,
                        self.value(new_state, current_depth, agent_index + 1)
                    )

                # otherwise we max for pacman
                else:
                    optimal_value = max(
                        optimal_value,
                        self.value(new_state, current_depth, agent_index + 1)
                    )

            return optimal_value

        # no more agents left to plan for on this depth
        # so we check to see if we've reached the max depth
        elif current_depth is self.depth or \
                game_state.is_win() or \
                game_state.is_lose():
                return self.evaluation_function(game_state)

        # otherwise we just move on to the next layer
        else:
            return self.value(game_state, current_depth + 1, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """Your minimax agent with alpha-beta pruning (question 3)."""

    def get_action(self, game_state):
        """Return the minimax action from the given game_state.

        Run minimax with alpha-beta pruning to max depth self.depth and
        evaluate "leaf" nodes using self.evaluation_function.
        """
        max_value = float("-inf")
        action = Directions.STOP

        # establish init alpha / beta values
        alpha = float("-inf")
        beta = float("inf")

        # we know that the 0 depth is pacman
        # so we check to see which action has the most value
        for pac_action in game_state.get_legal_actions(0):
            new_state = game_state.generate_successor(0, pac_action)

            # check value of entire branch
            value = self.value(new_state, 1, 1, alpha, beta)

            # whichever action is most valuable gets selected for return
            if value > max_value:
                max_value = value
                action = pac_action

            # implement alpha check
            if value > alpha:
                action = pac_action
                alpha = value

        return action

    def value(self, game_state, current_depth, agent_index, alpha, beta):
        """Return value of node based off of subtree.

        determines if the current depth is friend or foe then
        gets appropriate value for either
        """
        # make sure that we haven't gone through all our agents
        if agent_index < game_state.get_num_agents():
            # grab all eligible actions
            actions = game_state.get_legal_actions(agent_index)

            # if terminal node, we return its value
            if not actions:
                return self.evaluation_function(game_state)

            optimal_value = float("-inf") if agent_index is 0 else float("inf")

            for action in actions:
                # grab states for current action
                new_state = game_state.generate_successor(agent_index, action)

                # if the agent is foe we min
                if agent_index:
                    optimal_value = min(
                        optimal_value,
                        self.value(
                            new_state,
                            current_depth,
                            agent_index + 1,
                            alpha,
                            beta
                        )
                    )

                    # implement alpha check
                    if optimal_value < alpha:
                        return optimal_value

                    # update beta if necessary
                    beta = min(beta, optimal_value)

                # otherwise we max for pacman
                else:
                    optimal_value = max(
                        optimal_value,
                        self.value(
                            new_state,
                            current_depth,
                            agent_index + 1,
                            alpha,
                            beta
                        )
                    )

                    # implement beta check
                    if optimal_value > beta:
                        return optimal_value

                    # update alpha if necessary
                    alpha = max(alpha, optimal_value)

            return optimal_value

        # no more agents left to plan for on this depth
        # so we check to see if we've reached the max depth
        elif current_depth is self.depth or \
                game_state.is_win() or \
                game_state.is_lose():
            return self.evaluation_function(game_state)

        # otherwise we just move on to the next layer
        else:
            return self.value(game_state, current_depth + 1, 0, alpha, beta)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """Your expectimax agent (question 4)."""

    def get_action(self, game_state):
        """Return the expectimax action from the given game_state.

        Run expectimax to max depth self.depth and evaluate "leaf" nodes using
        self.evaluation_function.

        All ghosts are modeled as choosing uniformly at random from their
        legal moves.
        """
        max_value = float("-inf")
        action = Directions.STOP

        # we know that the 0 depth is pacman
        # so we check to see which action has the most value
        for pac_action in game_state.get_legal_actions(0):
            new_state = game_state.generate_successor(0, pac_action)

            # check value of entire branch
            action_value = self.value(new_state, 1, 1)

            # whichever action is most valuable gets selected for return
            if action_value > max_value:
                max_value = action_value
                action = pac_action

        return action

    def value(self, game_state, current_depth, agent_index):
        """Return value of node based off of subtree.

        determines if the current depth is friend or foe then
        gets appropriate value for either
        """
        # make sure that we haven't gone through all our agents
        if agent_index < game_state.get_num_agents():
            # grab all eligible actions
            actions = game_state.get_legal_actions(agent_index)

            # if terminal node, we return its value
            if not actions:
                return self.evaluation_function(game_state)

            values = []

            for action in actions:
                # grab states for current action
                new_state = game_state.generate_successor(agent_index, action)

                # add value to action to values array
                values.append(
                    self.value(
                        new_state,
                        current_depth,
                        agent_index + 1
                    )
                )

            # for coms we take average
            if agent_index:
                optimal_value = sum(values) / len(values)

            # for pacman we still max
            else:
                optimal_value = max(values)

            return optimal_value

        # no more agents left to plan for on this depth
        # so we check to see if we've reached the max depth
        elif current_depth is self.depth or \
                game_state.is_win() or \
                game_state.is_lose():
            return self.evaluation_function(game_state)

        # otherwise we just move on to the next layer
        else:
            return self.value(game_state, current_depth + 1, 0)


def better_evaluation_function(current_game_state):
    """Your awesome evaluation function (question 5).

    Description: Since we have no comparative data other than
    the current game score, we make that a centralized part of
    our answer. From there, we want the scenarios where there are
    a bunch of scared ghosts to be prioritized, as well as any
    scenario where there is food nearby and no ghosts
    """
    # grab current pacman position
    pac_pos = current_game_state.get_pacman_position()

    # grab ghost states to determine scared value and nearest ghost
    ghost_states = current_game_state.get_ghost_states()
    scared_value = sum([ghost.scared_timer for ghost in ghost_states])

    # grab nearest ghost dist
    ghost_dist = reduce(
        lambda score, ghost: min(
            DistanceManager.check_distance(ghost.get_position(), pac_pos),
            score
        ),
        current_game_state.get_ghost_states(),
        float("inf")
    )

    # determine closest food distance
    food_dist = reduce(
        lambda score, food_pos: min(
                DistanceManager.check_distance(food_pos, pac_pos),
                score
        ),
        current_game_state.get_food().as_list(),
        float("inf")
    )

    # this will prioritize food when ghosts aren't around
    ghost_food_priority = ghost_dist / food_dist

    # Evaluate and return score
    return current_game_state.get_score() + \
        scared_value + \
        ghost_food_priority


class DistanceManager:
    """Distance manager - determines distance between two points."""

    distances = defaultdict()

    @staticmethod
    def check_distance(pos_a, pos_b):
        """Get the manhattan distance between any two points."""
        if (pos_a, pos_b) in DistanceManager.distances:
            return DistanceManager.distances[(pos_a, pos_b)]

        elif (pos_b, pos_a) in DistanceManager.distances:
            return DistanceManager.distances[(pos_b, pos_a)]

        else:
            distance = manhattan_distance(pos_a, pos_b)
            DistanceManager.distances[(pos_a, pos_b)] = distance

            return distance


# Abbreviation
better = better_evaluation_function
