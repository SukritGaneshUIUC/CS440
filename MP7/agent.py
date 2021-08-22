import numpy as np
import utils
import random


class Agent:

    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

        self.s = None
        self.a = None

        # self.ACTION_DICT = {'U':0, 'D':1, 'L':2', 'R':3}

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    # return adjoining_wall_x, adjoining_wall_y
    def check_adjoining_wall(self, state):
        snake_head_x, snake_head_y, snake_body, food_x, food_y = state

        # OUT OF BOUNDS
        if (snake_head_x <= 0 or snake_head_x >= 520 or snake_head_y <= 0 or snake_head_y >= 520):
            return 0, 0

        # adjoining_wall_x
        if (snake_head_x == 40):
            adjoining_wall_x = 1
        elif (snake_head_x == 480):
            adjoining_wall_x = 2
        else:
            adjoining_wall_x = 0

        # adjoining_wall_y
        if (snake_head_y == 40):
            adjoining_wall_y = 1
        elif (snake_head_y == 480):
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 0

        return adjoining_wall_x, adjoining_wall_y

    # return food_dir_x, food_dir_y
    def check_food_dir(self, state):
        snake_head_x, snake_head_y, snake_body, food_x, food_y = state

        # food_dir_x
        if (food_x < snake_head_x):
            food_dir_x = 1
        elif (food_x > snake_head_x):
            food_dir_x = 2
        else:
            food_dir_x = 0

        # food_dir_y
        if (food_y < snake_head_y):
            food_dir_y = 1
        elif (food_y > snake_head_y):
            food_dir_y = 2
        else:
            food_dir_y = 0

        return food_dir_x, food_dir_y

    # return adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right
    def check_adjoining_body(self, state):
        snake_head_x, snake_head_y, snake_body, food_x, food_y = state

        if ((snake_head_x, snake_head_y - 40) in snake_body):
            adjoining_body_top = 1
        else:
            adjoining_body_top = 0

        if ((snake_head_x, snake_head_y + 40) in snake_body):
            adjoining_body_bottom = 1
        else:
            adjoining_body_bottom = 0

        if ((snake_head_x - 40, snake_head_y) in snake_body):
            adjoining_body_left = 1
        else:
            adjoining_body_left = 0

        if ((snake_head_x + 40, snake_head_y) in snake_body):
            adjoining_body_right = 1
        else:
            adjoining_body_right = 0

        return adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right

    def convert_coordinates_to_state(self, state):
        # print(state)

        # Adjoining Wall States
        adjoining_wall_x, adjoining_wall_y = self.check_adjoining_wall(state)

        # Food Dir States
        food_dir_x, food_dir_y = self.check_food_dir(state)

        # Adjoining body states
        adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = self.check_adjoining_body(state)

        # Return
        return [adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right]

    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''

        FIRST_MOVE = (self.s == None and self.a == None)
        # print()
        # print('State: ', state)
        # print('Points: ', points)
        # print(state, '\n', points, '\n')

        # make sure state is in game format, NOT just the coordinates of the snake head and body and food
        new_state_coordinates = state
        new_state = self.convert_coordinates_to_state(new_state_coordinates)

        # Update Q Table, make use of most recent state and action (Train Mode ONLY)
        if (self._train and not FIRST_MOVE):
            self.update_q_table(new_state, points, dead)

        # Update current state (s = s')
        # make sure to store self.s
        self.s = new_state_coordinates
        current_state = new_state
        self.points = points

        # Action to be taken (Train and Eval modes)
        self.a = self.get_next_action(current_state, points, dead)

        # BREAK if dead
        if (dead):
            # print('reset')
            self.reset()
            return self.a

        # Update N Table (Train Mode ONLY)
        if (self._train):
            self.update_n_table(current_state, self.a)

        # Return action taken
        # if (not self._train):
        #     print('Action taken: ', self.a)
        return self.a

    # f(u,n) returns 1 if n is less than a tuning parameter Ne, otherwise it returns u.
    def f(self, u, n):
        if (n < self.Ne):
            return 1
        else:
            return u

    def get_reward(self, points, dead):
        snake_head_x, snake_head_y, snake_body, food_x, food_y = self.s

        if (points > self.points):
            return 1
        elif (dead):
            return -1
        else:
            return -0.1

        # if (snake_head_x == food_x and snake_head_y == food_y):
        #     return 1
        # elif ((snake_head_x, snake_head_y) in snake_body):
        #     return -1
        # elif (snake_head_x <= 0 or snake_head_x >= 520 or snake_head_y <= 0 or snake_head_y >= 520):
        #     return -1
        # else:
        #     return -0.1


    def update_q_table(self, new_state, points, dead):
        # Unpack state (this is actually s')
        # adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = s_prime

        # Pick the action a' which maximizes the value Q(s', a')
        # Tie breaker: right (3) > left (2) > down (1) > up (0)
        Qvals = self.Q[tuple(new_state)]
        # print(Qvals)
        maxQ = Qvals[3]
        maxAction = 3
        for i in (3, 2, 1, 0):
            currQ = Qvals[i]
            if (currQ > maxQ):
                maxQ = currQ
                maxAction = i
        # print(maxAction)

        # Update the Q table (based on self.s, self.a, and the maxQ we just calculated)
        # Note that we use Q(s, a) as opposed to Q(s', a')
        R = self.get_reward(points, dead)
        # adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = self.convert_coordinates_to_state(self.s)
        # Qsa = self.Q[adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, self.a]
        # alpha = self.C / (self.C + self.N[adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, self.a])
        old_state = self.convert_coordinates_to_state(self.s)
        Qsa = self.Q[tuple(old_state)][self.a]
        alpha = self.C / (self.C + self.N[tuple(old_state)][self.a])

        updateVal = alpha * (R + self.gamma * maxQ - Qsa)

        # self.Q[adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, self.a] += updateVal
        self.Q[tuple(old_state)][self.a] += updateVal

    def get_next_action(self, s, points, dead):
        # Unpack state
        # adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = s

        # Get values of Q and N tables
        # Qvals = self.Q[adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, :]
        # Nvals = self.N[adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, :]
        Qvals = self.Q[tuple(s)]
        Nvals = self.N[tuple(s)]

        # Pick the action which maximizes the value f(Q(s, a), N(s, a))
        # Tie breaker: right (3) > left (2) > down (1) > up (0)
        # if (not self._train):
        #     print(Qvals)
        #     print(Nvals)
        if (self._train):
            maxF = self.f(Qvals[3], Nvals[3])
        else:
            maxF = Qvals[3]
        maxAction = 3
        for i in (3, 2, 1, 0):
            if (self._train):
                currF = self.f(Qvals[i], Nvals[i])
            else:
                currF = Qvals[i]
            if (currF > maxF):
                maxF = currF
                maxAction = i

        return maxAction

    def update_n_table(self, s, a):
        # Unpack state
        # adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = s

        # print('Current n-value: ', self.N[adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, a])

        # Update by incrementing respective cell in N table
        # self.N[adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, a] += 1
        self.N[tuple(s)][a] += 1
