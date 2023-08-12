from socket import has_dualstack_ipv6
import gym
import numpy as np
from gym import spaces
import random
import math

class WaterWorldActions:
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

class WaterWorldAgent:
    def __init__(self, current_loc, vx, vy):
        self._current_loc = current_loc
        self._vx = vx
        self._vy = vy
        self._max_vel = 2
        self._radius = 1
        self._cached_hash = None
        self.__hash__()

    def __str__(self):
        string = "("
        string += "("+str(self._current_loc[0])+","+str(self._current_loc[1])+"),"
        string += str(self._vx)+","
        string += str(self._vy)+")"
        return string

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, WaterWorldAgent):
            return False
        else:
            return self._cached_hash == other._cached_hash

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        self._cached_hash = hash((self._current_loc, self._vx, self._vy))
        return self._cached_hash


class WaterWorldBall:
    def __init__(self, type, range):
        self._type = type
        self._loc = (random.randint(0,range[0]), random.randint(0,range[1]))
        self._vx = 2
        self._vy = 2
        self._radius = 1
        if random.random() < 0.5: self._vx *= -1
        if random.random() < 0.5: self._vy *= -1
        self._cached_hash = None
        self.__hash__()

    def __str__(self):
        string = "("
        string += "("+str(self._loc[0])+","+str(self._loc[1])+"),"
        string += str(self._vx)+","
        string += str(self._vy)+")"
        return string

    def __eq__(self, other):
        if self is other:
            return True
        elif not isinstance(other, WaterWorldBall):
            return False
        else:
            return self._cached_hash == other._cached_hash

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        self._cached_hash = hash((self._loc, self._vx, self._vy))
        return self._cached_hash


class WaterworldEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, types, step_max):
        super(WaterworldEnv, self).__init__()

        self._grid_size = (100, 100)
        self._types = types
        self.agent = None
        self.balls = self.create_balls(types)
        self._visited = []
        self._n_state_variables = (len(self.balls))*2 + 4
        self._locations = []

        self.step_max = step_max
        self.steps = 0
        self.done = False
        self.success = False
        self.num_episodes = 0
        self.total_reward = 0

        num_actions = 4
        num_states = ((self._grid_size[0] * self._grid_size[1]) * 4)
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(num_actions)
        low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                0.0,
                0.0,
                # velocity bounds is 5x rated speed
                -2.0,
                -2.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ).astype(np.float32)
        high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                100.0,
                100.0,
                # velocity bounds is 5x rated speed
                2.0,
                2.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ]
        ).astype(np.float32)
        self.observation_space = spaces.Box(low, high)
        self.id_to_action = {0: "RIGHT", 1: "LEFT", 2: "UP", 3:"DOWN"}
        self.action_probs = {0:[2,3], 1:[2,3], 2:[0,1], 3:[0,1]}

    def create_balls(self, types):
        temp = types.keys()
        balls = []
        for t in temp:
            n = types[t][0]
            r = types[t][1]
            for i in range (n):
                balls.append(WaterWorldBall(t, self._grid_size))
        return balls

    def separate_balls(self):
        good = []
        bad = []
        for bubble in self.balls:
            if bubble._type == "good":
                good.append(bubble)
            elif bubble._type == "bad":
                bad.append(bubble)
        return good, bad

    def in_bound_x(self, x):
        if x < 0 or x > self._grid_size[0]: return False
        else: return True
    
    def in_bound_y(self, y):
        if y < 0 or y > self._grid_size[1]: return False
        else: return True

    def move_ball(self, ball):
        x0, y0 = ball._loc[0], ball._loc[1]
        vx, vy = ball._vx, ball._vy
        x = vx + x0
        y = vy + y0
        if not self.in_bound_x(x):
            ball._vx *= -1
            if x > 0:
                b = self._grid_size[0]
                off = x - b
                x = b - off
            else:
                x = abs(x)

        if not self.in_bound_y(y):
            ball._vy *= -1
            if y > 0:
                b = self._grid_size[1]
                off = y - b
                y = b - off
            else:
                y = abs(y)
        ball._loc = (x, y)

    def move_all_balls(self):
        for bubble in self.balls:
            self.move_ball(bubble)

    def move_agent(self, action):
        new_vx, new_vy = self.agent._vx, self.agent._vy
        if action == WaterWorldActions.UP: 
            if self.agent._vy < 0: 
                new_vy = 0
            else: 
                new_vy = self.agent._max_vel
        elif action == WaterWorldActions.DOWN: 
            if self.agent._vy > 0: 
                new_vy = 0
            else: 
                new_vy = -self.agent._max_vel
        elif action == WaterWorldActions.RIGHT: 
            if self.agent._vx < 0: 
                new_vx = 0
            else: 
                new_vx = self.agent._max_vel
        elif action == WaterWorldActions.LEFT: 
            if self.agent._vx > 0: 
                new_vx = 0
            else: 
                new_vx = -self.agent._max_vel

        x0 = self.agent._current_loc[0]
        y0 = self.agent._current_loc[1]
        x = new_vx + x0
        y = new_vy + y0
        if not self.in_bound_x(x):
            if x < 0: 
                x = 0
            else: 
                x = self._grid_size[0]

        if not self.in_bound_y(y):
            if y < 0: 
                y = 0
            else: 
                y = self._grid_size[1]
        self.agent = WaterWorldAgent((x, y),  new_vx, new_vy)

    def does_collide(self, ball):
        dist = math.sqrt(math.pow(ball._loc[0] - self.agent._current_loc[0],2) 
                         + math.pow(ball._loc[1] - self.agent._current_loc[1],2))
        threshold = ball._radius + self.agent._radius
        if dist < threshold: return True
        else: return False

    def no_moving_green_bubble(self, green_bubbles):
        for bubble in green_bubbles:
            if bubble._vx != 0 and bubble._vy != 0:
                return False
        return True

    def bad_bubble_collides(self, bad_bubbles):
        for bubble in bad_bubbles:
            if self.does_collide(bubble):
                return True
        return False

    def good_bubble_collides(self, good_bubbles):
        for bubble in good_bubbles:
            if self.does_collide(bubble):
                if bubble._vx !=0 and bubble._vy != 0:
                    bubble._vx, bubble._vy = 0, 0 
                    return True
        return False

    def step(self, action):
        reward = -1 # the episode's reward (-100 for pitfall, 0 for reaching the goal, and -1 otherwise)
        self.done = False # termination flag is true if the agent falls in a pitfall or reaches to the goal
        self.success = False

        self.move_agent(action)
        self.move_all_balls()
        good_bubbles, bad_bubbles = self.separate_balls()

        # print("action:",self.id_to_action[action])
        if self.no_moving_green_bubble(good_bubbles):
            reward = 1000
            self.done = True
            self.success = True
            # print("Goal reached")
        elif self.bad_bubble_collides(bad_bubbles):
            reward = -1000
            self.done = True
            self.success = False
        elif self.good_bubble_collides(good_bubbles):
            reward = 10
            self.done = False
            self.success = False

        self.state = self.encode(self.agent, self.balls)
        
        self.steps += 1
        if self.steps == self.step_max:
            self.done = True
        if self.done:
            # self.render()
            self.num_episodes += 1
        self.total_reward += reward
        # self.render()
        # print(new_agent_loc, self.state_id)

        info = {}
        info["done"] = self.done
        info["succ"] = self.success
        info["reward"] = self.total_reward
        info["steps"] = self.steps
        info["num_episodes"] = self.num_episodes
        return self.state, reward, self.done, info

    def encode(self, agent, balls):
        state = []
        state.append(agent._current_loc[0])
        state.append(agent._current_loc[1])
        state.append(agent._vx)
        state.append(agent._vy)
        for bubble in balls:
            state.append(bubble._loc[0])
            state.append(bubble._loc[1])
            #states.append(bubble._vx)
            #states.append(bubble._vy)
        return np.array(state, dtype=np.float32)

    def reset(self):
        self.steps = 0
        self.total_reward = 0
        self.done = False
        self.success = False

        current_loc = (self._grid_size[0]//2, self._grid_size[1]//2)
        vx = 0
        vy = 0
        self.agent = WaterWorldAgent(current_loc, vx, vy)
        self.balls = self.create_balls(self._types)
        self.state = self.encode(self.agent, self.balls)
        # print(self.state)
        return self.state

    def render(self, mode='human'):
        state = self.encode(self.agent, self.balls)
        print(state)

    def close (self):
        pass
