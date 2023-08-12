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

class WaterWorldParams:
    def __init__(self, gridsize):
        self.render = False
        if gridsize == (200, 200):
            self._grid_size = gridsize
            self.radius = 20
            self.agent_vel_delta = 30
            self.agent_vel_max = 90
            self.ball_vel = 30
            self.color_to_num_balls = {"green":1, "red":1}
        elif gridsize == (250, 250):
            self._grid_size = gridsize
            self.radius = 20
            self.agent_vel_delta = 30
            self.agent_vel_max = 90
            self.ball_vel = 30
            self.color_to_num_balls = {"green":1, "red":1}
        elif gridsize == (300, 300):
            self._grid_size = gridsize
            self.radius = 20
            self.agent_vel_delta = 30
            self.agent_vel_max = 90
            self.ball_vel = 30
            self.color_to_num_balls = {"green":1, "red":1}
        elif gridsize == (350, 350):
            self._grid_size = gridsize
            self.radius = 20
            self.agent_vel_delta = 30
            self.agent_vel_max = 90
            self.ball_vel = 30
            self.color_to_num_balls = {"green":1, "red":1}
        elif gridsize == (400, 400):
            self._grid_size = gridsize
            self.radius = 20
            self.agent_vel_delta = 30
            self.agent_vel_max = 90
            self.ball_vel = 30
            self.color_to_num_balls = {"green":1, "red":1}

class WaterWorldBall:
    def __init__(self, color, radius, loc, vel):
        self.color = color
        self.radius = radius
        self.loc = np.array(loc, dtype=np.float)
        self.vel = np.array(vel, dtype=np.float)

    def update_loc(self, max_x, max_y, elapsedTime):
        self.loc = self.loc + elapsedTime * self.vel
        # handle collisions with walls
        if self.loc[0] - self.radius < 0 or self.loc[0] + self.radius > max_x:
            # Place ball against edge
            if self.loc[0] - self.radius < 0: 
                self.loc[0] = self.radius          
            else: 
                self.loc[0] = max_x - self.radius
            # Reverse direction
            self.vel = self.vel * np.array([-1.0,1.0])
        if self.loc[1] - self.radius < 0 or self.loc[1] + self.radius > max_y:
            # Place ball against edge
            if self.loc[1] - self.radius < 0: 
                self.loc[1] = self.radius
            else: 
                self.loc[1] = max_y - self.radius
            # Reverse direction
            self.vel = self.vel * np.array([1.0,-1.0])

    def is_colliding(self, ball):
        d = np.linalg.norm(self.loc - ball.loc, ord=2)
        return d <= self.radius + ball.radius

    def get_info(self):
        return self.loc, self.vel

    def __str__(self):
        string = "("
        string += self.color+","
        string += "("+str(self.loc[0])+","+str(self.loc[1])+"),"
        string += "("+str(self.vel[0])+","+str(self.vel[1])+")"
        string += ")"
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

class WaterWorldAgent(WaterWorldBall):
    def __init__(self, color, radius, loc, vel, vel_delta, vel_max):
        super().__init__(color, radius, loc, vel)
        self._vel_delta = float(vel_delta)
        self._vel_max = float(vel_max)
    
    def update_loc_vel(self, action, max_x, max_y, elapsedTime):
        # updating velocity
        delta = np.array([0,0])
        if action == WaterWorldActions.UP: 
            delta = np.array([0,1.0])
        elif action == WaterWorldActions.DOWN: 
           delta = np.array([0,-1.0])
        elif action == WaterWorldActions.RIGHT: 
            delta = np.array([1.0,0])
        elif action == WaterWorldActions.LEFT: 
            delta = np.array([-1.0,0])
        self.vel += self._vel_delta * delta
        # checking limits
        self.vel = np.clip(self.vel, -self._vel_max, self._vel_max)
        # updating location
        self.update_loc(max_x, max_y, elapsedTime)


class WaterworldEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, step_max, gridsize):
        super(WaterworldEnv, self).__init__()

        self.params = WaterWorldParams(gridsize)
        if self.params.render:
            import pygame
            pygame.init()
            self.gameDisplay = pygame.display.set_mode((self.params._grid_size[0], self.params._grid_size[1]))
            pygame.display.set_caption('Water world :)')
        self.reset()

        self.step_max = step_max
        self.steps = 0
        self.done = False
        self.success = False
        self.num_episodes = 0
        self.total_reward = 0

        # Define action and observation space
        # They must be gym.spaces objects
        self._n_state_variables = 4 + (len(self.balls))*5
        self.action_space = spaces.Discrete(4)
        # low = np.array(
        #     [
        #         # these are bounds for position
        #         # realistically the environment should have ended
        #         # long before we reach more than 50% outside
        #         0.0,
        #         0.0,
        #         # velocity bounds is 5x rated speed
        #         -math.ceil(self.params.agent_vel_max)-1,
        #         -math.ceil(self.params.agent_vel_max)-1,
        #         0.0,
        #         0.0,
        #         0.0,
        #         -math.ceil(self.params.agent_vel_max)-1,
        #         -math.ceil(self.params.agent_vel_max)-1,
        #         0.0,
        #         0.0,
        #         0.0,
        #         -math.ceil(self.params.agent_vel_max)-1,
        #         -math.ceil(self.params.agent_vel_max)-1
        #     ]
        # ).astype(np.float32)
        # high = np.array(
        #     [
        #         # these are bounds for position
        #         # realistically the environment should have ended
        #         # long before we reach more than 50% outside
        #         self.params._grid_size[0],
        #         self.params._grid_size[1],
        #         math.ceil(self.params.agent_vel_max)+1,
        #         math.ceil(self.params.agent_vel_max)+1,
        #         2.0,
        #         self.params._grid_size[0],
        #         self.params._grid_size[1],
        #         math.ceil(self.params.agent_vel_max)+1,
        #         math.ceil(self.params.agent_vel_max)+1,
        #         2.0,
        #         self.params._grid_size[0],
        #         self.params._grid_size[1],
        #         math.ceil(self.params.agent_vel_max)+1,
        #         math.ceil(self.params.agent_vel_max)+1
        #     ]
        # ).astype(np.float32) 
        # self.observation_space = spaces.Box(low=low, high=high)
        self.observation_space = spaces.Box(low=-2, high=2, shape=(self._n_state_variables,), dtype=np.float32)
        self.id_to_action = {0: "RIGHT", 1: "LEFT", 2: "UP", 3:"DOWN"}

    def separate_balls(self):
        good = []
        bad = []
        for ball in self.balls:
            if ball.color == "green":
                good.append(ball)
            elif ball.color == "red":
                bad.append(ball)
        return good, bad

    def agent_collides(self, balls):
        for ball in balls:
            if self.agent.is_colliding(ball):
                return True
        return False

    def step(self, action, elapsedTime=0.1):
        reward = -1
        self.done = False
        self.success = False

        self.agent.update_loc_vel(action, self.params._grid_size[0], self.params._grid_size[1], elapsedTime)
        for ball in self.balls:
            ball.update_loc(self.params._grid_size[0], self.params._grid_size[1], elapsedTime)
        
        good_balls, bad_balls = self.separate_balls()
        # print("action:",self.id_to_action[action])
        if self.agent_collides(good_balls):
            reward = 1000
            self.done = True
            self.success = True
            # print("Goal reached")
        elif self.agent_collides(bad_balls):
            reward = -1000
            self.done = True
            self.success = False

        self.state = self.encode(self.agent, self.balls)
        
        self.steps += 1
        if self.steps == self.step_max:
            self.done = True
        if self.done:
            # self.render()
            self.num_episodes += 1
        self.total_reward += reward
        if self.params.render:
            self.render()

        info = {}
        info["done"] = self.done
        info["succ"] = self.success
        info["reward"] = self.total_reward
        info["steps"] = self.steps
        info["num_episodes"] = self.num_episodes
        return self.state, reward, self.done, info

    def _is_collising(self, radius, loc):
        for b in self.balls + [self.agent]:
            if np.linalg.norm(b.loc - np.array(loc), ord=2) < 2 * radius:
                return True
        return False

    def _get_random_loc(self, radius):
        return [2*radius + random.random()*(self.params._grid_size[0] - 2*radius), 2*radius + random.random()*(self.params._grid_size[1] - 2*radius)]

    def _get_loc_vel_new_ball(self, radius, ball_vel):
        angle = random.random()*2*math.pi
        vel = ball_vel*math.sin(angle),ball_vel*math.cos(angle)
        while True:
            loc = self._get_random_loc(radius)
            if not self._is_collising(radius, loc) and np.linalg.norm(self.agent.loc - np.array(loc), ord=2) > 4*radius:
                break
        return loc, vel  

    def reset(self):
        self.steps = 0
        self.total_reward = 0
        self.done = False
        self.success = False
     
        radius = self.params.radius
        loc = self._get_random_loc(radius)
        vel = [0.0, 0.0]
        vel_delta = self.params.agent_vel_delta
        vel_max = self.params.agent_vel_max
        self.agent = WaterWorldAgent("black", radius, loc, vel, vel_delta, vel_max)

        color_to_num_balls = self.params.color_to_num_balls
        ball_vel = self.params.ball_vel
        self.balls = list()
        for color in color_to_num_balls.keys():
            for i in range(color_to_num_balls[color]):
                loc, vel = self._get_loc_vel_new_ball(radius, ball_vel)
                ball = WaterWorldBall(color, radius, loc, vel)
                self.balls.append(ball)

        self.state = self.encode(self.agent, self.balls)
        # print(self.state)
        if self.params.render:
            self.render()
        return self.state

    def encode(self, agent, balls):
        n_features = 4 + len(balls) * 5
        features = np.zeros(n_features,dtype=np.float32)

        pos_max = np.array([float(self.params._grid_size[0]), float(self.params._grid_size[1])])
        vel_max = float(self.params.ball_vel + agent._vel_max)

        features[0:2] = agent.loc/pos_max
        features[2:4] = agent.vel/float(agent._vel_max)
        init = 4
        for i in range(len(balls)):
            # If the balls are colliding, I'll not include them 
            # (because there is nothing that the agent can do about it)
            b = balls[i]
            if b.color == 'green': color = 1
            elif b.color == 'red': color = 0
            if not agent.is_colliding(b):
                # init = 4*(i+1)
                features[init:init+2]   = (b.loc - agent.loc)/pos_max
                features[init+2:init+4] = (b.vel - agent.vel)/vel_max
                features[init+4] = color
                init = init+5

        # without normalization
        # features[0] = agent.loc[0]
        # features[1] = agent.loc[1]
        # features[2] = agent.vel[0]
        # features[3] = agent.vel[1]
        # features[4] = 1 if agent.color=='green' else 0
        # init = 4
        # for i in range(len(balls)):
        #     b = balls[i]
        #     if b.color == 'green': color = 1
        #     elif b.color == 'red': color = 0
        #     if not agent.is_colliding(b):
        #         features[init]   = b.loc[0]
        #         features[init+1]   = b.loc[1]
        #         features[init+2]   = b.vel[0] 
        #         features[init+3]   = b.vel[1]
        #         features[init+4] = color
        #         init = init+5
        return features

    def get_position(self, ball, max_y):
        return int(round(ball.loc[0])), int(max_y) - int(round(ball.loc[1]))

    def draw_ball(self, ball, colors, thickness, gameDisplay, pygame, max_y):
        pygame.draw.circle(gameDisplay, colors[ball.color], self.get_position(ball, max_y), ball.radius, thickness)

    def render(self, mode='human'):
        import pygame
        state = self.encode(self.agent, self.balls)
        max_x = self.params._grid_size[0]
        max_y = self.params._grid_size[1]

        colors = {"red": "red", "green": "green", "white": (255,255,255), "black": (0,0,0)}

        clock = pygame.time.Clock()
        crashed = False

        # printing image
        self.gameDisplay.fill(colors["white"])
        for b in self.balls:
            self.draw_ball(b, colors, 0, self.gameDisplay, pygame, max_y)
        self.draw_ball(self.agent, colors, 0, self.gameDisplay, pygame, max_y)
        pygame.display.update()
        clock.tick(20)
        # print(state)

    def close (self):
        pass
