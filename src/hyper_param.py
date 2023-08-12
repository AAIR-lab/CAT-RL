from src.envs.taxiworld import *
from src.envs.officeworld import *
from src.envs.waterworld import *
from src.envs.wumpusworld import *
from src.envs.mountaincar import *

# Mountain Car
#epsilon_min = 0.01
#alpha = 0.05
#decay = 0.9
#gamma = 0.99
#lam = 0.5
#k_cap = 1
#step_max = 200
#episode_max = 2500
#map_name = "Mountain_Car"
#env = Mountain_Car()
#bootstrap = 'from_ancestor'

# grid domain 64 64
#epsilon_min = 0.05
#alpha = 0.05
#decay = 0.9991
#gamma = 0.95
#lam = 0.5
#k_cap = 20
#step_max = 1200
#episode_max = 10000
#map_name = "grid_64x64_map1"
#env = Simple_Grid (map_name, [0,0], [63,63])
#bootstrap = 'from_concrete'


# grid domain 44 44
#epsilon_min = 0.05
#alpha = 0.05
#decay = 0.9991
#gamma = 0.95
#lam = 0.5
#k_cap = 20
#step_max = 800
#episode_max = 5000
#map_name = "grid_44x44_map1"
#env = Simple_Grid (map_name, [0,0], [43,43])
#bootstrap = 'from_concrete'


# grid domain 32 32
epsilon_min = 0.05
alpha = 0.05
decay = 0.9991
gamma = 0.95
lam = 0.5
k_cap = 20
step_max = 600
episode_max = 5000
map_name = "grid_32x32_map1"
env = Simple_Grid (map_name, [0,0], [31,31])
bootstrap = 'from_concrete'


# grid domain 16 16
#epsilon_min = 0.05
#alpha = 0.05
#decay = 0.9991
#gamma = 0.95
#lam = 0.5
#k_cap = 2
#step_max = 300
#episode_max = 800
#map_name = "grid_32x32_map4"
#env = Simple_Grid (map_name, [0,31], [0,0])
#bootstrap = 'from_concrete'

# grid domain 8 8
# epsilon_min = 0.05
# alpha = 0.05
# decay = 0.998
# gamma = 0.95
# lam = 0.5
# k_cap = 2
# step_max = 100
# episode_max = 5000
# map_name = "grid_8x8_map1"
# env = Simple_Grid (map_name, [0,0], [7,7])
# bootstrap = 'from_concrete'



# taxi domain 30 30
#epsilon_min = 0.05
#alpha = 0.05
#decay = 0.999
#gamma = 1
#lam = 0.5
#k_cap = 10
#step_max = 1500
#episode_max = 20000
#map_name = "taxi_30x30_map1"
#env = Taxi_Domain (map_name, [0,0], 1)
#bootstrap = 'from_concrete' #from_concrete for discrete domains


# office domain 54 x 54
#epsilon_min = 0.05
#alpha = 0.05
#decay = 0.9993
#gamma = 0.99
#lam = 0.5
#k_cap = 20
#step_max = 1500
#episode_max = 10000
#gridsize = (54,54) # (27,27) (36,36) (45,45) (54,54)
#map_name = "office_"+str(gridsize[0])+"x"+str(gridsize[1])+"_map1"
#env = Office_Domain (map_name)
#bootstrap = 'from_concrete'

# office domain 45 x 45
#epsilon_min = 0.05
#alpha = 0.05
#decay = 0.9992
#gamma = 0.99
#lam = 0.5
#k_cap = 20
#step_max = 1000
#episode_max = 3000
#gridsize = (45,45) # (27,27) (36,36) (45,45) (54,54)
#map_name = "office_"+str(gridsize[0])+"x"+str(gridsize[1])+"_map1"
#env = Office_Domain (map_name)
# bootstrap = 'from_concrete'

# office domain 36 x 36
#epsilon_min = 0.05
#alpha = 0.05
#decay = 0.995
#gamma = 0.99
#lam = 0.5
#k_cap = 20
#step_max = 700
#episode_max = 10000
#gridsize = (36,36) # (27,27) (36,36) (45,45) (54,54)
#map_name = "office_"+str(gridsize[0])+"x"+str(gridsize[1])+"_map1"
#env = Office_Domain (map_name)
#bootstrap = 'from_concrete'

# office domain 27 x 27
#epsilon_min = 0.05
#alpha = 0.05
#decay = 0.999
#gamma = 0.99
#lam = 0.5
#k_cap = 20
#step_max = 500
#episode_max = 10000
#gridsize = (27,27) # (27,27) (36,36) (45,45) (54,54)
#map_name = "office_"+str(gridsize[0])+"x"+str(gridsize[1])+"_map1"
#env = Office_Domain (map_name)
#bootstrap = 'from_concrete'


# water domain 300 300
#epsilon_min = 0.05
#alpha = 0.05
#decay = 0.999
#gamma = 0.95
#lam = 0.5
#k_cap = 1
#step_max = 100
#episode_max = 5000
#gridsize = (200,200) # (250,250) (300,300) (350,350) (400,400)
#map_name = "water_"+str(gridsize[0])+"x"+str(gridsize[1])
#env = WaterworldEnv(gridsize)
#bootstrap = 'from_init'
