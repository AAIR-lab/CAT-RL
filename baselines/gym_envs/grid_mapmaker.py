import numpy as np
import pickle as pk
from PIL import Image

def get_map(file_map):
    color_array = {}
    count = 2 
    image = Image.open(file_map)
    image = image.convert("RGB")
    old_map = np.asarray(image)
    map = np.zeros((old_map.shape[0], old_map.shape[1]))
    # print(old_map)
    for x in range(old_map.shape[0]):
        for y in range(old_map.shape[1]):
            if np.array_equal(old_map[x, y], np.array((0,0,0))): # obstacles
                map[x,y] = 1
            elif not np.array_equal(old_map[x, y], np.array((255,255,255))) and not np.array_equal(old_map[x, y], np.array((0,0,0))): # pits
                map[x,y] = 2
    # print(map)
    return map

def make_map_txt(map, txt_map, height, width):
    # 0 for free location, 1 for obstacle, 2 for pits
    string = ""
    for row in range(height):
        line = ""
        for col in range(width):
            symbol = int(map[row, col])
            if symbol == 0:
                line += "-"
            elif symbol == 1: # obstacle
                line += "X"
            elif symbol == 2: # pit
                line += "O"
        line += "\n"
        string += line
    f = open(txt_map,"w")
    f.write(string)
    f.close()

      
if __name__=="__main__":
    height = 8
    width = height
    file_map = "../maps/grid_"+str(height)+"x"+str(width)+"_map1.png"
    txt_map = "../maps/grid_"+str(height)+"x"+str(width)+"_map1.map"
    map = get_map(file_map)
    make_map_txt(map, txt_map, height, width)