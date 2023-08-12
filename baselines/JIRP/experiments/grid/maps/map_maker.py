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
    for x in range(old_map.shape[0]):
        for y in range(old_map.shape[1]):
            if np.array_equal(old_map[x, y], np.array((0,0,0))):
                map[x,y] = 1
            #adds any other color than black or white as pitfall, can add additional colors here for different enviroments
            elif np.array_equal(old_map[x,y], np.array((255,0,0))):
                map[x,y] = -1
            elif not np.array_equal(old_map[x, y], np.array((255,255,255))):
                map_bytes = old_map[x,y].tobytes()
                if map_bytes in color_array.keys():
                    map[x,y] = color_array[map_bytes]
                else:
                    map[x,y] = count
                    color_array[map_bytes] = count
                    count +=1
    return map

def make_map_txt(map, txt_map):
    # 0 for free location, 1 for obstacle, 2 for pits
    height = 64
    width = 64
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
    file_map = "./map_3.png"
    txt_map = "./map_3.map"
    map = get_map(file_map)
    make_map_txt(map, txt_map)