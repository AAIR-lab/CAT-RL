import os
import pickle as pk
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

class Results:
  
    def __init__(self, map_name, exp_number, baseline_included = True):
        self._experiment_number = exp_number
        self._data_abs = []
        self._data_baseline = []

        path_abs = os.getcwd() + '/results/' + map_name + '_'
        path_base = os.getcwd() + '/results/' + "base_" + map_name + '_'

        for i in range (1,self._experiment_number + 1):

            with open(path_abs + str(i), "rb") as file_name:
                self._data_abs.append( pk.load(file_name) )
            if baseline_included:
                with open(path_base + str(i), "rb") as file_name:
                    self._data_baseline.append (pk.load(file_name))
        
        self._color_list = []
        self._color_bounds = [-0.5]
        self._color_assignment = {}
        self._color_map = None
        self._color_track = 0
        self._sub_a = 0
        self._sub_b = 0
        self._abs_plot_dim = None
        self._counter = 0
        self._color_maps = []
        self._phase_success = []
        self._maze = self._data_abs[0]
        self._baseline_included = baseline_included
        self.data_extraction()
        self._episode_number = len(self._rewards_abs[0])


    def adjust_rate (self, data):
        epi_counter = 0
        success_counter = 0
        data_adjusted = []
        for i in range (len(data)):
            epi_counter += 1
            if data[i] == 1: success_counter +=1
            rate = success_counter / epi_counter
            data_adjusted.append(rate)
        return data_adjusted


    def data_extraction (self):
        self._rewards_abs = []
        self._rewards_base = []
        self._success_abs = []
        self._success_base = []
        self._abstraction_result = []
        self._qtables = []
        self._map = self._data_abs[0][0]

        for i in range ( self._experiment_number):
            temp_abs = []
            self._abstraction_result.append(self._data_abs[i][1][-1])
            self._qtables.append (self._data_abs[i][3])
            for item in self._data_abs[i][2][1]:
                temp_abs = temp_abs + item
            self._rewards_abs.append(temp_abs)

            temp_abs = []
            for item in self._data_abs[i][2][2]:
                temp_abs = temp_abs + item
            self._success_abs.append(self.adjust_rate(temp_abs))
            if self._baseline_included:
                self._rewards_base.append (self._data_baseline[i][2][1][0])
                self._success_base.append (self.adjust_rate(self._data_baseline[i][2][2][0]))
            if self._baseline_included: self._episodes = self._data_baseline[i][2][0][0]
   
        return None



    def prepare_avg_bound (self, param):
        abs = np.zeros ((3,self._episode_number))
        base = np.zeros ((3,self._episode_number))
        
        for i in range (self._episode_number):
            epi = self._episodes
            temp_abs = []
            temp_base = []
            data_dict_abs = {"rewards": self._rewards_abs, "success rate": self._success_abs}
            data_dict_base = {"rewards": self._rewards_base, "success rate": self._success_base}
  
            for j in range (self._experiment_number):
                
                temp_abs.append( data_dict_abs[param][j][i] )
                temp_base.append( data_dict_base[param][j][i] )

            temp_abs = np.array(temp_abs)
            temp_base = np.array(temp_base)
            abs_avg = np.average(temp_abs)
            base_avg = np.average(temp_base)
            abs_std = np.std(temp_abs)
            base_std = np.std(temp_base)
            abs[0][i] = abs_avg + abs_std
            abs[1][i] = abs_avg
            abs[2][i] = abs_avg - abs_std

            base[0][i] = base_avg + base_std
            base[1][i] = base_avg
            base[2][i] = base_avg - base_std
        return abs, base, epi

    def compare_bound (self, param, moving_number):

        abs, base, epi = self.prepare_avg_bound (param)
        abs_smooth, base_smooth, epi_smooth = [], [], []
        if moving_number != None:
            for i in range (3):
                temp_base, epi_smooth = self.moving_average(moving_number, base[i,::], epi)
                base_smooth.append(temp_base)
                temp_abs, aaa = self.moving_average(moving_number, abs[i,::], epi)
                abs_smooth.append(temp_abs)

        line_w = 1
        plt.plot(epi_smooth, abs_smooth[1], color='#009dff', linestyle='solid',
            linewidth=line_w,label='AD Q-learning')
        plt.fill_between(epi_smooth, abs_smooth[0], abs_smooth[2],
            alpha=0.5, edgecolor='#CC4F1B', facecolor='#b3e2ff',linewidth=0)
        
        plt.plot(epi_smooth, base_smooth[1], color='#ff6600', linestyle='solid',
            linewidth=line_w, label='Q-Learning')
        plt.fill_between(epi_smooth, base_smooth[0], base_smooth[2],
            alpha=0.5, edgecolor='#CC4F1B', facecolor='#ffd0b0ff',linewidth=0)
        plt.xlabel("episodes")
        plt.ylabel(param)
        plt.legend()
        plt.show()

    def moving_average (self, moving_number,y, x):
        y_m = []
        x_m = []
        if moving_number != np.inf:
            for i in range (moving_number, len(y)):
                sum_temp = 0
                for j in range (i - moving_number, i):
                    sum_temp += y[j]
                sum_temp /= moving_number
                y_m.append(sum_temp)
                x_m.append(i)
        else:
            for i in range (len(y)):
                sum_temp = 0
                for j in range (0, i):
                    sum_temp += y[j]
                sum_temp /= i + 1
                y_m.append(sum_temp)
                x_m.append(i)
        return y_m, x_m

    # Add your code here #
    def make_symbols(image_arr, map_arr, scale, thickness):
        for x in range(map_arr.shape[0]):
            for y in range(map_arr.shape[1]):
                if map_arr[x,y] == -1: 
                    image_arr[x*scale : x*scale + scale, y*scale + int(scale/2) -int(thickness/2):y*scale + int(scale/2) +int(thickness/2)] = np.array((255,0,0,255))
                    image_arr[x*scale + int(scale/2) -int(thickness/2): x*scale + int(scale/2) +int(thickness/2), y*scale : y*scale + scale] = np.array((255,0,0,255))
                elif map_arr[x,y] == 1:
                    image_arr[x*scale : x*scale + scale, y* scale : y*scale + scale] = np.array((0,0,0,255))
        return image_arr

    def mark_destination(image_arr, map_arr, scale, thickness, goal, p_locs):
        for x in range(map_arr.shape[0]):
            for y in range(map_arr.shape[1]):
                if [x,y] == goal: 
                    image_arr[x*scale : x*scale + scale, y*scale + int(scale/2) -int(thickness):y*scale + int(scale/2) +int(thickness)] = np.array((240,110,24,255))
                    image_arr[x*scale + int(scale/2) -int(thickness): x*scale + int(scale/2) +int(thickness), y*scale : y*scale + scale] = np.array((240,110,24,255))
                if [x,y] in p_locs: 
                    image_arr[x*scale : x*scale + scale, y*scale + int(scale/2) -int(thickness/2):y*scale + int(scale/2) +int(thickness/2)] = np.array((234,63,247,255))
                    image_arr[x*scale + int(scale/2) -int(thickness/2): x*scale + int(scale/2) +int(thickness/2), y*scale : y*scale + scale] = np.array((234,63,247,255))
                elif map_arr[x,y] == 1:
                    image_arr[x*scale : x*scale + scale, y* scale : y*scale + scale] = np.array((0,0,0,255))
        return image_arr

    def get_abstraction_visualization(abstraction_array):
        image_array = np.full((abstraction_array.shape[0],abstraction_array.shape[1],3), 255)
        colored_abstactions = {}
        for x in range(abstraction_array.shape[0]):
            for y in range(abstraction_array.shape[1]):
                if abstraction_array[x, y] not in colored_abstactions.keys():
                    color = Results.get_random_color()
                    colored_abstactions[abstraction_array[x, y]] = color
                    
                    image_array[x,y] = color
                else:
                    color = colored_abstactions[abstraction_array[x, y]]
                    image_array[x,y] = color
        im = Image.fromarray(image_array.astype(np.uint8))
        return im

    def get_heatmap_color(q, max, min, avg):

        if min < 0:
            min = abs (min)
            max += min
            q += min
            avg += min

        if max != 0:
            max_factor = 255/max
        else:
            max_factor = 0

        min_factor = 255/min

        if q > avg:
            r = 255 - int(q*max_factor)
            g = 255 -int(q*max_factor)
            b = 255
        else:
            r = 255
            g = int(q*min_factor) + int(q*max_factor)
            b = int(q*min_factor) + int(q*max_factor)
            
        return np.array((r,g,b))
    
    def get_max_min (q_table):
        states = list(q_table.keys())
        max_value, min_value = -np.inf, np.inf
        avg = 0
        for s in states:
            max_temp = np.max(q_table[s])
            min_temp = np.min(q_table[s])
            if max_temp > max_value: max_value = max_temp
            if min_temp < min_value: min_value = min_temp
            for v in q_table[s][0]: avg += v
        return max_value, min_value, avg/(len(states)*len(q_table[s][0]))

    def get_abstraction_heatmap(abstraction_array, q_table, colored_abstractions, best_actions):
        image_array = np.full((abstraction_array.shape[0],abstraction_array.shape[1],3), 255)
        # colored_abstactions = {}
        qmax, qmin, avg = Results.get_max_min (q_table)

        for lst in q_table.values():
            loc_max = lst.max()
            loc_min = lst.min()
            if loc_max > qmax:
                qmax = loc_max
            if loc_min < qmin:
                qmin = loc_min

        for x in range(abstraction_array.shape[0]):
            for y in range(abstraction_array.shape[1]):
                if abstraction_array[x, y] not in colored_abstractions.keys():
                    if abstraction_array[x,y] not in q_table: 
                        max_q_val = 0
                        best_action = -1
                    else: 
                        max_q_val = q_table[abstraction_array[x,y]].max()
                        best_action = q_table[abstraction_array[x,y]].argmax()
                    color = Results.get_heatmap_color(max_q_val, qmax, qmin, avg)
                    colored_abstractions[abstraction_array[x,y]] = color
                    best_actions[abstraction_array[x,y]] = best_action
                    image_array[x,y] = color
                else:
                    image_array[x,y] = colored_abstractions[abstraction_array[x,y]]
        im = Image.fromarray(image_array.astype(np.uint8))
        return im, colored_abstractions, best_actions

    def add_heatmap_bar(image, q_table, file_name):
        font = ImageFont.truetype(os.getcwd()+'/src/resources/FreeMono.ttf', 30)
        num_neg = 0
        num_pos = 0
        num_0 = 0
        qmax = -100000
        qmin = 100000
        for lst in q_table.values():
            if lst.max() < 0:
                num_neg += 1
            elif lst.max() > 0:
                num_pos += 1
            else:
                num_0 += 1
            loc_max = lst.max()
            if loc_max > qmax:
                qmax = loc_max
            if loc_max < qmin:
                qmin = loc_max
        total = num_0 + num_neg + num_pos
        percent_neg = num_neg/total
        percent_pos = num_pos/total
        percent_0 = num_0/total
        im_arr = np.asarray(image)
        width = im_arr.shape[0]
        height = im_arr.shape[1]
        new_im = np.full((width+60, height+200, 3), 255)

        for x in range(width):
            for y in range(height, height+50):
                if y >= height and y < height+11:
                    new_im[x+30,y] = (0,0,0)
                elif x < width * percent_neg:#fill with negative gradient
                    new_im[x+30,y] = (0+x*(255/int(width*percent_neg)), 255, 0+x*(255/int(width*percent_neg)))
                elif x > width * (percent_0+percent_neg):
                    num_pix = width - (width-width*percent_pos)
                    new_im[x+30,y] = (255 - ((x-(width*(1-percent_pos)))* (255/num_pix)), 255 - ((x-(width*(1-percent_pos)))* (255/num_pix)), 255)
                    #new_im[x,y] = (255- (x-width*(percent_pos))*(255/(width*percent_pos)),255 - (x-(width*percent_pos))*(255/(percent_pos)), 255)
        new_im[30:width+30,height+50: height+53] = (0,0,0)
        new_im[20:30, 0:height+53] = (0,0,0)
        new_im[width+30:width+40, 0:height+53] = (0,0,0)
        if percent_0 + percent_pos == 1:
            new_im[30:33,height+11:height+50] = (255,0,0)#red bar for minimum
            new_im[width+30-3 : width+30, height+11 : height+50] = (255,0,0) #red bar for max
            im = Image.fromarray(new_im.astype(np.uint8))
            im = ImageOps.flip(im)
            draw = ImageDraw.Draw(im)
            draw.text((height+55 ,30-15), str(int(qmax)), (255,0,0), font = font)#max
            draw.text((height+55 , 30+width-15), str(int(qmin)), (255,0,0), font = font)#min
            #im.show()
            im.paste(image, (0,30), 0)
            im.show('Results.png')
            im.save(file_name)
        elif percent_0 + percent_neg == 1:
            new_im[30:33,height+11:height+50] = (255,0,0)#red bar for minimum
            new_im[width+30-3 : width+30, height+11 : height+50] = (255,0,0) #red bar for max
            im = Image.fromarray(new_im.astype(np.uint8))
            im = ImageOps.flip(im)
            draw = ImageDraw.Draw(im)
            draw.text((height+55 ,30-15), str(int(qmax)), (255,0,0), font = font)#max
            draw.text((height+55 , 30+width-15), str(int(qmin)), (255,0,0), font = font)#min
            im.paste(image, (0,30), 0)

            im.show('Results.png')
            im.save(file_name)
        else:
            new_im[30:33,height+11:height+50] = (255,0,0)#red bar for minimum
            new_im[width+30-3 : width+30, height+11 : height+50] = (255,0,0) #red bar for max
            min_dist = 10000
            for x in range(30,width+30):
                if  min_dist > int(255-new_im[x, height+25, 0]+ 255-new_im[x, height+25, 1] + 255-new_im[x, height+25, 2]) and min_dist >= 0:
                    mid_x = x
                    min_dist = int(255-new_im[x, height+25, 0]+ 255-new_im[x, height+25, 1] + 255-new_im[x, height+25, 2])
            new_im[mid_x-1:mid_x+1, height+11:height+50] = (255,0,0)
            im = Image.fromarray(new_im.astype(np.uint8))
            im = ImageOps.flip(im)
            draw = ImageDraw.Draw(im)
            draw.text((height+55 ,30-15), str(int(qmax)), (255,0,0), font = font)#max
            draw.text((height+55 , width-mid_x+15+30), '0', (255,0,0), font = font)#middle
            draw.text((height+55 , 30+width-15), str(int(qmin)), (255,0,0), font = font)#min

            im.paste(image, (0,30), 0)
            im.show('Results.png')
            im.save(file_name)
            
    def get_qtable_heatmap(map_arr, abstraction_array, scale, q_table, file_name, goal, p_locs, abstraction_colors, best_actions):
        abstraction_image, abstraction_colors, best_actions = Results.get_abstraction_heatmap(abstraction_array, q_table, abstraction_colors, best_actions)
        image = Image.new("RGBA", (abstraction_array.shape[0]*scale, abstraction_array.shape[1]*scale), (255,255,255,0))
        image_arr = np.asarray(image)
        symbols = Image.fromarray(Results.mark_destination(image_arr, map_arr, scale, 7, goal, p_locs).astype(np.uint8))
        overlay = Results.get_faded_overlay(Results.clear_resize(abstraction_image, scale), symbols, 255)
        #overlay.show("Results.png")
        Results.add_heatmap_bar(overlay,q_table, file_name)
        return abstraction_colors, best_actions

    def get_full_image(map_arr, abstraction_array, scale):
        abstraction_image = Results.get_abstraction_visualization(abstraction_array)
        image = Image.new("RGBA", (abstraction_array.shape[0]*scale, abstraction_array.shape[1]*scale), (255,255,255,0))
        image_arr = np.asarray(image)
        symbols = Image.fromarray(Results.make_symbols(image_arr, map_arr, scale, 7).astype(np.uint8))
        overlay = Results.get_faded_overlay(Results.clear_resize(abstraction_image, scale), symbols, 100)
        overlay.show("Results.png")
        overlay.save("abs.png")

    def clear_resize(image, scale):
        old_image = np.asarray(image.convert("RGBA"))
        image_array = np.full((old_image.shape[0] * scale,old_image.shape[1] * scale, 4), 255)
        for x in range(old_image.shape[0]):
            for y in range(old_image.shape[1]):
                color = old_image[x, y]
                image_array[x*scale : x*scale + scale, y* scale : y*scale + scale] = color
        return Image.fromarray(image_array.astype(np.uint8))

    def get_random_color():
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        return np.array((r,g,b))

    def get_faded_overlay(background_im, foreground_im, opacity):
        white_back = Image.new("RGBA" , background_im.size, "WHITE")
        new_back = np.asanyarray(background_im.convert("RGBA"))
        new_back[:, :, 3] = opacity
        new_back_image = Image.fromarray(new_back.astype(np.uint8))
        transparent = foreground_im
        new_back_image.paste(transparent, (0,0), transparent)
        white_back.paste(new_back_image, (0,0), new_back_image)
        return white_back

    #________________________________________________
    
    def show_abstraction_result(self, experiment_number):
        map = self._map
        abstraction_map = self._abstraction_result[experiment_number]
        temp = {}
        for i in range ( len(abstraction_map)):
            for j in range ( len(abstraction_map)):
                temp [abstraction_map[i][j]] = True
        qtable = self._qtables[experiment_number] #returns a dict. states are keys, qvalues are the values
        Results.get_full_image(map, abstraction_map, 40)
        Results.get_qtable_heatmap(map, abstraction_map, 40, qtable)

    # ______________________________________________


def main():
    res = Results("grid_64x64_map1_adrl", exp_number = 1, baseline_included = False)
    #res.compare_bound("success rate", 40)
    res.show_abstraction_result(1)

if __name__ == '__main__':
    main()