import numpy as np
from random import shuffle, randint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math


result_path = '../result/'

sample_x_path = '../data/polydata_data_sampx.txt'
sample_y_path = '../data/polydata_data_sampy.txt'

gt_x_path = '../data/polydata_data_polyx.txt'
gt_y_path = '../data/polydata_data_polyy.txt'


cnt_sample_x_path = '../data/count_data_trainx.txt'
cnt_sample_y_path = '../data/count_data_trainy.txt'

cnt_gt_x_path = '../data/count_data_testx.txt'
cnt_gt_y_path = '../data/count_data_testy.txt'

max_counting_feature = [1.75, 2.07, 2.38, 3.49, 3.46, 2.37, 2.25, 3.1, 2.61]
min_counting_feature = [-1.5, -1.75, -2.32, -1.76, -1.39, -2.05, -1.67, -1.7, -2.2]
range_counting_feature = [max_counting_feature[i] - min_counting_feature[i] for i in range(9)]