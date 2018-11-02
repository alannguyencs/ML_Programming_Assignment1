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