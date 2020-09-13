import numpy as np
from numba import jit
import csv
import json
import sys
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
#from choose_strategy_functions import *
#from graph_generation import *

data_file_path = "test.csv"
with open(data_file_path, 'w', newline='\n') as output_file:
                writer = csv.writer(output_file)
                writer.writerow("Testing")
