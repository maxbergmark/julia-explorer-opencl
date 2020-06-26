#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
from time import perf_counter as clock
from time import sleep
from multiprocessing import Process, Array, Value
import ctypes
import cv2
import sys
import os
from renderer import StandardKernel
import random

# matplotlib.use("TkAgg")
# np.set_printoptions(threshold=sys.maxsize)
# matplotlib.rcParams['figure.figsize'] = [10, 10]



def run_single():
	x_dim, y_dim = 1920*1, 1080*1
	x_lims = [-2, 2]
	y_lims = [-1.5, 1.5]
	animate = True
	debug = False
	game = StandardKernel("kernels/standard.cpp", debug, 1, 1000)
	game.make_kernel()
	game.start_julia(x_dim, y_dim, x_lims, y_lims, 1)
	game.setup_opencl()
	if animate:
		p = Process(target = game.setup_plot)
		p.start()
	game.run_simulation()

if __name__ == "__main__":
	run_single()
	# run_benchmark()
