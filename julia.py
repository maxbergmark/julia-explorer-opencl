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
from renderer import *
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
	# game = StandardKernel("kernels/if_else_kernel.cpp", debug, 1000)
	# game = StandardKernel("kernels/standard_minimum.cpp", debug, 1, 1000)
	game = StandardKernel("kernels/standard.cpp", debug, 1, 1000)
	# game = ImageBased("kernels/image_kernel.cpp", debug, 1000)
	# game = StandardKernel("kernels/standard_global_dims.cpp", debug, 1000)
	game.make_kernel()
	game.start_julia(x_dim, y_dim, x_lims, y_lims, 4)
	# game.load_mat_file("../puffer.mat")
	# game.decode_rle("patterns/twoglidersyntheses.rle")
	# game.decode_rle("patterns/workerbee_synth.rle")
	# game.decode_rle("glider.rle")
	# game.decode_rle("test.rle")
	# game.decode_rle("clock.rle")
	game.setup_opencl()
	if animate:
		p = Process(target = game.setup_plot)
		p.start()
	game.run_simulation()
	# game.print_stats()

if __name__ == "__main__":
	print(cl.device_fp_config.ROUND_TO_ZERO)
	run_single()
	# run_benchmark()
