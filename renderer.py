import numpy as np
import pyopencl as cl
from time import perf_counter as clock
from time import sleep
from multiprocessing import Process, Array, Value
import ctypes
import cv2
import uuid
# import re
# import sys
# import os

class JuliaRenderer:
	def __init__(self, filename, debug, resize = 1, iterations = 0):
		self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx, 
			properties=cl.command_queue_properties.PROFILING_ENABLE)
		self.debug = debug
		self.filename = filename
		self.running = Value(ctypes.c_bool, False)
		self.iterations = iterations
		self.resize = resize
		# self.max_iters = np.int32(10)
		self.max_iters = Value(ctypes.c_int32, 200)
		self.radius = Value(ctypes.c_float, 1e2)
		self.fps = Value(ctypes.c_float, 0)

	def start_julia(self, xdim, ydim):
		pass

	def setup_opencl(self):
		pass

	def make_kernel(self):

		kernel = open(self.filename, "r").read()
		t0 = clock()
		self.prg = cl.Program(self.ctx, kernel).build(["-cl-fast-relaxed-math"])
		t1 = clock()
		self.compilation_time = t1-t0

	def print_stats(self):
		print()
		print("\tStats for %s" % self.filename)
		print("\tCompilation time: %.3f ms" % (1e3*self.compilation_time,))
		print("\tKernel time: %.3f ms" % (1e3*self.kernel_time,))
		self.queue.finish()
		# cl.enqueue_copy(self.queue, self.b_np, self.b_g)
		# self.queue.finish()
		# print("\tChecksum: %d" % (self.b_np.sum(),))
		print()

	def calc_mouse_pos(self, xlim, ylim):
		x_f = self.mouse_pixel[0] / self.xdim / self.resize
		y_f = self.mouse_pixel[1] / self.ydim / self.resize
		self.mouse_pos[0] = xlim[0] + (xlim[1] - xlim[0]) * x_f
		self.mouse_pos[1] = ylim[0] + (ylim[1] - ylim[0]) * y_f
		delta = self.mouse_pos - self.center_pos
		# print(self.mouse_pos, self.center)
		if self.move_enabled:
			# self.c0[:] = self.mouse_pos
			self.c0[:] = self.center + delta
		else:
			self.center[:] = self.c0


	def save_mouse_position(self, event, x, y, flags, param):
		self.mouse_pixel[:] = [x, y]
		self.calc_mouse_pos(self.xlim, self.ylim)

	def zoom_in(self):
		self.target_xlim[0] = 0.8 * self.target_xlim[0] + 0.2 * self.mouse_pos[0]
		self.target_xlim[1] = 0.8 * self.target_xlim[1] + 0.2 * self.mouse_pos[0]
		self.target_ylim[0] = 0.8 * self.target_ylim[0] + 0.2 * self.mouse_pos[1]
		self.target_ylim[1] = 0.8 * self.target_ylim[1] + 0.2 * self.mouse_pos[1]

	def zoom_out(self):
		self.target_xlim[0] = 1.2 * self.target_xlim[0] - 0.2 * self.mouse_pos[0]
		self.target_xlim[1] = 1.2 * self.target_xlim[1] - 0.2 * self.mouse_pos[0]
		self.target_ylim[0] = 1.2 * self.target_ylim[0] - 0.2 * self.mouse_pos[1]
		self.target_ylim[1] = 1.2 * self.target_ylim[1] - 0.2 * self.mouse_pos[1]

	def reset_zoom(self):
		self.target_xlim[:] = self.xlim_orig
		self.target_ylim[:] = self.ylim_orig
		self.center[:] = [0, 0]
		self.calc_mouse_pos(self.target_xlim, self.target_ylim)

	def update_view(self):
		self.xlim[:] = 0.8 * self.xlim + 0.2 * self.target_xlim
		self.ylim[:] = 0.8 * self.ylim + 0.2 * self.target_ylim
		self.center_pos[0] = 0.5 * (self.xlim[0] + self.xlim[1])
		self.center_pos[1] = 0.5 * (self.ylim[0] + self.ylim[1])

	def save_image(self):
		filename = 'images/' + str(uuid.uuid4()) + '.png'
		iters = np.int32(self.max_iters.value)
		radius = np.float32(self.radius.value)
		print("Saving image %s..." % filename)

		self.prg.calculate(
			self.queue, self.large_a_np.shape, None, 
			self.large_a_g, self.xlim, self.ylim, self.c0, 
			self.large_dims_g, iters, radius)

		cl.enqueue_copy(self.queue, self.large_a_np, self.large_a_g)
		self.queue.finish()
		self.prg.render(self.queue, self.large_a_np.shape, None, 
			self.large_a_g, self.large_render_g, self.large_dims_g, iters)
		cl.enqueue_copy(self.queue, self.large_render_np, self.large_render_g)
		self.queue.finish()



		cv2.imwrite(filename, self.large_render_np * 255)
		print("Saved image %s" % filename)

	def check_movement(self, key):
		zoom = self.xlim[1] - self.xlim[0]
		step = 0.1
		if key == ord('w'):
			self.target_ylim -= step * zoom
			self.mouse_pos[1] -= step * zoom
		if key == ord('a'):
			self.target_xlim -= step * zoom
			self.mouse_pos[0] -= step * zoom
		if key == ord('s'):
			self.target_ylim += step * zoom
			self.mouse_pos[1] += step * zoom
		if key == ord('d'):
			self.target_xlim += step * zoom
			self.mouse_pos[0] += step * zoom

	def handle_keypress(self, key):
		if key == ord('q'):
			return False
		elif key == ord('p'):
			self.save_image()
		elif key == ord('+'):
			self.zoom_in()
		elif key == ord('-'):
			self.zoom_out()
		elif key == ord('0'):
			self.reset_zoom()
		elif key == ord(','):
			p = (len(str(self.max_iters.value-1)) - 1)
			self.max_iters.value -= 10 ** p
			self.max_iters.value = max(1, self.max_iters.value)
		elif key == ord('.'):
			p = (len(str(self.max_iters.value-0)) - 1)
			self.max_iters.value += 10 ** p
			self.max_iters.value = min(100000, self.max_iters.value)
		elif key == ord('k'):
			p = (len(str(self.radius.value-1)) - 3)
			self.radius.value -= 10 ** p
			self.radius.value = max(1, self.radius.value)
			# print("\n%.3f\t%d\n" % (self.radius.value, p))
		elif key == ord('l'):
			p = (len(str(self.radius.value-0)) - 3)
			self.radius.value += 10 ** p
			self.radius.value = min(1e5, self.radius.value)
			# print("\n%.3f\t%d\n" % (self.radius.value, p))
		elif key == ord('m'):
			self.move_enabled = not self.move_enabled
		else:
			self.check_movement(key)
		return True

	def setup_plot(self):
		name = "Julia Set Explorer"
		cv2.namedWindow(name)
		cv2.setMouseCallback(name, self.save_mouse_position)
		self.mouse_pos = np.array(self.c0)
		self.center = np.array([0.0, 0.0])
		self.center_pos = np.array([0.0, 0.0])
		self.mouse_pixel = np.array(self.c0)
		self.move_enabled = True
		running = True
		while running:
			key = cv2.waitKey(10)
			running = self.handle_keypress(key)
			self.update_view()

			res = cv2.resize(self.res_np, 
				dsize = (int(self.xdim * self.resize), int(self.ydim * self.resize)), 
				interpolation = cv2.INTER_LANCZOS4)
			iters_info = "max_iters: %d" % (self.max_iters.value,)
			radius_info = "radius: %d" % (self.radius.value,)
			c0_info = "c: %.8f + %.8fi" % (self.c0[0], self.c0[1])
			fps_info = "fps: %.1f" % (self.fps.value-0)
			cv2.putText(res, iters_info, (16,32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
			cv2.putText(res, radius_info, (16,64), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
			cv2.putText(res, c0_info, (16,96), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
			cv2.putText(res, fps_info, (16,128), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
			cv2.imshow(name, res)
		self.running.value = False

	def run_simulation(self):
		pass

	def print_fps(self, stat_frequency):
		# self.queue.finish()
		temp_fps = stat_frequency / (clock()-self.fps_clock)
		self.fps.value = 0.9 * self.fps.value + 0.1 * temp_fps
		# print("\r\tfps: %8.3f / %8.3f" % (self.fps, temp_fps), end = "")

class StandardKernel(JuliaRenderer):

	def setup_limits(self, x_lim, y_lim):
		self.multi_xlim = Array(ctypes.c_float, 2)
		self.multi_ylim = Array(ctypes.c_float, 2)
		self.xlim = np.ctypeslib.as_array(self.multi_xlim.get_obj())
		self.ylim = np.ctypeslib.as_array(self.multi_ylim.get_obj())
		self.xlim[:] = x_lim
		self.ylim[:] = y_lim
		self.multi_target_xlim = Array(ctypes.c_float, 2)
		self.multi_target_ylim = Array(ctypes.c_float, 2)
		self.target_xlim = np.ctypeslib.as_array(self.multi_target_xlim.get_obj())
		self.target_ylim = np.ctypeslib.as_array(self.multi_target_ylim.get_obj())
		self.target_xlim[:] = x_lim
		self.target_ylim[:] = y_lim
		self.xlim_orig = x_lim
		self.ylim_orig = y_lim

	def start_julia(self, x_dim, y_dim, x_lim, y_lim, render_upscale):
		
		self.xdim, self.ydim = x_dim, y_dim
		self.render_xdim = x_dim * render_upscale
		self.render_ydim = y_dim * render_upscale
		self.setup_limits(x_lim, y_lim)
		self.multi_c0 = Array(ctypes.c_float, 2)
		self.c0 = np.ctypeslib.as_array(self.multi_c0.get_obj())
		self.c0[:] = [0.4, 0.3]

		self.a_np = np.zeros((y_dim, x_dim)).astype(np.float32)

	def setup_opencl(self):
		mf = cl.mem_flags
		self.dims_np = np.array([self.xdim, self.ydim], dtype = np.int32)
		self.a_g = cl.Buffer(self.ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.a_np)
		self.dims_g = cl.Buffer(self.ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.dims_np)
		self.multi_arr = Array(ctypes.c_float, self.ydim * self.xdim * 3)
		self.res_np = np.ctypeslib.as_array(self.multi_arr.get_obj())
		self.res_np.shape = (self.ydim, self.xdim, 3)
		self.render_g = cl.Buffer(self.ctx, 
			mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=self.res_np)
		self.running.value = True


		self.large_dims_np = np.array([self.render_xdim, self.render_ydim], dtype = np.int32)
		self.large_dims_g = cl.Buffer(self.ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.large_dims_np)
		self.large_a_np = np.zeros(
			(self.render_ydim, self.render_xdim)).astype(np.float32)
		self.large_a_g = cl.Buffer(self.ctx, 
			mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.large_a_np)
		self.large_render_np = np.zeros(
			(self.render_ydim, self.render_xdim, 3), dtype = np.float32)
		self.large_render_g = cl.Buffer(self.ctx, 
			mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf = self.large_render_np)

	def handle_data_presentation(self, event, 
		render_frequency, stat_frequency, iters):
		# event.wait()
		# event_time = (event.profile.end-event.profile.start)*1e-6
		# print("\r%7.3f" % (event_time,), end = "")
		if self.frame % render_frequency == 0:
			self.render_frame(iters)
		if self.frame % stat_frequency == 0:
			self.print_fps(stat_frequency)
			self.fps_clock = clock()
		if self.debug:
			self.print_debug()

	def run_simulation(self):
		self.frame = 0
		t0 = clock()
		render_frequency = 1
		stat_frequency = 1
		self.fps_clock = clock()

		while self.running.value:
		# for _ in range(self.iterations):
			iters = np.int32(self.max_iters.value)
			radius = np.float32(self.radius.value)
			event = self.prg.calculate(
				self.queue, self.a_np.shape, None, 
				self.a_g, self.xlim, self.ylim, self.c0, 
				self.dims_g, iters, radius)
			self.handle_data_presentation(event, 
				render_frequency, stat_frequency, iters)
			self.frame += 1

		self.queue.finish()
		t1 = clock()
		self.kernel_time = t1-t0

	def render_frame(self, iters):
		cl.enqueue_copy(self.queue, self.a_np, self.a_g)
		self.queue.finish()
		self.prg.render(self.queue, self.a_np.shape, None, 
			self.a_g, self.render_g, self.dims_g, iters)
		cl.enqueue_copy(self.queue, self.res_np, self.render_g)
		self.queue.finish()
		# print("\t%.1e\t%.1e" % (self.a_np.sum(), (self.a_np**2).sum()), end = "")


	def print_debug(self):
		cl.enqueue_copy(self.queue, self.a_np, self.a_g)
		self.queue.finish()
		print()
		print(self.a_np)
		print(self.a_np.min(), self.a_np.max(), self.a_np.sum())
