# Julia Set Explorer (OpenCL)

A realtime Julia set explorer written in Python using PyOpenCL and OpenCV. Accepts mouse input to modify the complex number input to the Julia set calculation. 

<p align="center">
	<img src="/examples/julia.png" width="80%" />
</p>


## Available fractals

There are currently four fractals available for exploration. To change between different fractals, uncomment the corresponding row in `kernels/standard.cpp`. 

- [Julia set](https://en.wikipedia.org/wiki/Julia_set)
- [Tricorn](https://en.wikipedia.org/wiki/Tricorn_(mathematics))
- [Mandelbrot](https://en.wikipedia.org/wiki/Mandelbrot_set)
- [Burning ship](https://en.wikipedia.org/wiki/Burning_Ship_fractal)

## Setup

Set up your `virtualenv`:

    python3 -m pip venv .venv
    source .venv/bin/activate

Install packages:

	pip3 install -r requirements.txt

Run the visualizer:

	python3 julia.py


## Controls

### Changing input number to the fractal

Change the input number using the mouse. The pixel which the mouse rests on is converted to a corresponding complex number. The only fractal not using the mouse input is the Mandelbrot set visualizer. 

### Freeze the mouse input

When a desired fractal is found, freeze the view using the `M` key. This will cause all further mouse movement to be ignored. 

### Move the view

Zoom in and out using the `-` and `+` keys. Move the view using the `WASD` keys. 

### Reset the view

Reset the view to the default zoom and position using the `0` key. 

### Change the maximum number of iterations

Change the maximum number of iterations for the fractal using `,` and `.`.

### Export the current fractal to an image

Use the `P` key to export the current view to a file. The file will be saved in the `images` directory, so make sure to create such a directory before saving. The filename can be found in the terminal when a save command is issued. 
