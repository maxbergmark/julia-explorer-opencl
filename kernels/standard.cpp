#include <pyopencl-complex.h>

kernel void calculate(__global float *a_g, float2 x_lims, float2 y_lims, float2 c0,
	__constant const int *dims, int max_iters, float radius) {

	int x_id = get_global_id(1);
	int y_id = get_global_id(0);
	int idx = dims[0]*y_id + x_id;
	int xd = dims[0];
	int yd = dims[1];

	float x = (float) x_id / xd;
	float y = (float) y_id / yd;

	float real = x_lims.x + (x_lims.y - x_lims.x) * x;
	float imag = y_lims.x + (y_lims.y - y_lims.x) * y;
	cfloat_t z = cfloat_new(real, imag);
	cfloat_t z0 = cfloat_new(real, imag);
	cfloat_t z_old = cfloat_new(real, imag);
	cfloat_t c = cfloat_new(c0.x, c0.y);
	float tmp = 0;
	float dist = 0;
	while (tmp < max_iters && dist < radius) {
		z = cfloat_add(cfloat_mul(z, z), c); // julia set
		// z = cfloat_add(cfloat_powr(cfloat_conj(z), 3), c); // tricorn
		// z = cfloat_add(cfloat_mul(cfloat_conj(z), cfloat_conj(z)), c); // tricorn
		// z = cfloat_add(cfloat_mul(z, z), z0); // mandelbrot
		// cfloat_t tmp_c = cfloat_new(fabs(z.real), fabs(z.imag)); //burning ship
		// z = cfloat_add(cfloat_mul(tmp_c, tmp_c), c); // burning ship
		dist += cfloat_abs(cfloat_sub(z, z_old));
		z_old = z;
		tmp += 1.f;
	}
	
	if (dist > radius) {
		tmp -= log2(log(dist) / log(radius));
		// tmp -= log(log(dist) / log(radius)) / log(16.f);
	} else {
		// tmp = 0;
		tmp = max_iters * dist / radius;
	}

	a_g[idx] = tmp;
}

kernel void render(__global float *a_g, 
	__global float *r_g, __constant const int *dims, int max_iters) {
	
	int x_id = get_global_id(1);
	int y_id = get_global_id(0);
	int idx = dims[0]*y_id + x_id;
	float a = a_g[idx];
	// float a = pow(a_g[idx] / max_iters, .3f);
	// float a = a_g[idx] / max_value;
	float a_n = a / max_iters;
	float a_t = pow(a_n, 0.3f);
	float a_s = pow(a_n, 1.0f);

	// r_g[3*idx+0] = 0;
	// r_g[3*idx+1] = a;
	// r_g[3*idx+2] = 0;

	r_g[3*idx+0] = a_t * (0.5f + 0.5f * sin(6.28f * (a_s + 0.33f)));
	r_g[3*idx+1] = a_t * (0.5f + 0.5f * sin(6.28f * (a_s + 0.67f)));
	r_g[3*idx+2] = a_t * (0.5f + 0.5f * sin(6.28f * a_s));
}

