#include <pyopencl-complex.h>
// #define CSR_FLUSH_TO_ZERO         (1 << 15)

kernel void calculate(__global float *a_g, float2 x_lims, float2 y_lims, float2 c0,
	__constant const int *dims, int max_iters, float radius) {

	// unsigned csr = __builtin_ia32_stmxcsr();
	// csr |= CSR_FLUSH_TO_ZERO;
	// __builtin_ia32_ldmxcsr(csr);

	int x_id = get_global_id(1);
	int y_id = get_global_id(0);
	int idx = dims[0]*y_id + x_id;
	int xd = dims[0];
	int yd = dims[1];
	// int n = xd * yd;

	float x = (float) x_id / xd;
	float y = (float) y_id / yd;

	float real = x_lims.x + (x_lims.y - x_lims.x) * x;
	float imag = y_lims.x + (y_lims.y - y_lims.x) * y;
	cfloat_t z = cfloat_new(real, imag);
	cfloat_t z_old = cfloat_new(real, imag);
	cfloat_t z0 = cfloat_new(real, imag);
	cfloat_t c = cfloat_new(c0.x, c0.y);
	float tmp = 0;
	float dist = 0;
	// while (tmp < max_iters) {
	// for (int i = 0; i < max_iters; i++) {
	// while (tmp < max_iters && cfloat_abs(z) < radius) {
	while (tmp < max_iters && dist < radius) {
		// z = cfloat_add(cfloat_mul(z, z), c);
		// z = cfloat_mul(z, z);
		// z = cfloat_add(cfloat_mul(z, z), z0);
		// z = cfloat_add(cfloat_pow(z, c), z0);
		// z = cfloat_add(cfloat_mul(z, z), cfloat_add(c, z0));
		dist += cfloat_abs(cfloat_sub(z, z_old));
		z_old = z;
		tmp++;
	}
	
	// if (tmp < max_iters) {
	if (dist > radius) {
		// tmp -= log2(log(cfloat_abs(z)) / log(radius));
		tmp -= log2(log(dist) / log(radius));
	} else {
		tmp = max_iters * dist / radius;
	}

	// a_g[idx] = dist;
	a_g[idx] = tmp;
}

kernel void render(__global float *a_g, 
	__global float *r_g, __constant const int *dims, int max_iters) {
	
	int x_id = get_global_id(1);
	int y_id = get_global_id(0);
	int idx = dims[0]*y_id + x_id;
	float a = a_g[idx] / max_iters;
	float a_t = pow(a, 0.3f);

	// r_g[3*idx+0] = 0;
	// r_g[3*idx+1] = pow(a/50.f, 0.9f);
	// r_g[3*idx+2] = 0;

	r_g[3*idx+0] = a_t * (0.5f + 0.5f * sin(6.28f * (a + 0.33f)));
	r_g[3*idx+1] = a_t * (0.5f + 0.5f * sin(6.28f * (a + 0.67f)));
	r_g[3*idx+2] = a_t * (0.5f + 0.5f * sin(6.28f * a));
}

