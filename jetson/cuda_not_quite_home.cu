/*
 * Not quite home -- CUDA port of https://www.shadertoy.com/view/Mt33Wj
 * Earth-like planet shader by xwize (2016), based on Inigo Quilez's work.
 * Original notice: Created by inigo quilez - iq/2013
 * License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
 * https://creativecommons.org/licenses/by-nc-sa/3.0/
 * This adapted shader retains that license; the repository's MIT license
 * does not supersede it. CUDA/X11 adaptation, 2026.
 *
 * Uses the original USE_PROCEDURAL branch: no texture assets required.
 * Q/Escape quit, Space pause/resume, R restart.
 * --width N --height N --time S --output frame.ppm (headless rendering).
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#ifndef _WIN32
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#endif

#define D __device__ __forceinline__
struct V3
{
	float x, y, z;
	D V3(float a = 0) : x(a), y(a), z(a) {}
	D V3(float a, float b, float c) : x(a), y(b), z(c) {}
};
D V3 operator+(V3 a, V3 b) { return V3(a.x + b.x, a.y + b.y, a.z + b.z); }
D V3 operator-(V3 a, V3 b) { return V3(a.x - b.x, a.y - b.y, a.z - b.z); }
D V3 operator*(V3 a, V3 b) { return V3(a.x * b.x, a.y * b.y, a.z * b.z); }
D V3 operator*(V3 a, float b) { return V3(a.x * b, a.y * b, a.z * b); }
D V3 operator*(float a, V3 b) { return b * a; }
D V3 operator/(V3 a, float b) { return a * (1 / b); }
D V3 operator+(V3 a, float b) { return a + V3(b); }
D V3 operator-(V3 a, float b) { return a - V3(b); }
D V3 operator-(V3 a) { return a * (-1); }
D V3 &operator+=(V3 &a, V3 b)
{
	a = a + b;
	return a;
}
D V3 &operator*=(V3 &a, float b)
{
	a = a * b;
	return a;
}
D float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
D float length(V3 a) { return sqrtf(dot(a, a)); }
D V3 normalize(V3 a) { return a / fmaxf(length(a), 1e-12f); }
D V3 cross(V3 a, V3 b) { return V3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
D V3 reflect(V3 a, V3 n) { return a - 2 * dot(a, n) * n; }
D V3 abs3(V3 a) { return V3(fabsf(a.x), fabsf(a.y), fabsf(a.z)); }
D float fract(float a) { return a - floorf(a); }
D float mix(float a, float b, float t) { return a + (b - a) * t; }
D V3 mix(V3 a, V3 b, float t) { return a + (b - a) * t; }
D float clamp(float a) { return fminf(fmaxf(a, 0), 1); }

// GLSL matrices are specified column-major. This is m*q, not q*m.
D V3 rotateNoise(V3 q) { return V3(-.80f * q.y - .60f * q.z, .80f * q.x + .36f * q.y - .48f * q.z, .60f * q.x - .48f * q.y + .64f * q.z); }
D float hash(float n) { return fract(sinf(n) * 753.5453123f); }
D float noise(V3 x)
{
	V3 p(floorf(x.x), floorf(x.y), floorf(x.z));
	V3 f(fract(x.x), fract(x.y), fract(x.z));
	f = f * f * (V3(3) - 2 * f);
	float n = p.x + p.y * 157 + 113 * p.z;
	return mix(mix(mix(hash(n), hash(n + 1), f.x), mix(hash(n + 157), hash(n + 158), f.x), f.y),
			   mix(mix(hash(n + 113), hash(n + 114), f.x), mix(hash(n + 270), hash(n + 271), f.x), f.y), f.z);
}
D float starFunc(V3 p)
{
	float a = 0, previous = 0;
	for (int i = 0; i < 17; i++)
	{
		// The original divides by dot(p,p); guard its isolated singularity.
		p = abs3(p) / fmaxf(dot(p, p), 1e-12f) - .75f;
		float l = length(p);
		a += fabsf(l - previous);
		previous = l;
	}
	a /= 17;
	return a * a * a * .625f;
}
D V3 galaxy(V3 start, V3 dir)
{
	V3 p = start + dir * 100 + V3(0, 150, 0);
	V3 c = starFunc(p) * V3(1.2f, 1, 2);
	p += V3(10.1f, 15.2f, 10.4f);
	c += starFunc(p) * V3(1.2f, 1.2f, 3);
	p += V3(10.1f, 15.2f, 10.5f);
	return (c + starFunc(p) * V3(1.2f, 1.2f, 1.4f)) * .03f;
}
D float cloudForm(V3 q)
{
	q.x += sinf(q.y * 2);
	q.z += cosf(q.x * 2) * .3f;
	float f = -.15f + noise(q) + noise(q * 2) * .5f + noise(q * 4) * .25f + noise(q * 8) * .125f + noise(q * 16) * .05f + noise(q * 32) * .025f + noise(q * 64) * .0125f;
	return fminf(powf(fabsf(f), 10) * .5f, .5f);
}
D float octaves6(V3 q)
{
	float f = 0, t = 1;
	for (int i = 0; i < 6; i++)
	{
		f += noise(q * t) / t;
		t *= 2;
	}
	return f;
}
D V3 atmosphereGlow(V3 ro, V3 rd, V3 center)
{
	float glow = 0, di = (128.0f / 50) / 8;
	for (int i = 0; i < 8; i++)
	{
		V3 toCenter = center - (ro + rd * (i * di));
		float d = length(toCenter);
		float x = fabsf(d - 1), n = .01f + fmaxf(-toCenter.z / fmaxf(d, 1e-12f), 0);
		glow += 5.5f * expf(-x * 16) * n * di;
	}
	return glow * V3(0, .041f, .11f) * 1.5f;
}
D V3 shadePlanetSurface(V3 ro, V3 rd, V3 center, float t, float time)
{
	V3 pos = ro + t * rd, nor = normalize(pos - center), q = pos;
	float sun = fmaxf(0, .05f + .95f * nor.z);
	V3 col(0, .031f, .11f);
	col *= .5f + .5f * noise(q * 4);
	col *= .9f + .2f * noise(q * 8);
	col *= .9f + .2f * noise(q * 16);
	col *= .9f + .1f * noise(q * 32);
	col *= .9f + .1f * noise(q * 64);
	q = rotateNoise(q) * 2.01f;
	float hm = octaves6(q + 137) / 1.5f, f = hm * hm, landBlend = fminf(4 * fmaxf(f - .3f, 0), 1);
	float ng = expf(-fabsf(f - .4f) * 20), nd = expf(-fabsf(f - .6f) * 20), snowLevel = .8f + noise(q * 64) * .1f;
	float ns = expf(-fabsf(f - snowLevel) * 20);
	V3 snow = V3(.85f) / 1.85f, desert = V3(1.5f, 1.3f, 1) / 13, grass = V3(1, 1.3f, .5f) / 12;
	float grassD = 1 + (noise(q * 128) + noise(q * 64) + noise(q * 32)) * .5f;
	float snowD = 1 + (noise(q * V3(5, 1, 1) * 3) + noise(q * 16) * .5f + noise(q * 64) * .5f) * .25f;
	float desertD = 1 + noise(q * V3(1, 7, 1) * 3) + noise(q * 32) * .5f + noise(q * 64) * .5f;
	V3 land = (desertD * desert * nd + grassD * grass * ng + snowD * snow * ns) / fmaxf(ng + nd + ns, 1e-20f);
	col = mix(col, land, landBlend);
	float shore = clamp(1.42f * (1 - fabsf(landBlend - .4f)) - 1.2f);
	col += V3(0, .1f, .05f) * shore;
	// Reapply the original matrix before moving the cloud field over time.
	q = rotateNoise(q) * 2.01f + time * .03f;
	f = cloudForm(q + .1f);
	f = .2f + .8f * powf(1 - f * .75f, 2);
	col *= f;
	f = cloudForm(q);
	V3 cloudCol = V3(.2f) + .7f * noise(q * 2) + .4f * noise(q * 6) + .2f * noise(q * 16) + .05f * noise(q * 32) + .04f * noise(q * 64);
	cloudCol = (cloudCol + f * .05f) * .55f;
	col = mix(col, cloudCol * .7f, fminf(f * 1.4f, 1));
	col *= .005f + sun * 4;
	// Preserve the source's reflect(nor, +Z) argument order.
	V3 ref = reflect(nor, V3(0, 0, 1));
	float spec = .125f * powf(fmaxf(dot(ref, rd), 0), 512);
	col += spec * mix(col, V3(1), .05f) * sun;
	float depth = t * 2.7f / 2, atm = -fmaxf(-1, 1.1f - depth);
	return mix(col, V3(0, .041f, .11f) * 1.5f * sun * 4, atm * atm);
}
__global__ void renderPlanet(uchar4 *pixels, int width, int height, float time)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
		return;
	float px = (-width + 2 * (x + .5f)) / height, py = (-height + 2 * (y + .5f)) / height;
	float angle = .5f + .02f * time;
	V3 ro(2 * cosf(angle), 1, 2 * sinf(angle)), center(0, 1, 0);
	V3 ww = normalize(center - ro), uu = normalize(cross(ww, V3(0, 1, 0))), vv = normalize(cross(uu, ww));
	V3 rd = normalize(px * uu + py * vv + 1.7f * ww), ce = ro - center;
	float b = dot(rd, ce), c = dot(ce, ce) - 1, h = b * b - c;
	V3 background = galaxy(ro, rd) + atmosphereGlow(ro, rd, center), col = background;
	// Skip expensive surface noise for rays that miss the sphere. The source
	// computes it at t=10000 and then blends it out completely.
	if (h > 0)
	{
		float t = -b - sqrtf(h);
		col = mix(shadePlanetSurface(ro, rd, center, t, time), background, 1 - expf(-.00001f * t * t));
	}
	col = V3(sqrtf(fmaxf(col.x, 0)), sqrtf(fmaxf(col.y, 0)), sqrtf(fmaxf(col.z, 0)));
	pixels[x + (height - 1 - y) * width] = make_uchar4((unsigned char)(255 * clamp(col.z)), (unsigned char)(255 * clamp(col.y)), (unsigned char)(255 * clamp(col.x)), 255);
}

static void check(cudaError_t e, const char *call, int line)
{
	if (e != cudaSuccess)
	{
		fprintf(stderr, "CUDA line %d: %s: %s\n", line, call, cudaGetErrorString(e));
		exit(1);
	}
}
#define CUDA(call) check((call), #call, __LINE__)
#ifndef _WIN32
static double seconds()
{
	using namespace std::chrono;
	return duration<double>(steady_clock::now().time_since_epoch()).count();
}
#endif
struct Renderer
{
	uchar4 *pixels;
	int w, h;
	Renderer(int width, int height) : w(width), h(height) { CUDA(cudaMalloc(&pixels, size_t(w) * h * sizeof(uchar4))); }
	void frame(float time)
	{
		dim3 block(16, 8), grid((w + 15) / 16, (h + 7) / 8);
		renderPlanet<<<grid, block>>>(pixels, w, h, time);
		CUDA(cudaGetLastError());
	}
	~Renderer() { cudaFree(pixels); }
};

int main(int argc, char **argv)
{
	int w = 640, h = 360;
	float time = 0;
	const char *output = NULL;
	for (int i = 1; i < argc; i++)
	{
		if (!strcmp(argv[i], "--help"))
		{
			puts("cuda_not_quite_home [--width N] [--height N] [--time S] [--output file.ppm]\nKeys: Q/Escape quit, Space pause, R restart");
			return 0;
		}
		if (i + 1 >= argc)
		{
			fprintf(stderr, "Missing value for %s\n", argv[i]);
			return 1;
		}
		if (!strcmp(argv[i], "--width"))
			w = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--height"))
			h = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--time"))
			time = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--output"))
			output = argv[++i];
		else
		{
			fprintf(stderr, "Unknown option: %s\n", argv[i]);
			return 1;
		}
	}
	if (w < 16 || h < 16 || w > 4096 || h > 4096 || !std::isfinite(time) || time < 0)
	{
		fputs("Width/height must be 16..4096; time must be finite and nonnegative.\n", stderr);
		return 1;
	}
#ifdef _WIN32
	if (!output)
	{
		fputs("Use --output frame.ppm on Windows; interactive display uses X11 on Jetson.\n", stderr);
		return 1;
	}
#endif
#ifndef _WIN32
	Display *display = NULL;
	Window window = 0;
	GC gc = 0;
	XImage *image = NULL;
	Atom closeAtom = 0;
	if (!output)
	{
		display = XOpenDisplay(NULL);
		if (!display)
		{
			fputs("Cannot open X11 display. Run on the Jetson desktop or use --output frame.ppm.\n", stderr);
			return 1;
		}
		int screen = DefaultScreen(display);
		Visual *visual = DefaultVisual(display, screen);
		int depth = DefaultDepth(display, screen);
		if (visual->red_mask != 0xff0000 || visual->green_mask != 0xff00 || visual->blue_mask != 0xff)
		{
			fputs("Requires an RGB TrueColor X11 visual.\n", stderr);
			XCloseDisplay(display);
			return 1;
		}
		window = XCreateSimpleWindow(display, RootWindow(display, screen), 0, 0, w, h, 0, 0, 0);
		XSelectInput(display, window, ExposureMask | KeyPressMask);
		closeAtom = XInternAtom(display, "WM_DELETE_WINDOW", False);
		XSetWMProtocols(display, window, &closeAtom, 1);
		XSizeHints hints = {};
		hints.flags = PMinSize | PMaxSize;
		hints.min_width = hints.max_width = w;
		hints.min_height = hints.max_height = h;
		XSetWMNormalHints(display, window, &hints);
		XStoreName(display, window, "CUDA Not quite home");
		XMapWindow(display, window);
		gc = XCreateGC(display, window, 0, NULL);
		image = XCreateImage(display, visual, depth, ZPixmap, 0, NULL, w, h, 32, 0);
		if (!image || image->bits_per_pixel != 32 || image->byte_order != LSBFirst)
		{
			fputs("Requires a 32-bit little-endian XImage.\n", stderr);
			if (image)
				XDestroyImage(image);
			XFreeGC(display, gc);
			XDestroyWindow(display, window);
			XCloseDisplay(display);
			return 1;
		}
		image->data = (char *)calloc(image->bytes_per_line, h);
		if (!image->data)
		{
			fputs("Cannot allocate XImage.\n", stderr);
			XDestroyImage(image);
			XFreeGC(display, gc);
			XDestroyWindow(display, window);
			XCloseDisplay(display);
			return 1;
		}
	}
#endif
	cudaDeviceProp prop;
	CUDA(cudaGetDeviceProperties(&prop, 0));
	printf("Not quite home by xwize / Inigo Quilez | GPU: %s | %dx%d\n", prop.name, w, h);
	Renderer renderer(w, h);
	std::vector<uchar4> host(size_t(w) * h);
	if (output)
	{
		renderer.frame(time);
		CUDA(cudaMemcpy(host.data(), renderer.pixels, host.size() * sizeof(uchar4), cudaMemcpyDeviceToHost));
		FILE *f = fopen(output, "wb");
		if (!f)
		{
			perror(output);
			return 1;
		}
		bool ok = fprintf(f, "P6\n%d %d\n255\n", w, h) > 0;
		for (size_t i = 0; i < host.size() && ok; i++)
		{
			unsigned char rgb[] = {host[i].z, host[i].y, host[i].x};
			ok = fwrite(rgb, 1, 3, f) == 3;
		}
		if (fclose(f) != 0)
			ok = false;
		if (!ok)
		{
			fputs("Failed to write image.\n", stderr);
			return 1;
		}
		printf("Saved %s at t=%.3fs\n", output, time);
		return 0;
	}
#ifndef _WIN32
	bool running = true, paused = false;
	double last = seconds(), fpsStart = last;
	int frames = 0;
	puts("Q/Escape quit | Space pause | R restart");
	while (running)
	{
		while (XPending(display))
		{
			XEvent e;
			XNextEvent(display, &e);
			if (e.type == ClientMessage && (Atom)e.xclient.data.l[0] == closeAtom)
				running = false;
			if (e.type == KeyPress)
			{
				KeySym k = XLookupKeysym(&e.xkey, 0);
				if (k == XK_q || k == XK_Q || k == XK_Escape)
					running = false;
				else if (k == XK_space)
					paused = !paused;
				else if (k == XK_r || k == XK_R)
					time = 0;
			}
		}
		if (!running)
			break;
		double now = seconds();
		if (!paused)
			time += float(now - last);
		last = now;
		renderer.frame(time);
		CUDA(cudaMemcpy(host.data(), renderer.pixels, host.size() * sizeof(uchar4), cudaMemcpyDeviceToHost));
		for (int y = 0; y < h; y++)
			memcpy(image->data + y * image->bytes_per_line, host.data() + size_t(y) * w, size_t(w) * sizeof(uchar4));
		XPutImage(display, window, gc, image, 0, 0, 0, 0, w, h);
		XFlush(display);
		frames++;
		now = seconds();
		if (now - fpsStart >= 1)
		{
			char title[180];
			snprintf(title, sizeof(title), "Not quite home | %.1f FPS | t=%.1fs", frames / (now - fpsStart), time);
			XStoreName(display, window, title);
			frames = 0;
			fpsStart = now;
		}
	}
	XDestroyImage(image);
	XFreeGC(display, gc);
	XDestroyWindow(display, window);
	XCloseDisplay(display);
#endif
	return 0;
}
