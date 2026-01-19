# CUDA Graphics Demos - Windows Port

This folder contains 18 CUDA graphics demos ported from Jetson Nano (Linux/X11) to Windows.

## Requirements

- **CUDA Toolkit 12.x** - The NVIDIA CUDA compiler (nvcc)
- **Visual Studio 2019/2022/2025** - Required for MSVC (cl.exe) as the host compiler
  - Install the "Desktop development with C++" workload
  - You can also use just the [Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
- **NVIDIA GPU** - Tested on RTX 3080 (Compute Capability 8.6)

> **Note:** CUDA on Windows requires MSVC as the host compiler. GCC/MinGW is not supported.

## Building

### Option 1: Using build.bat (Recommended)

The build script automatically sets up the MSVC environment:

```batch
# Build all demos
build.bat

# Build a specific demo
build.bat cuda_mandelbrot

# Clean build artifacts
build.bat clean
```

### Option 2: Manual Build

Open a **Developer Command Prompt** for Visual Studio, then:

```batch
nmake all
```

## Available Demos

| Demo | Description | Controls |
|------|-------------|----------|
| `cuda_render` | Plasma effect visualization | ESC to quit |
| `cuda_particles` | Particle system with trails | Mouse to attract, ESC to quit |
| `cuda_mandelbrot` | Interactive Mandelbrot fractal | Arrow keys to pan, +/- to zoom, ESC to quit |
| `cuda_3d_cube` | 3D bouncing ball inside a cube | Q/ESC to quit |
| `cuda_fluid` | Real-time fluid simulation | Mouse to stir, ESC to quit |
| `cuda_raymarcher` | SDF raymarching demo | 1-5 to change scenes, ESC to quit |
| `cuda_nbody` | N-body gravitational simulation | Space to reset, ESC to quit |
| `cuda_primitives` | Parallel primitives visualization | Space to cycle, ESC to quit |
| `cuda_functions` | Math function visualizer | 1-8 to change functions, ESC to quit |
| `cuda_life` | Conway's Game of Life | Space to pause, R to randomize, ESC to quit |
| `cuda_perlin` | Perlin noise terrain | ESC to quit |
| `cuda_flame` | Fire flame effect | ESC to quit |
| `cuda_tunnel` | Volumetric tunnel flythrough | ESC to quit |
| `cuda_cornell` | Cornell box path tracer | ESC to quit |
| `cuda_corridor` | Infinite corridor shader | ESC to quit |
| `cuda_fractal` | Kishimisu fractal zoom | ESC to quit |
| `cuda_pyramid` | Folding fractal pyramid | ESC to quit |
| `cuda_teapot` | Utah teapot software rasterizer | Arrow keys to rotate, ESC to quit |

## Technical Notes

### Porting Changes Summary

The following changes were made to port from Linux to Windows:

1. **X11 → Win32**: Replaced X11/Xlib windowing with native Win32 API
   - `win32_display.h` provides a cross-platform-ish abstraction
2. **Timing**: Replaced `gettimeofday()` with `QueryPerformanceCounter()`
3. **Sleep**: Replaced `usleep()` with `Sleep()`
4. **Keywords**: Renamed `near`/`far` parameters (reserved in `windows.h`)

### CUDA Kernels

All CUDA kernels remain **100% unchanged** from the original Jetson code. Only the host-side windowing and timing code was modified.

### GPU Architecture

The Makefile targets `sm_86` (RTX 30 series). To build for other GPUs, edit the Makefile:

| GPU | Architecture |
|-----|--------------|
| GTX 10xx | sm_61 |
| RTX 20xx | sm_75 |
| RTX 30xx | sm_86 |
| RTX 40xx | sm_89 |

## Troubleshooting

### "Cannot find compiler 'cl.exe'"
Run the build from a Developer Command Prompt, or use `build.bat`.

### "unsupported Microsoft Visual Studio version"
CUDA 12.9 officially supports VS 2017-2022. The `-allow-unsupported-compiler` flag is used for VS 2025.

### Display issues
Ensure your NVIDIA drivers are up to date. These demos use software blitting to a Win32 window.

---

## Detailed Porting Guide: Linux/X11 to Windows/Win32

This section documents the exact code changes required to port a CUDA graphics demo from Linux (X11) to Windows (Win32).

### 1. Header Replacements

**Remove these Linux headers:**
```cpp
// DELETE these:
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <sys/time.h>
#include <unistd.h>
```

**Add this Windows header:**
```cpp
// ADD this:
#include "win32_display.h"
```

### 2. Display/Window Types

| Linux (X11) | Windows (Win32) |
|-------------|-----------------|
| `Display*` | `Win32Display*` |
| `Window` | (embedded in `Win32Display`) |
| `GC` | (not needed) |
| `XImage*` | (not needed) |

### 3. Window Creation

**Linux:**
```cpp
Display* display = XOpenDisplay(NULL);
int screen = DefaultScreen(display);
Window window = XCreateSimpleWindow(display, RootWindow(display, screen),
    0, 0, WIDTH, HEIGHT, 1,
    BlackPixel(display, screen), WhitePixel(display, screen));
XSelectInput(display, window, ExposureMask | KeyPressMask | ButtonPressMask | PointerMotionMask);
XMapWindow(display, window);
GC gc = XCreateGC(display, window, 0, NULL);
XImage* image = XCreateImage(display, DefaultVisual(display, screen),
    24, ZPixmap, 0, (char*)pixels, WIDTH, HEIGHT, 32, 0);
```

**Windows:**
```cpp
Win32Display* display = win32_create_window("Window Title", WIDTH, HEIGHT);
```

### 4. Timing Functions

**Linux (`gettimeofday`):**
```cpp
double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}
// Usage:
double time = getTime();
```

**Windows (`QueryPerformanceCounter`):**
```cpp
// Use the built-in function from win32_display.h:
double time = win32_get_time(display);
```

**Linux (`clock_gettime`):**
```cpp
struct timespec ts;
clock_gettime(CLOCK_MONOTONIC, &ts);
double time = ts.tv_sec + ts.tv_nsec * 1e-9;
```

**Windows:**
```cpp
double time = win32_get_time(display);
```

### 5. Sleep/Delay Functions

| Linux | Windows |
|-------|---------|
| `usleep(16000)` (microseconds) | `win32_sleep_ms(16)` (milliseconds) |
| `sleep(1)` (seconds) | `win32_sleep_ms(1000)` |

### 6. Event Loop

**Linux:**
```cpp
while (running) {
    while (XPending(display)) {
        XEvent event;
        XNextEvent(display, &event);
        switch (event.type) {
            case KeyPress:
                if (XLookupKeysym(&event.xkey, 0) == XK_Escape)
                    running = 0;
                break;
            case ButtonPress:
                mouseX = event.xbutton.x;
                mouseY = event.xbutton.y;
                break;
            case MotionNotify:
                mouseX = event.xmotion.x;
                mouseY = event.xmotion.y;
                break;
        }
    }
    // render frame...
}
```

**Windows:**
```cpp
while (running) {
    win32_process_events(display);
    Win32Event event;
    while (win32_pop_event(display, &event)) {
        switch (event.type) {
            case WIN32_EVENT_KEY_PRESS:
                if (event.key == XK_Escape)
                    running = 0;
                break;
            case WIN32_EVENT_MOUSE_PRESS:
                mouseX = event.x;
                mouseY = event.y;
                break;
            case WIN32_EVENT_MOUSE_MOVE:
                mouseX = event.x;
                mouseY = event.y;
                break;
            case WIN32_EVENT_CLOSE:
                running = 0;
                break;
        }
    }
    // render frame...
}
```

### 7. Key Code Mappings

The `win32_display.h` header defines X11-style key constants that map to Win32 virtual keys:

| X11 Constant | Win32 Virtual Key | Value |
|--------------|-------------------|-------|
| `XK_Escape` | `VK_ESCAPE` | 0x1B |
| `XK_space` | `VK_SPACE` | 0x20 |
| `XK_Left` | `VK_LEFT` | 0x25 |
| `XK_Up` | `VK_UP` | 0x26 |
| `XK_Right` | `VK_RIGHT` | 0x27 |
| `XK_Down` | `VK_DOWN` | 0x28 |
| `XK_q` / `XK_Q` | `'Q'` | 0x51 |
| `XK_r` / `XK_R` | `'R'` | 0x52 |
| `XK_plus` | `VK_OEM_PLUS` | 0xBB |
| `XK_minus` | `VK_OEM_MINUS` | 0xBD |
| `XK_1` - `XK_9` | `'1'` - `'9'` | 0x31-0x39 |

### 8. Pixel Blitting

**Linux:**
```cpp
XPutImage(display, window, gc, image, 0, 0, 0, 0, WIDTH, HEIGHT);
```

**Windows:**
```cpp
win32_blit_pixels(display, pixels, WIDTH, HEIGHT);
```

**Pixel Format:** Both use BGRA 32-bit format (Blue in low byte, Alpha in high byte).

### 9. Window Title Updates

**Linux:**
```cpp
char title[256];
sprintf(title, "FPS: %.1f", fps);
XStoreName(display, window, title);
```

**Windows:**
```cpp
char title[256];
sprintf(title, "FPS: %.1f", fps);
SetWindowTextA(display->hwnd, title);
```

### 10. Cleanup

**Linux:**
```cpp
XDestroyImage(image);  // Note: this frees the pixel buffer too
XFreeGC(display, gc);
XDestroyWindow(display, window);
XCloseDisplay(display);
```

**Windows:**
```cpp
win32_destroy_window(display);
// Note: Does NOT free your pixel buffer - you must free it yourself
```

### 11. Reserved Keywords

The following identifiers are macros in `windows.h` and must be renamed:

| Problematic Name | Suggested Replacement |
|------------------|----------------------|
| `near` | `nearVal`, `nearPlane`, `zNear` |
| `far` | `farVal`, `farPlane`, `zFar` |

**Example fix:**
```cpp
// Linux (original):
void perspective(float* m, float fov, float aspect, float near, float far);

// Windows (fixed):
void perspective(float* m, float fov, float aspect, float nearVal, float farVal);
```

### 12. Complete Minimal Example

**Linux version:**
```cpp
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <sys/time.h>
#include <cuda_runtime.h>

__global__ void render(unsigned int* pixels, int w, int h, float t) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        unsigned char r = (unsigned char)(128 + 127 * sinf(x * 0.1f + t));
        unsigned char g = (unsigned char)(128 + 127 * sinf(y * 0.1f + t));
        pixels[y * w + x] = (r << 16) | (g << 8) | 255;
    }
}

int main() {
    Display* dpy = XOpenDisplay(NULL);
    Window win = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), 0, 0, 800, 600, 0, 0, 0);
    XMapWindow(dpy, win);
    // ... X11 setup and event loop ...
}
```

**Windows version:**
```cpp
#include "win32_display.h"
#include <cuda_runtime.h>

__global__ void render(unsigned int* pixels, int w, int h, float t) {
    // CUDA kernel unchanged!
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        unsigned char r = (unsigned char)(128 + 127 * sinf(x * 0.1f + t));
        unsigned char g = (unsigned char)(128 + 127 * sinf(y * 0.1f + t));
        pixels[y * w + x] = (r << 16) | (g << 8) | 255;
    }
}

int main() {
    Win32Display* display = win32_create_window("Demo", 800, 600);
    unsigned int* d_pixels;
    cudaMalloc(&d_pixels, 800 * 600 * 4);

    while (!display->shouldClose) {
        win32_process_events(display);
        Win32Event event;
        while (win32_pop_event(display, &event)) {
            if (event.type == WIN32_EVENT_KEY_PRESS && event.key == XK_Escape)
                display->shouldClose = 1;
        }

        float t = (float)win32_get_time(display);
        dim3 block(16, 16);
        dim3 grid((800 + 15) / 16, (600 + 15) / 16);
        render<<<grid, block>>>(d_pixels, 800, 600, t);
        cudaDeviceSynchronize();

        unsigned int* pixels = (unsigned int*)malloc(800 * 600 * 4);
        cudaMemcpy(pixels, d_pixels, 800 * 600 * 4, cudaMemcpyDeviceToHost);
        win32_blit_pixels(display, pixels, 800, 600);
        free(pixels);
    }

    cudaFree(d_pixels);
    win32_destroy_window(display);
    return 0;
}
```
