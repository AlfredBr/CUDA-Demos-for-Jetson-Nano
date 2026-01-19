/*
 * Win32 Display Header for CUDA Graphics Demos
 *
 * This provides a simple abstraction layer that replaces X11 functionality
 * for porting CUDA graphics demos from Linux to Windows.
 *
 * Usage:
 *   1. Call win32_create_window() to create a window
 *   2. Call win32_get_time() for high-resolution timing
 *   3. Call win32_blit_pixels() to display your CUDA-rendered framebuffer
 *   4. Call win32_process_events() in your main loop
 *   5. Call win32_destroy_window() when done
 */

#ifndef WIN32_DISPLAY_H
#define WIN32_DISPLAY_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

// Key codes (matching X11 keysym names for easier porting)
#define XK_Escape   VK_ESCAPE
#define XK_q        'Q'
#define XK_r        'R'
#define XK_c        'C'
#define XK_space    VK_SPACE
#define XK_Left     VK_LEFT
#define XK_Right    VK_RIGHT
#define XK_Up       VK_UP
#define XK_Down     VK_DOWN
#define XK_plus     VK_OEM_PLUS
#define XK_minus    VK_OEM_MINUS
#define XK_equal    VK_OEM_PLUS
#define XK_w        'W'
#define XK_a        'A'
#define XK_s        'S'
#define XK_d        'D'
#define XK_1        '1'
#define XK_2        '2'
#define XK_3        '3'
#define XK_4        '4'
#define XK_5        '5'
#define XK_6        '6'
#define XK_7        '7'
#define XK_8        '8'
#define XK_9        '9'
#define XK_0        '0'
#define XK_p        'P'
#define XK_g        'G'
#define XK_t        'T'
#define XK_f        'F'
#define XK_h        'H'
#define XK_l        'L'
#define XK_m        'M'

// Mouse buttons
#define Button1     1  // Left
#define Button2     2  // Middle
#define Button3     3  // Right

// Event types
typedef enum {
    WIN32_EVENT_NONE = 0,
    WIN32_EVENT_KEY_PRESS,
    WIN32_EVENT_KEY_RELEASE,
    WIN32_EVENT_MOUSE_PRESS,
    WIN32_EVENT_MOUSE_RELEASE,
    WIN32_EVENT_MOUSE_MOVE,
    WIN32_EVENT_CLOSE,
    WIN32_EVENT_RESIZE
} Win32EventType;

typedef struct {
    Win32EventType type;
    int key;           // Key code for keyboard events
    int button;        // Button number for mouse events
    int mouseX;        // Mouse X position
    int mouseY;        // Mouse Y position
    int width;         // Window width (for resize events)
    int height;        // Window height (for resize events)
} Win32Event;

// Display context
typedef struct {
    HWND hwnd;
    HDC hdc;
    HDC memDC;
    HBITMAP bitmap;
    BITMAPINFO bmi;
    int width;
    int height;
    unsigned char* pixels;  // Points to bitmap bits
    int shouldClose;

    // Event queue
    Win32Event events[64];
    int eventHead;
    int eventTail;

    // Mouse state
    int mouseX;
    int mouseY;
    int mouseButtons;

    // High-resolution timer
    LARGE_INTEGER timerFreq;
    LARGE_INTEGER timerStart;
} Win32Display;

// Global display pointer for window procedure
static Win32Display* g_display = NULL;

// Push an event to the queue
static void win32_push_event(Win32Display* disp, Win32Event* event) {
    int next = (disp->eventTail + 1) % 64;
    if (next != disp->eventHead) {
        disp->events[disp->eventTail] = *event;
        disp->eventTail = next;
    }
}

// Window procedure
static LRESULT CALLBACK win32_window_proc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    Win32Display* disp = g_display;
    Win32Event event;
    memset(&event, 0, sizeof(Win32Event));

    switch (msg) {
        case WM_CLOSE:
            event.type = WIN32_EVENT_CLOSE;
            if (disp) {
                disp->shouldClose = 1;
                win32_push_event(disp, &event);
            }
            return 0;

        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;

        case WM_KEYDOWN:
            event.type = WIN32_EVENT_KEY_PRESS;
            event.key = (int)wParam;
            if (disp) win32_push_event(disp, &event);
            return 0;

        case WM_KEYUP:
            event.type = WIN32_EVENT_KEY_RELEASE;
            event.key = (int)wParam;
            if (disp) win32_push_event(disp, &event);
            return 0;

        case WM_LBUTTONDOWN:
            event.type = WIN32_EVENT_MOUSE_PRESS;
            event.button = Button1;
            event.mouseX = LOWORD(lParam);
            event.mouseY = HIWORD(lParam);
            if (disp) {
                disp->mouseButtons |= 1;
                win32_push_event(disp, &event);
            }
            return 0;

        case WM_RBUTTONDOWN:
            event.type = WIN32_EVENT_MOUSE_PRESS;
            event.button = Button3;
            event.mouseX = LOWORD(lParam);
            event.mouseY = HIWORD(lParam);
            if (disp) {
                disp->mouseButtons |= 4;
                win32_push_event(disp, &event);
            }
            return 0;

        case WM_MBUTTONDOWN:
            event.type = WIN32_EVENT_MOUSE_PRESS;
            event.button = Button2;
            event.mouseX = LOWORD(lParam);
            event.mouseY = HIWORD(lParam);
            if (disp) {
                disp->mouseButtons |= 2;
                win32_push_event(disp, &event);
            }
            return 0;

        case WM_LBUTTONUP:
            event.type = WIN32_EVENT_MOUSE_RELEASE;
            event.button = Button1;
            event.mouseX = LOWORD(lParam);
            event.mouseY = HIWORD(lParam);
            if (disp) {
                disp->mouseButtons &= ~1;
                win32_push_event(disp, &event);
            }
            return 0;

        case WM_RBUTTONUP:
            event.type = WIN32_EVENT_MOUSE_RELEASE;
            event.button = Button3;
            event.mouseX = LOWORD(lParam);
            event.mouseY = HIWORD(lParam);
            if (disp) {
                disp->mouseButtons &= ~4;
                win32_push_event(disp, &event);
            }
            return 0;

        case WM_MBUTTONUP:
            event.type = WIN32_EVENT_MOUSE_RELEASE;
            event.button = Button2;
            event.mouseX = LOWORD(lParam);
            event.mouseY = HIWORD(lParam);
            if (disp) {
                disp->mouseButtons &= ~2;
                win32_push_event(disp, &event);
            }
            return 0;

        case WM_MOUSEMOVE:
            event.type = WIN32_EVENT_MOUSE_MOVE;
            event.mouseX = LOWORD(lParam);
            event.mouseY = HIWORD(lParam);
            if (disp) {
                disp->mouseX = event.mouseX;
                disp->mouseY = event.mouseY;
                win32_push_event(disp, &event);
            }
            return 0;

        case WM_SIZE:
            event.type = WIN32_EVENT_RESIZE;
            event.width = LOWORD(lParam);
            event.height = HIWORD(lParam);
            if (disp) win32_push_event(disp, &event);
            return 0;

        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            if (disp && disp->memDC) {
                BitBlt(hdc, 0, 0, disp->width, disp->height, disp->memDC, 0, 0, SRCCOPY);
            }
            EndPaint(hwnd, &ps);
            return 0;
        }
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// Create a window
static Win32Display* win32_create_window(const char* title, int width, int height) {
    Win32Display* disp = (Win32Display*)calloc(1, sizeof(Win32Display));
    if (!disp) return NULL;

    disp->width = width;
    disp->height = height;

    // Initialize high-resolution timer
    QueryPerformanceFrequency(&disp->timerFreq);
    QueryPerformanceCounter(&disp->timerStart);

    // Register window class
    WNDCLASSEXA wc = {0};
    wc.cbSize = sizeof(WNDCLASSEXA);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = win32_window_proc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = "CUDADemoWindow";
    RegisterClassExA(&wc);

    // Calculate window size including borders
    RECT rect = {0, 0, width, height};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    // Create window
    disp->hwnd = CreateWindowExA(
        0,
        "CUDADemoWindow",
        title,
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rect.right - rect.left, rect.bottom - rect.top,
        NULL, NULL,
        GetModuleHandle(NULL),
        NULL
    );

    if (!disp->hwnd) {
        free(disp);
        return NULL;
    }

    // Set global pointer for window procedure
    g_display = disp;

    // Get DC
    disp->hdc = GetDC(disp->hwnd);
    disp->memDC = CreateCompatibleDC(disp->hdc);

    // Create DIB section for pixel access
    ZeroMemory(&disp->bmi, sizeof(BITMAPINFO));
    disp->bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    disp->bmi.bmiHeader.biWidth = width;
    disp->bmi.bmiHeader.biHeight = -height;  // Negative for top-down
    disp->bmi.bmiHeader.biPlanes = 1;
    disp->bmi.bmiHeader.biBitCount = 32;
    disp->bmi.bmiHeader.biCompression = BI_RGB;

    disp->bitmap = CreateDIBSection(disp->memDC, &disp->bmi, DIB_RGB_COLORS,
                                     (void**)&disp->pixels, NULL, 0);
    SelectObject(disp->memDC, disp->bitmap);

    // Show window
    ShowWindow(disp->hwnd, SW_SHOW);
    UpdateWindow(disp->hwnd);

    return disp;
}

// Destroy window
static void win32_destroy_window(Win32Display* disp) {
    if (!disp) return;

    if (disp->bitmap) DeleteObject(disp->bitmap);
    if (disp->memDC) DeleteDC(disp->memDC);
    if (disp->hdc) ReleaseDC(disp->hwnd, disp->hdc);
    if (disp->hwnd) DestroyWindow(disp->hwnd);

    g_display = NULL;
    free(disp);
}

// Process pending events, returns 0 if should continue, 1 if should quit
static int win32_process_events(Win32Display* disp) {
    MSG msg;
    while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            disp->shouldClose = 1;
            return 1;
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return disp->shouldClose;
}

// Check if there are events in the queue
static int win32_has_events(Win32Display* disp) {
    return disp->eventHead != disp->eventTail;
}

// Pop an event from the queue
static int win32_pop_event(Win32Display* disp, Win32Event* event) {
    if (disp->eventHead == disp->eventTail) return 0;
    *event = disp->events[disp->eventHead];
    disp->eventHead = (disp->eventHead + 1) % 64;
    return 1;
}

// Blit pixels to window (pixels should be BGRA format)
static void win32_blit_pixels(Win32Display* disp, unsigned char* pixels) {
    if (!disp || !pixels) return;

    // Copy pixels to DIB
    memcpy(disp->pixels, pixels, disp->width * disp->height * 4);

    // Blit to window
    BitBlt(disp->hdc, 0, 0, disp->width, disp->height, disp->memDC, 0, 0, SRCCOPY);
}

// Get time in seconds since window creation
static double win32_get_time(Win32Display* disp) {
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)(now.QuadPart - disp->timerStart.QuadPart) / (double)disp->timerFreq.QuadPart;
}

// Sleep for milliseconds
static void win32_sleep_ms(int ms) {
    Sleep(ms);
}

// Get mouse position
static void win32_get_mouse(Win32Display* disp, int* x, int* y) {
    if (x) *x = disp->mouseX;
    if (y) *y = disp->mouseY;
}

// Check if window should close
static int win32_should_close(Win32Display* disp) {
    return disp->shouldClose;
}

#endif // WIN32_DISPLAY_H
