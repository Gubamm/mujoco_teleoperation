"""Render-only GLFW window: no MuJoCo viewer UI, shortcuts, or mouse perturbations."""
import glfw
import mujoco
import numpy as np
from robot_app.instruction_ui import HEADER_HEIGHT, InstructionHeader

CAMERA_NAMES = ('front', 'top_camera', 'gripper')


class TeleopWindow:
    MOVEMENT_KEYS = frozenset('WSADQEIKJLUOGT')
    COMMAND_KEYS = frozenset('BNH')

    def __init__(self, model, key_callback):
        self.window = None
        self.context = None
        self.scene = None
        self.model = model
        self.instruction_header = InstructionHeader()
        self.key_callback = key_callback
        self.camera_ids = [model.camera(name).id for name in CAMERA_NAMES]
        if not glfw.init():
            raise RuntimeError('Cannot initialize GLFW. Run in a desktop session with a display.')
        try:
            self.window = glfw.create_window(900, 960, 'MuJoCo Teleoperation', None, None)
            if not self.window:
                raise RuntimeError('Cannot create teleoperation window')
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
            self.scene = mujoco.MjvScene(model, maxgeom=10000)
            self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self.options = mujoco.MjvOption()
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            glfw.set_key_callback(self.window, self._on_key)
            glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
            # Only the instruction button is clickable; no simulation perturbations.
        except BaseException:
            self.close()
            raise

    def _on_key(self, window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
        elif 0 <= key < 128:
            character = chr(key)
            if character in self.MOVEMENT_KEYS or (
                    character in self.COMMAND_KEYS and action == glfw.PRESS):
                self.key_callback(key)

    def _on_mouse_button(self, window, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
            return
        x, y = glfw.get_cursor_pos(window)
        win_width, win_height = glfw.get_window_size(window)
        width, height = glfw.get_framebuffer_size(window)
        if not win_width or not win_height:
            return
        x, y = x * width / win_width, y * height / win_height
        rect = self.instruction_header.button
        if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
            self.key_callback(ord('H'))

    def poll(self):
        glfw.poll_events()
        return not glfw.window_should_close(self.window)

    def _update_scene(self, data, camera_id):
        self.camera.fixedcamid = camera_id
        mujoco.mjv_updateScene(self.model, data, self.options, None, self.camera,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)

    def capture(self, data, size=(640, 480)):
        """Capture clean RGB images of all cameras at the exact same simulator state."""
        glfw.make_context_current(self.window)
        width, height = size
        if (self.context.offWidth, self.context.offHeight) != (width, height):
            mujoco.mjr_resizeOffscreen(width, height, self.context)
        frames = {}
        try:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
            viewport = mujoco.MjrRect(0, 0, width, height)
            for name, camera_id in zip(CAMERA_NAMES, self.camera_ids):
                self._update_scene(data, camera_id)
                mujoco.mjr_render(viewport, self.scene, self.context)
                rgb = np.empty((height, width, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(rgb, None, viewport, self.context)
                # OpenGL starts at the bottom; video frames start at the top.
                frames[name] = np.ascontiguousarray(rgb[::-1])
        finally:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
        return frames

    def render(self, data, recording=False, samples=0, instruction=''):
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width < 300 or height < 230:
            return
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
        full = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_rectangle(full, 0.08, 0.08, 0.08, 1)
        # Keep each view at 4:3: one large front view above two half-width views.
        gap = 4
        grid_width = max(4, min(width, (height - HEADER_HEIGHT - 110 - gap) * 8 // 9))
        left = (width - grid_width) // 2
        small_width = max(1, (grid_width - gap) // 2)
        small_height = max(1, small_width * 3 // 4)
        front_height = max(1, grid_width * 3 // 4)
        viewports = [
            mujoco.MjrRect(left, small_height + gap, grid_width, front_height),
            mujoco.MjrRect(left, 0, small_width, small_height),
            mujoco.MjrRect(left + grid_width - small_width, 0, small_width, small_height),
        ]
        for name, camera_id, viewport in zip(CAMERA_NAMES, self.camera_ids, viewports):
            self._update_scene(data, camera_id)
            mujoco.mjr_render(viewport, self.scene, self.context)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               viewport, name, '', self.context)
        status = 'RECORDING' if recording else 'Recording OFF'
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           mujoco.MjrRect(0, 0, width, height - HEADER_HEIGHT), f'{status} | {samples} samples | Front / Top / Gripper\n'
                           'TCP: W/S A/D Q/E XYZ   I/K J/L U/O rotation\n'
                           'G/T: close/open   B: record/save   N: reset   Esc: save and exit',
                           '', self.context)
        pixels = self.instruction_header.image(width, instruction)
        mujoco.mjr_drawPixels(pixels.ravel(), None,
                             mujoco.MjrRect(0, height - HEADER_HEIGHT, width, HEADER_HEIGHT),
                             self.context)
        glfw.swap_buffers(self.window)

    def close(self):
        if self.window is not None:
            glfw.make_context_current(self.window)
        if self.context is not None:
            self.context.free()
            self.context = None
        self.scene = None
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
