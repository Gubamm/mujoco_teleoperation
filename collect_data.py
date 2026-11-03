"""Keyboard Cartesian teleoperation and state/action episode recording."""
import argparse
import fcntl
import json
from pathlib import Path
from queue import Empty, SimpleQueue
import time
from uuid import uuid4

import mujoco
import imageio_ffmpeg
import numpy as np

from robot_app.robot_env import ARM_JOINTS, RobotEnv
from robot_app.instruction_ui import edit_instruction
from robot_app.teleop_window import CAMERA_NAMES, TeleopWindow

HELP = '''
Click the teleoperation window before using these keys (one increment per key event).
  W/S: TCP +X/-X    A/D: TCP +Y/-Y    Q/E: TCP +Z/-Z
  I/K: TCP +Rx/-Rx  J/L: TCP +Ry/-Ry  U/O: TCP +Rz/-Rz
  G: close gripper by 0.1    T: open gripper by 0.1
  B: start/stop recording   N: save recording and reset to home
  R: save/stop recording and randomize tray, banana, apple
  H or the Instruction button: edit the task instruction
  All three cameras are shown together. Esc: save and exit
Close the window or press Ctrl+C to save the current recording and exit.
'''


class EpisodeRecorder:
    def __init__(self, output, env, instruction='', video_size=None):
        self.output = Path(output)
        self.env = env
        self.instruction = instruction
        self.video_size = video_size
        if video_size is not None and (len(video_size) != 2 or any(
                not isinstance(v, int) or v <= 0 or v % 2 for v in video_size)):
            raise ValueError('Video width and height must be positive even integers')
        self.writers = {}
        self.pending = None
        self.destination = None
        self.rows = []
        self.recording = False
        self.initial_qpos = None
        self.initial_qvel = None
        self.initial_ctrl = None

    def start(self):
        if self.recording:
            return
        self.rows = []
        self.pending = self.output / ('.episode_' + uuid4().hex + '.partial')
        self.episode_instruction = self.instruction
        self.initial_qpos = self.env.data.qpos.copy()
        self.initial_qvel = self.env.data.qvel.copy()
        self.initial_ctrl = self.env.data.ctrl.copy()
        self.recording = True
        print('RECORDING started', flush=True)

    def change_instruction(self, instruction):
        if instruction is None or instruction == self.instruction:
            return False
        was_recording = self.recording
        if was_recording:
            self.save()
        self.instruction = instruction
        if was_recording:
            self.start()
        print(f'INSTRUCTION: {self.instruction or "(unset)"}', flush=True)
        return True

    def _close_videos(self):
        errors = []
        for writer in self.writers.values():
            try:
                writer.close()
            except Exception as exc:
                errors.append(exc)
        self.writers = {}
        if errors:
            raise errors[0]

    def append(self, transition, frames=None):
        if not self.recording:
            return
        if self.video_size is not None:
            width, height = self.video_size
            if frames is None or set(frames) != set(CAMERA_NAMES):
                raise ValueError('Every sample needs all three camera frames')
            for frame in frames.values():
                if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                    raise ValueError('Expected uint8 RGB frames with the configured video size')
            try:
                if not self.writers:
                    self.pending.mkdir(parents=True, exist_ok=False)
                    for name in CAMERA_NAMES:
                        writer = imageio_ffmpeg.write_frames(
                            str(self.pending / f'{name}.mp4'), self.video_size,
                            fps=1 / self.env.dt, codec='libx264', quality=8,
                            macro_block_size=2, output_params=['-movflags', '+faststart'])
                        self.writers[name] = writer
                        writer.send(None)
                for name, writer in self.writers.items():
                    writer.send(np.ascontiguousarray(frames[name]))
            except BaseException:
                self.recording = False
                self._close_videos()
                print(f'Video write failed; incomplete recording retained at {self.pending}', flush=True)
                raise
        self.rows.append({key: np.array(value, copy=True) for key, value in transition.items()})

    def save(self):
        if not self.recording:
            return None
        if not self.rows:
            self.recording = False
            print('Empty recording: no file written', flush=True)
            return None
        self._close_videos()
        self.pending.mkdir(parents=True, exist_ok=True)
        path = self.pending / 'trajectory.json'
        metadata = {
            'schema_version': 3,
            'instruction': self.episode_instruction,
            'state_names': ARM_JOINTS + ['gripper'],
            'action_names': ARM_JOINTS + ['gripper'],
            'units': ['rad'] * 6 + ['normalized_closure'],
            'gripper': '0=open, 1=closed; state=mean normalized driver position, action=ctrl/255',
            'action_type': 'absolute_joint_position_target',
            'alignment': 'state before action; next_state after one control interval',
            'time': 'MuJoCo simulation seconds, not wall time',
            'control_dt': self.env.dt,
            'physics_dt': float(self.env.model.opt.timestep),
            'eef_site': 'pinch',
            'teleop_frame': 'tcp_target',
            'eef_target_format': 'world x,y,z (m), quaternion w,x,y,z',
            'mujoco_version': mujoco.__version__,
            'num_steps': len(self.rows),
            'videos': ({name: f'{name}.mp4' for name in CAMERA_NAMES}
                       if self.video_size is not None else {}),
            'video_size': self.video_size,
            'video_fps': 1 / self.env.dt if self.video_size is not None else None,
            'video_alignment': 'frame[t] matches state[t] and time[t], before action[t]',
        }
        payload = {key: [row[key].tolist() for row in self.rows] for key in self.rows[0]}
        payload.update(metadata=metadata, initial_qpos=self.initial_qpos.tolist(),
                       initial_qvel=self.initial_qvel.tolist(), initial_ctrl=self.initial_ctrl.tolist())
        # Rename after the write completes so partial output is not treated as an episode.
        temporary = path.with_suffix('.json.tmp')
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
        temporary.replace(path)
        # Assign a number only to completed episodes; serialize concurrent saves.
        with (self.output / '.episode.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            numbers = [int(p.name) for p in self.output.iterdir() if p.name.isdigit()]
            self.destination = self.output / str(max(numbers, default=0) + 1)
            self.pending.rename(self.destination)
        path = self.destination / 'trajectory.json'
        self.recording = False
        self.rows = []
        print(f'SAVED {path} ({metadata["num_steps"]} steps)', flush=True)
        return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--instruction', default='', help='Task description saved with each episode')
    parser.add_argument('--video-width', type=int, default=640)
    parser.add_argument('--video-height', type=int, default=480)
    parser.add_argument('--control-hz', type=float, default=20)
    parser.add_argument('--move-step', type=float, default=0.01, help='Meters per key event')
    parser.add_argument('--rotate-step', type=float, default=5, help='Degrees per key event')
    args = parser.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (args.move_step, args.rotate_step)):
        parser.error('Movement and rotation increments must be positive and finite')
    env = RobotEnv(args.control_hz)
    recorder = EpisodeRecorder(args.output, env, args.instruction,
                               video_size=(args.video_width, args.video_height))
    keys = SimpleQueue()
    translation_keys = {'W': (0, 1), 'S': (0, -1), 'A': (1, -1),
                        'D': (1, 1), 'Q': (2, 1), 'E': (2, -1)}
    rotation_keys = {'I': (0, 1), 'K': (0, -1), 'J': (1, 1),
                     'L': (1, -1), 'U': (2, 1), 'O': (2, -1)}
    print(HELP, flush=True)
    print(f'Control frequency: {1/env.dt:g} Hz; recording OFF', flush=True)
    try:
        with TeleopWindow(env.model, key_callback=keys.put) as window:
            while window.poll():
                started = time.monotonic()
                # Process only events queued at the start of this control cycle.
                for _ in range(keys.qsize()):
                    try:
                        code = keys.get_nowait()
                    except Empty:
                        break
                    key = chr(code).upper() if 0 <= code < 128 else ''
                    if key == 'B':
                        recorder.save() if recorder.recording else recorder.start()
                    elif key == 'H':
                        recorder.change_instruction(edit_instruction(recorder.instruction))
                        # Discard stale input from before the modal editing dialog.
                        while not keys.empty():
                            keys.get_nowait()
                        break
                    elif key == 'R':
                        recorder.save()
                        env.randomize_objects()
                    elif key == 'N':
                        recorder.save()
                        env.reset()
                        print('RESET to home; recording OFF', flush=True)
                    elif key in ('G', 'T'):
                        env.set_gripper(float(np.clip(env.action[6] + (0.1 if key == 'G' else -0.1), 0, 1)))
                    elif key in translation_keys or key in rotation_keys:
                        move, rotate = np.zeros(3), np.zeros(3)
                        if key in translation_keys:
                            axis, direction = translation_keys[key]
                            move[axis] = direction * args.move_step
                        else:
                            axis, direction = rotation_keys[key]
                            rotate[axis] = direction * np.deg2rad(args.rotate_step)
                        if not env.nudge(move, rotate):
                            print('IK failed: previous target retained', flush=True)
                # All images observe state[t], before the command advances physics.
                frames = window.capture(env.data, recorder.video_size) if recorder.recording else None
                recorder.append(env.step(), frames)
                window.render(env.data, recorder.recording, len(recorder.rows), recorder.instruction)
                time.sleep(max(0, env.dt - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        recorder.save()


if __name__ == '__main__':
    main()
