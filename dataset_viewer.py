"""Local dataset browser: uv run dataset_viewer.py [--data data] [--port 8000]."""
import argparse
import os
os.environ.setdefault("MUJOCO_GL", "egl")
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import threading
from urllib.parse import unquote, urlparse, parse_qs

import mujoco
import numpy as np
from robot_app.robot_env import RobotEnv

WEB = Path(__file__).resolve().parent / 'dataset_web'


class Dataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.env = RobotEnv()
        self.lock = threading.Lock()
        self.cache = {}
        from robot_app.mujoco_replay import ReplayRenderer
        self.replay = ReplayRenderer()

    def episode_path(self, name):
        if not name or name.startswith('.') or Path(name).name != name:
            raise ValueError('Invalid episode name')
        path = (self.root / name).resolve()
        if path.parent != self.root or not path.is_dir():
            raise FileNotFoundError('Episode not found')
        return path

    def listing(self):
        result = []
        if not self.root.exists():
            return result
        for path in self.root.iterdir():
            if path.name.startswith('.') or not path.is_dir() or path.is_symlink():
                continue
            file = path / 'trajectory.json'
            if not file.is_file():
                continue
            try:
                data = json.loads(file.read_text(encoding='utf-8'))
                meta = data.get('metadata', {})
                count = len(data['state'])
                dt = float(meta.get('control_dt', 0.05))
                result.append({'id': path.name, 'instruction': meta.get('instruction', ''),
                               'samples': count, 'duration': count * dt})
            except (ValueError, KeyError, TypeError, OSError):
                result.append({'id': path.name, 'error': 'trajectory.json을 읽을 수 없습니다.'})
        return sorted(result, key=lambda row: (0, int(row['id'])) if row['id'].isdigit()
                      else (1, row['id']))

    def detail(self, name):
        path = self.episode_path(name)
        file = path / 'trajectory.json'
        stamp = (file.stat().st_mtime_ns, file.stat().st_size)
        with self.lock:
            if name in self.cache and self.cache[name][0] == stamp:
                return self.cache[name][1]
            source = json.loads(file.read_text(encoding='utf-8'))
            state = np.asarray(source['state'], dtype=float)
            action = np.asarray(source['action'], dtype=float)
            if state.ndim != 2 or state.shape[1] != 7 or state.shape != action.shape or len(state) == 0:
                raise ValueError('state/action must have matching nonempty (T, 7) shapes')
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError('Non-finite state/action values')
            meta = source.get('metadata', {})
            expected = self.env.model
            # Forward kinematics from recorded joints, never from the desired EEF target.
            work = mujoco.MjData(expected)
            work.qpos[:] = expected.qpos0
            initial = source.get('initial_qpos')
            if initial is not None and len(initial) == expected.nq:
                work.qpos[:] = initial
            base_id = expected.body('base').id
            trajectories = {}
            for kind, values in (('state', state), ('action', action)):
                points = []
                for row in values:
                    work.qpos[self.env.qpos_ids] = row[:6]
                    # Pinch is fixed to the gripper base, independent of finger closure.
                    mujoco.mj_kinematics(expected, work)
                    base_rotation = work.xmat[base_id].reshape(3, 3)
                    points.append(base_rotation.T @ (work.site_xpos[self.env.site_id]
                                                       - work.xpos[base_id]))
                trajectories[kind] = np.asarray(points).tolist()
            videos = {}
            for camera in ('front', 'top_camera', 'gripper'):
                filename = meta.get('videos', {}).get(camera, f'{camera}.mp4')
                if isinstance(filename, str) and Path(filename).name == filename and (path / filename).is_file():
                    videos[camera] = filename
            dt = float(meta.get('control_dt', 0.05))
            times = np.asarray(source.get('time', np.arange(len(state)) * dt), dtype=float)
            if times.shape != (len(state),) or not np.isfinite(times).all():
                raise ValueError('Invalid timestamps')
            result = {'id': name, 'metadata': meta, 'state': state.tolist(), 'action': action.tolist(),
                      'time': (times - times[0]).tolist(), 'dt': dt, 'videos': videos,
                      'trajectories': trajectories,
                      'full_qpos': source.get('full_qpos'), 'initial_qpos': source.get('initial_qpos'),
                      'full_replay': source.get('full_qpos') is not None, 'frame': 'base', 'eef_site': 'pinch'}
            # Bound memory when browsing many episodes.
            if len(self.cache) >= 4:
                self.cache.pop(next(iter(self.cache)))
            self.cache[name] = (stamp, result)
            return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass 
    
    def __init__(self, *args, dataset, **kwargs):
        self.dataset = dataset
        super().__init__(*args, **kwargs)

    def do_GET(self):
        try:
            path = unquote(urlparse(self.path).path)
            if path == '/api/episodes':
                return self.send_json(self.dataset.listing())
            if path == '/api/render':
                query = parse_qs(urlparse(self.path).query)
                name = query.get('episode', [''])[0]
                frame = int(query.get('frame', ['0'])[0])
                mode = query.get('mode', ['state'])[0]
                angles = [float(query.get(k, [v])[0]) for k,v in
                          [('yaw','0'), ('pitch','0'), ('zoom','1')]]
                if mode not in ('state', 'action') or not np.isfinite(angles).all():
                    raise ValueError('Invalid replay parameters')
                episode = self.dataset.detail(name)
                if not 0 <= frame < len(episode['state']):
                    raise ValueError('Invalid frame')
                raw = self.dataset.replay.render(episode, frame, mode, *angles)
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(raw)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(raw)
                return
            if path.startswith('/api/episodes/'):
                return self.send_json({k:v for k,v in self.dataset.detail(path[len('/api/episodes/'):]).items() if k not in ('full_qpos', 'initial_qpos')})
            if path.startswith('/media/'):
                parts = path.split('/')
                if len(parts) != 4:
                    raise ValueError('Invalid media path')
                folder = self.dataset.episode_path(parts[2])
                file = (folder / parts[3]).resolve()
                if file.parent != folder or file.suffix != '.mp4':
                    raise ValueError('Invalid media file')
                return self.send_file(file)
            files = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}
            if path not in files:
                raise FileNotFoundError('Not found')
            self.send_file(WEB / files[path])
        except (FileNotFoundError, KeyError):
            self.send_json({'error': '파일이나 에피소드를 찾을 수 없습니다.'}, 404)
        except (ValueError, TypeError) as exc:
            self.send_json({'error': str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.send_json({'error': f'MuJoCo 재생 오류: {exc}'}, 500)

    def send_json(self, value, status=200):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(raw)

    def send_file(self, path):
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get('Range')
        if range_header:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', range_header)
            if not match or not any(match.groups()):
                return self.invalid_range(size)
            lo, hi = match.groups()
            if lo:
                start = int(lo)
                end = min(int(hi), size - 1) if hi else size - 1
            else:
                start = max(0, size - int(hi))
            if start > end or start >= size:
                return self.invalid_range(size)
        self.send_response(206 if range_header else 200)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(max(0, end - start + 1)))
        self.send_header('Accept-Ranges', 'bytes')
        if range_header:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                block = stream.read(min(65536, remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining -= len(block)

    def invalid_range(self, size):
        self.send_response(416)
        self.send_header('Content-Range', f'bytes */{size}')
        self.send_header('Content-Length', '0')
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, dataset=Dataset(args.data)))
    print(f'Dataset viewer: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.RequestHandlerClass.keywords['dataset'].replay.close()


if __name__ == '__main__':
    main()
