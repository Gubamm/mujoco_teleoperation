"""Serialize OpenGL work on one thread and render recorded robot poses on demand."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import mujoco
import numpy as np
from PIL import Image
from robot_app.robot_env import RobotEnv


class ReplayRenderer:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mujoco-render')
        self.renderer = None

    def render(self, *args):
        return self.pool.submit(self._render, *args).result()

    def _render(self, episode, frame, mode, yaw, pitch, zoom):
        if self.renderer is None:
            self.env = RobotEnv()
            model = self.env.model
            base_id = model.body('base').id
            robot_bodies = {base_id}
            for body_id in range(1, model.nbody):
                if int(model.body_parentid[body_id]) in robot_bodies:
                    robot_bodies.add(body_id)
            self.robot_geom_ids = {i for i in range(model.ngeom)
                                   if int(model.geom_bodyid[i]) in robot_bodies}
            self.renderer = mujoco.Renderer(self.env.model, height=480, width=640)
        env = self.env
        mujoco.mj_resetData(env.model, env.data)
        full = episode.get('full_qpos')
        initial = episode.get('initial_qpos')
        if full is not None:
            qpos = np.asarray(full[frame], dtype=float)
            if qpos.shape != (env.model.nq,) or not np.isfinite(qpos).all():
                raise ValueError('full_qpos does not match the current model')
            env.data.qpos[:] = qpos
        elif initial is not None and len(initial) == env.model.nq:
            env.data.qpos[:] = initial
        values = episode[mode][frame]
        env.data.qpos[env.qpos_ids] = values[:6]
        if mode == 'action' or full is None:
            # Reconstruct the linkage's nominal closure when individual joints are absent.
            angle = float(np.clip(values[6], 0, 1)) * 0.8
            for side in ('left', 'right'):
                for link, sign in [('driver', 1), ('coupler', -1), ('spring_link', 1), ('follower', -1)]:
                    joint = env.model.joint(f'{side}_{link}_joint').id
                    env.data.qpos[env.model.jnt_qposadr[joint]] = sign * angle
        mujoco.mj_forward(env.model, env.data)
        base = env.model.body('base').id
        target = env.data.xpos[base] + np.array([0.25, 0, 0.25])
        front = env.data.cam_xpos[env.model.camera('front').id]
        delta = front - target
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = target
        camera.distance = np.linalg.norm(delta) / np.clip(zoom, 0.3, 5)
        camera.azimuth = np.degrees(np.arctan2(delta[1], delta[0])) + yaw
        camera.elevation = np.clip(-np.degrees(np.arctan2(delta[2], np.linalg.norm(delta[:2]))) + pitch, -89, 89)
        options = mujoco.MjvOption()
        # Replay shows only the robot and its trajectories; collection assets stay intact.
        self.renderer.update_scene(env.data, camera=camera, scene_option=options)
        scene = self.renderer.scene
        for geom in scene.geoms[:scene.ngeom]:
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid >= 0:
                if geom.objid not in self.robot_geom_ids:
                    geom.rgba[3] = 0
        rotation = env.data.xmat[base].reshape(3, 3)
        offset = env.data.xpos[base]
        for kind, color in [('state', [0.31, 0.85, 0.77, 1]), ('action', [1, 0.7, 0.41, 1])]:
            points = np.asarray(episode['trajectories'][kind]) @ rotation.T + offset
            indices = np.unique(np.r_[np.arange(0, len(points), max(1, len(points)//400)), len(points)-1])
            for a, b in zip(indices[:-1], indices[1:]):
                if scene.ngeom >= scene.maxgeom or np.linalg.norm(points[b]-points[a]) < 1e-7:
                    continue
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.002, points[a], points[b])
                scene.ngeom += 1
        image = self.renderer.render()
        output = BytesIO()
        Image.fromarray(image).save(output, format='JPEG', quality=85)
        return output.getvalue()

    def close(self):
        if self.renderer is not None:
            self.pool.submit(self.renderer.close).result()
        self.pool.shutdown(wait=True)
