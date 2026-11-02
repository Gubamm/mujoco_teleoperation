"""MuJoCo joint control, measured state, and Cartesian inverse kinematics."""
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parent.parent / 'assets/mujoco_xml/scene.xml'
ARM_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
ARM_ACTUATORS = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']


def rotation_vector(matrix):
    """Shortest world-frame rotation vector for a relative rotation matrix."""
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix).ravel())
    if quat[0] < 0:
        quat *= -1
    length = np.linalg.norm(quat[1:])
    if length < 1e-10:
        return 2 * quat[1:]
    return quat[1:] * (2 * np.arctan2(length, quat[0]) / length)


class RobotEnv:
    def __init__(self, control_hz=20.0):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError('control_hz must be positive and finite')
        self.substeps = round(1 / (control_hz * self.model.opt.timestep))
        if self.substeps < 1:
            raise ValueError('control_hz exceeds the physics frequency')
        self.dt = self.substeps * self.model.opt.timestep
        joints = [self.model.joint(name).id for name in ARM_JOINTS]
        self.qpos_ids = self.model.jnt_qposadr[joints]
        self.dof_ids = self.model.jnt_dofadr[joints]
        self.actuator_ids = [self.model.actuator(name).id for name in ARM_ACTUATORS]
        self.gripper_id = self.model.actuator('fingers_actuator').id
        self.lower = np.maximum(self.model.jnt_range[joints, 0],
                                self.model.actuator_ctrlrange[self.actuator_ids, 0])
        self.upper = np.minimum(self.model.jnt_range[joints, 1],
                                self.model.actuator_ctrlrange[self.actuator_ids, 1])
        drivers = [self.model.joint(name).id for name in
                   ('right_driver_joint', 'left_driver_joint')]
        self.driver_qpos = self.model.jnt_qposadr[drivers]
        self.driver_lower = self.model.jnt_range[drivers, 0]
        self.driver_span = self.model.jnt_range[drivers, 1] - self.driver_lower
        self.site_id = self.model.site('pinch').id
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('home').id)
        mujoco.mj_forward(self.model, self.data)
        self.action = np.r_[self.data.ctrl[self.actuator_ids],
                            self.data.ctrl[self.gripper_id] / 255.0]
        self.target_position, self.target_quaternion = self.eef_pose()
        return self.state()

    def state(self):
        closure = np.mean((self.data.qpos[self.driver_qpos] - self.driver_lower)
                          / self.driver_span)
        return np.r_[self.data.qpos[self.qpos_ids], np.clip(closure, 0, 1)]

    def eef_pose(self):
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id])
        return self.data.site_xpos[self.site_id].copy(), quat

    def set_eef_target(self, position, quaternion):
        """Solve an absolute world-frame pinch pose; commit only on convergence.

        Quaternion order is w,x,y,z. Failure leaves the previous action/target intact.
        The solver uses a separate MjData and never moves the live simulation.
        """
        position = np.asarray(position, dtype=float)
        quaternion = np.asarray(quaternion, dtype=float)
        if (position.shape != (3,) or quaternion.shape != (4,)
                or not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion))
                or np.linalg.norm(quaternion) < 1e-10):
            raise ValueError('Expected finite position (3) and nonzero quaternion (4)')
        quaternion = quaternion / np.linalg.norm(quaternion)
        target_matrix = np.empty(9)
        mujoco.mju_quat2Mat(target_matrix, quaternion)
        target_matrix = target_matrix.reshape(3, 3)
        work = self.ik_data
        work.qpos[:] = self.data.qpos
        # Warm-start from the previous command to keep the IK branch continuous.
        work.qpos[self.qpos_ids] = self.action[:6]
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        for _ in range(150):
            mujoco.mj_forward(self.model, work)
            pos_error = position - work.site_xpos[self.site_id]
            rot_error = rotation_vector(target_matrix @ work.site_xmat[self.site_id].reshape(3, 3).T)
            if np.linalg.norm(pos_error) < 0.0005 and np.linalg.norm(rot_error) < 0.005:
                self.action[:6] = work.qpos[self.qpos_ids]
                self.target_position = position.copy()
                self.target_quaternion = quaternion.copy()
                return True
            mujoco.mj_jacSite(self.model, work, jac_pos, jac_rot, self.site_id)
            # Orientation is weighted in meters/radian for the least-squares solve.
            jac = np.vstack((jac_pos[:, self.dof_ids], 0.2 * jac_rot[:, self.dof_ids]))
            error = np.r_[pos_error, 0.2 * rot_error]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.002**2 * np.eye(6), error)
            delta *= min(1.0, 0.15 / max(np.max(np.abs(delta)), 1e-12))
            work.qpos[self.qpos_ids] = np.clip(work.qpos[self.qpos_ids] + delta,
                                              self.lower, self.upper)
        return False

    def nudge(self, translation, rotation):
        """Increment the last accepted target in its TCP (pinch) frame.

        Translation is in meters; rotation is a local axis-angle vector in radians.
        IK still receives an absolute world-frame pose after this conversion.
        """
        translation = np.asarray(translation, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if (translation.shape != (3,) or rotation.shape != (3,)
                or not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation))):
            raise ValueError('Nudge requires finite 3D translation and rotation')
        angle = np.linalg.norm(rotation)
        delta = np.array([1., 0., 0., 0.])
        if angle > 1e-10:
            delta = np.r_[np.cos(angle / 2), rotation / angle * np.sin(angle / 2)]
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, self.target_quaternion, delta)
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, self.target_quaternion)
        world_translation = matrix.reshape(3, 3) @ translation
        return self.set_eef_target(self.target_position + world_translation, quat)

    def set_gripper(self, closure):
        if not np.isfinite(closure) or not 0 <= closure <= 1:
            raise ValueError('Gripper closure must be in [0, 1]')
        self.action[6] = closure

    def step(self):
        """Capture observation, apply absolute joint action, then advance one control tick."""
        state = self.state()
        full_qpos = self.data.qpos.copy()
        action = self.action.copy()
        timestamp = float(self.data.time)
        self.data.ctrl[self.actuator_ids] = action[:6]
        self.data.ctrl[self.gripper_id] = action[6] * 255.0
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        mujoco.mj_forward(self.model, self.data)
        next_state = self.state()
        if not np.all(np.isfinite(np.r_[state, action, next_state])):
            raise RuntimeError('Non-finite robot state; stop and reset the simulation')
        return {'state': state, 'action': action, 'next_state': next_state,
                'full_qpos': full_qpos,
                'time': timestamp, 'next_time': float(self.data.time),
                'eef_target': np.r_[self.target_position, self.target_quaternion]}
