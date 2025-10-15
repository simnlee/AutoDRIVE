"""
autodrive_gym.py
----------------
Gymnasium environment wrapper for AutoDRIVE F1TENTH simulator.

Usage:
    env = AutoDRIVEGymEnv(waypoint_csv_path="track_waypoints_trimmed.csv")
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            obs, info = env.reset()
    env.close()
"""

import math
import time
import threading
from queue import Queue, Empty
from typing import Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from scipy.spatial import KDTree

import socketio
import eventlet
from flask import Flask

from autodrive import F1TENTH


# ===========================
# Utility Functions
# ===========================

def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def resample_lidar_to_1080(lidar_array: np.ndarray, max_range: float = 20.0) -> np.ndarray:
    """Resample LiDAR array to exactly 1080 points."""
    arr = np.asarray(lidar_array, dtype=np.float32)
    if arr.size == 1080:
        return np.clip(arr, 0.0, max_range)
    elif arr.size == 1081:
        return np.clip(arr[:-1], 0.0, max_range)
    else:
        # Linear interpolation
        x_old = np.linspace(0, 1, arr.size)
        x_new = np.linspace(0, 1, 1080)
        resampled = np.interp(x_new, x_old, arr)
        return np.clip(resampled, 0.0, max_range).astype(np.float32)


def load_waypoints(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load waypoints from CSV. Returns (x, y, s, yaw) arrays."""
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    x = np.asarray(data['x'], dtype=np.float64)
    y = np.asarray(data['y'], dtype=np.float64)
    s = np.asarray(data['s'], dtype=np.float64)
    yaw = np.asarray(data['yaw'], dtype=np.float64)

    # Sort by s
    idx = np.argsort(s)
    return x[idx], y[idx], s[idx], yaw[idx]


def project_to_track(px: float, py: float,
                     wp_x: np.ndarray, wp_y: np.ndarray, wp_s: np.ndarray,
                     kdtree: KDTree, s_total: float) -> Tuple[float, float]:
    """Project point (px, py) onto track. Returns (s_proj, yaw_ref)."""
    # Find nearest waypoint
    _, idx = kdtree.query([px, py])

    # Check adjacent segments
    n = len(wp_x)
    candidates = []

    for i in [idx - 1, idx]:
        i0 = i % n
        i1 = (i + 1) % n

        # Project onto segment
        x0, y0 = wp_x[i0], wp_y[i0]
        x1, y1 = wp_x[i1], wp_y[i1]

        dx, dy = x1 - x0, y1 - y0
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq < 1e-12:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / seg_len_sq))

        xp = x0 + t * dx
        yp = y0 + t * dy
        dist = math.hypot(px - xp, py - yp)

        # Compute s along track
        ds = wp_s[i1] - wp_s[i0]
        if ds < -0.5 * s_total:  # Handle wraparound
            ds += s_total
        s_proj = wp_s[i0] + t * ds
        if s_proj >= s_total:
            s_proj -= s_total

        # Compute reference yaw (tangent direction)
        yaw_ref = math.atan2(dy, dx)

        candidates.append((dist, s_proj, yaw_ref))

    # Return projection with smallest distance
    _, s_proj, yaw_ref = min(candidates, key=lambda x: x[0])
    return float(s_proj), float(yaw_ref)


def get_lidar_sector_min(lidar_1080: np.ndarray, center_angle: float,
                         half_width: float = math.radians(15),
                         fov: float = 4.7) -> float:
    """Get minimum range in LiDAR sector centered at angle."""
    # LiDAR FOV spans [-fov/2, +fov/2] with 1080 beams
    angle_per_beam = fov / 1079
    start_angle = -fov / 2

    # Convert center angle to beam indices
    angle_start = center_angle - half_width
    angle_end = center_angle + half_width

    idx_start = int((angle_start - start_angle) / angle_per_beam)
    idx_end = int((angle_end - start_angle) / angle_per_beam)

    idx_start = max(0, min(1079, idx_start))
    idx_end = max(0, min(1079, idx_end))

    if idx_start > idx_end:
        idx_start, idx_end = idx_end, idx_start

    sector = lidar_1080[idx_start:idx_end+1]
    sector = sector[np.isfinite(sector)]

    if len(sector) == 0:
        return float('inf')
    return float(np.min(sector))


# ===========================
# Gymnasium Environment
# ===========================

class AutoDRIVEGymEnv(gym.Env):
    """Gymnasium environment for AutoDRIVE F1TENTH racing."""

    metadata = {"render_modes": []}

    def __init__(self,
                 waypoint_csv_path: str,
                 host: str = '127.0.0.1',
                 port: int = 4567,
                 vehicle_id: str = 'V1',
                 config: Optional[Dict] = None):
        """
        Initialize AutoDRIVE Gym environment.

        Args:
            waypoint_csv_path: Path to track waypoints CSV
            host: SocketIO server host
            port: SocketIO server port
            vehicle_id: Vehicle ID (default 'V1')
            config: Optional config dict
        """
        super().__init__()

        # Configuration
        self.cfg = {
            'lidar_max': 20.0,
            'lidar_fov': 4.7,
            'collision_threshold': 0.15,  # meters
            'throttle_lut': {0: 0.0, 1: 0.35, 2: 0.70},
            'steering_lut': {0: -0.35, 1: 0.0, 2: 0.35},
        }
        if config:
            self.cfg.update(config)

        # Load track waypoints
        self.wp_x, self.wp_y, self.wp_s, self.wp_yaw = load_waypoints(waypoint_csv_path)
        self.s_total = float(self.wp_s[-1])
        self.kdtree = KDTree(np.column_stack([self.wp_x, self.wp_y]))

        # Vehicle
        self.vehicle = F1TENTH()
        self.vehicle.id = vehicle_id
        # Initialize commands to valid values (not None)
        self.vehicle.throttle_command = 0.0
        self.vehicle.steering_command = 0.0

        # Action space: [drive_action, steer_action]
        # drive_action: 0=coast, 1=medium, 2=full
        # steer_action: 0=left, 1=straight, 2=right
        self.action_space = gym.spaces.MultiDiscrete([3, 3])

        # Observation space
        self.observation_space = gym.spaces.Dict({
            'lidar_ranges': gym.spaces.Box(low=0.0, high=self.cfg['lidar_max'],
                                          shape=(1080,), dtype=np.float32),
            'yaw_angle': gym.spaces.Box(low=-math.pi, high=math.pi,
                                       shape=(1,), dtype=np.float32),
            'yaw_angle_ref': gym.spaces.Box(low=-math.pi, high=math.pi,
                                           shape=(1,), dtype=np.float32),
            'yaw_angle_error': gym.spaces.Box(low=-math.pi, high=math.pi,
                                             shape=(1,), dtype=np.float32),
            'lin_vel_xy': gym.spaces.Box(low=-np.inf, high=np.inf,
                                        shape=(2,), dtype=np.float32),
            'ang_vel_z': gym.spaces.Box(low=-np.inf, high=np.inf,
                                       shape=(1,), dtype=np.float32),
            'track_edge_dist_lr': gym.spaces.Box(low=0.0, high=self.cfg['lidar_max'],
                                                shape=(2,), dtype=np.float32),
            'lap_time': gym.spaces.Box(low=0.0, high=np.inf,
                                      shape=(1,), dtype=np.float32),
            'lap_count': gym.spaces.Box(low=0.0, high=np.inf,
                                       shape=(1,), dtype=np.float32),
        })

        # State tracking
        self.prev_pos = None
        self.prev_s = None
        self.prev_time = None
        self.lap_time = 0.0
        self.lap_count = 0

        # Communication queue
        self.data_queue = Queue(maxsize=1)
        self.action_queue = Queue(maxsize=1)

        # SocketIO server
        self.sio = socketio.Server(async_mode='eventlet')
        self.app = Flask(__name__)
        self.server_thread = None
        self.running = True  # Set to True BEFORE starting server

        # Setup SocketIO handlers
        self._setup_socketio()

        # Start server
        self._start_server(host, port)

    def _setup_socketio(self):
        """Setup SocketIO event handlers."""

        @self.sio.on('connect')
        def connect(sid, environ):
            print(f'AutoDRIVE connected: {sid}')

        @self.sio.on('Bridge')
        def bridge(sid, data):
            if not data or not self.running:
                return

            # Parse incoming sensor data
            self.vehicle.parse_data(data, verbose=False)

            # Put data in queue for step() to consume
            if self.data_queue.full():
                try:
                    self.data_queue.get_nowait()
                except Empty:
                    pass
            self.data_queue.put(data)

            # Get action from queue
            try:
                action = self.action_queue.get_nowait()
                throttle = action['throttle']
                steering = action['steering']
            except Empty:
                # Default: no action
                throttle = 0.0
                steering = 0.0

            # Send commands
            self.vehicle.throttle_command = throttle
            self.vehicle.steering_command = steering

            json_msg = self.vehicle.generate_commands(verbose=False)
            print(f'DEBUG: json_msg = {json_msg}')
            try:
                self.sio.emit('Bridge', data=json_msg)
            except Exception as e:
                print(f'Error emitting: {e}')

    def _start_server(self, host: str, port: int):
        """Start SocketIO server in background thread."""
        def run_server():
            app = socketio.Middleware(self.sio, self.app)
            eventlet.wsgi.server(eventlet.listen((host, port)), app)

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        print(f'AutoDRIVE Gym server started on {host}:{port}')
        time.sleep(1.0)  # Give server time to start

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment. Returns initial observation and info."""
        super().reset(seed=seed)

        # Reset tracking variables
        self.prev_pos = None
        self.prev_s = None
        self.prev_time = time.time()
        self.lap_time = 0.0
        # Note: lap_count is NOT reset - it accumulates across episodes

        # Wait for first data from simulator
        obs = self._get_observation()

        return obs, {}

    def step(self, action):
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        # Map discrete action to continuous commands
        drive_action = int(action[0])
        steer_action = int(action[1])

        throttle = self.cfg['throttle_lut'][drive_action]
        steering = self.cfg['steering_lut'][steer_action]

        # Send action to SocketIO handler
        if self.action_queue.full():
            try:
                self.action_queue.get_nowait()
            except Empty:
                pass
        self.action_queue.put({'throttle': throttle, 'steering': steering})

        # Wait for new observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward(obs)

        # Check termination (only lap completion)
        terminated = self._check_lap_complete(obs)
        truncated = False

        info = {
            'lap_time': self.lap_time,
            'lap_count': self.lap_count,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get observation from simulator data."""
        # Wait for data with timeout
        try:
            data = self.data_queue.get(timeout=5.0)
        except Empty:
            raise RuntimeError("Timeout waiting for simulator data")

        # Update time
        current_time = time.time()
        if self.prev_time is None:
            dt = 0.05  # Default
        else:
            dt = current_time - self.prev_time
        self.prev_time = current_time
        self.lap_time += dt

        # Extract sensor data
        position = self.vehicle.position  # [x, y, z]
        euler = self.vehicle.orientation_euler_angles  # [roll, pitch, yaw]
        ang_vel = self.vehicle.angular_velocity  # [wx, wy, wz]
        lidar_raw = self.vehicle.lidar_range_array

        # Process LiDAR
        lidar_1080 = resample_lidar_to_1080(lidar_raw, self.cfg['lidar_max'])

        # Track projection
        s_proj, yaw_ref = project_to_track(
            position[0], position[1],
            self.wp_x, self.wp_y, self.wp_s,
            self.kdtree, self.s_total
        )

        # Yaw angle and error (convert from [0, 2π] to [-π, π])
        yaw = float(euler[2]) - math.pi
        yaw_err = wrap_to_pi(yaw - yaw_ref)

        # Linear velocity (finite difference)
        if self.prev_pos is None:
            vx, vy = 0.0, 0.0
        else:
            dx = position[0] - self.prev_pos[0]
            dy = position[1] - self.prev_pos[1]

            # Transform to vehicle frame
            c, s = math.cos(yaw), math.sin(yaw)
            vx = (c * dx + s * dy) / max(dt, 1e-6)
            vy = (-s * dx + c * dy) / max(dt, 1e-6)

        self.prev_pos = (position[0], position[1])
        self.prev_s = s_proj

        # Track edge distances (left/right from LiDAR)
        dist_left = get_lidar_sector_min(lidar_1080, math.pi/2,
                                         math.radians(15), self.cfg['lidar_fov'])
        dist_right = get_lidar_sector_min(lidar_1080, -math.pi/2,
                                          math.radians(15), self.cfg['lidar_fov'])

        if not np.isfinite(dist_left):
            dist_left = self.cfg['lidar_max']
        if not np.isfinite(dist_right):
            dist_right = self.cfg['lidar_max']

        # Build observation dict
        obs = {
            'lidar_ranges': lidar_1080,
            'yaw_angle': np.array([yaw], dtype=np.float32),
            'yaw_angle_ref': np.array([yaw_ref], dtype=np.float32),
            'yaw_angle_error': np.array([yaw_err], dtype=np.float32),
            'lin_vel_xy': np.array([vx, vy], dtype=np.float32),
            'ang_vel_z': np.array([ang_vel[2]], dtype=np.float32),
            'track_edge_dist_lr': np.array([dist_left, dist_right], dtype=np.float32),
            'lap_time': np.array([self.lap_time], dtype=np.float32),
            'lap_count': np.array([float(self.lap_count)], dtype=np.float32),
        }

        return obs

    def _compute_reward(self, obs: Dict[str, np.ndarray]) -> float:
        """Compute reward for current step."""
        reward = 0.0

        # Progress reward (main component)
        if self.prev_s is not None:
            current_s = self._get_s_from_obs()
            ds = current_s - self.prev_s

            # Handle wraparound
            if ds < -0.5 * self.s_total:
                ds += self.s_total
            elif ds > 0.5 * self.s_total:
                ds -= self.s_total

            reward += ds * 10.0  # Scale progress reward

        # Speed reward (encourage high forward velocity)
        vx = float(obs['lin_vel_xy'][0])
        reward += vx * 0.1

        # Heading error penalty (quadratic)
        yaw_err = float(obs['yaw_angle_error'][0])
        reward -= (yaw_err ** 2) * 2.0

        # Wall proximity penalty (exponential)
        dist_left = float(obs['track_edge_dist_lr'][0])
        dist_right = float(obs['track_edge_dist_lr'][1])
        min_wall_dist = min(dist_left, dist_right)
        reward -= math.exp(-min_wall_dist / 2.0) * 1.0

        # Collision penalty
        if self._check_collision(obs):
            reward -= 50.0

        return reward

    def _get_s_from_obs(self) -> float:
        """Get current arc-length from stored position."""
        if self.vehicle.position is None:
            return 0.0
        s_proj, _ = project_to_track(
            self.vehicle.position[0], self.vehicle.position[1],
            self.wp_x, self.wp_y, self.wp_s,
            self.kdtree, self.s_total
        )
        return s_proj

    def _check_collision(self, obs: Dict[str, np.ndarray]) -> bool:
        """Check if vehicle is in collision."""
        lidar = obs['lidar_ranges']
        min_range = np.min(lidar[np.isfinite(lidar)])
        return min_range < self.cfg['collision_threshold']

    def _check_lap_complete(self, obs: Dict[str, np.ndarray]) -> bool:
        """Check if lap is complete (s wrapped around)."""
        if self.prev_s is None:
            return False

        current_s = self._get_s_from_obs()

        # Detect wraparound: previous s > 90% of track, current s < 10% of track
        if self.prev_s > 0.9 * self.s_total and current_s < 0.1 * self.s_total:
            self.lap_count += 1
            return True

        return False

    def close(self):
        """Clean up resources."""
        self.running = False
        if self.server_thread:
            print("Closing AutoDRIVE Gym environment...")
