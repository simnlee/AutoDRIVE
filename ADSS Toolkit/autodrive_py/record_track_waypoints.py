#!/usr/bin/env python

"""
Script to record vehicle trajectory as a polyline CSV
Saves waypoints with: x, y, z, s, yaw
where s is cumulative arc length and yaw is vehicle heading
"""

# Import libraries
import socketio
import eventlet
from flask import Flask
import autodrive
import numpy as np
import csv
import time
from datetime import datetime

################################################################################

class TrajectoryRecorder:
    def __init__(self, vehicle_id='V1', output_file=None):
        self.vehicle_id = vehicle_id

        # Generate filename with timestamp if not provided
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_file = f'track_waypoints_{timestamp}.csv'
        else:
            self.output_file = output_file

        # Initialize vehicle
        self.vehicle = autodrive.F1TENTH()
        self.vehicle.id = vehicle_id

        # Recording state
        self.waypoints = []
        self.cumulative_distance = 0.0
        self.previous_position = None
        self.recording = False
        self.sample_count = 0

        # Sampling parameters
        self.min_distance_threshold = 0.05  # Minimum distance between waypoints (meters)

        # Initialize Socket.IO server
        self.sio = socketio.Server()
        self.app = Flask(__name__)

        # Setup event handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup Socket.IO event handlers"""

        @self.sio.on('connect')
        def connect(sid, environ):
            print('='*80)
            print('AutoDRIVE Simulator Connected!')
            print('='*80)
            print('\nControls:')
            print('  - Drive the vehicle manually in the simulator')
            print('  - Recording will start automatically')
            print('  - Press Ctrl+C to stop and save\n')
            self.recording = True

        @self.sio.on('disconnect')
        def disconnect(sid):
            print('\nSimulator disconnected!')
            self._save_waypoints()

        @self.sio.on('Bridge')
        def bridge(sid, data):
            if data and self.recording:
                # Parse vehicle data
                self.vehicle.parse_data(data, verbose=False)

                # Extract position and yaw
                x, y, z = self.vehicle.position
                yaw = self.vehicle.orientation_euler_angles[2]  # Z-axis rotation

                # Check if we should record this waypoint
                current_position = np.array([x, y, z])

                if self.previous_position is None:
                    # First waypoint
                    self.waypoints.append({
                        'x': x,
                        'y': y,
                        'z': z,
                        's': 0.0,
                        'yaw': yaw
                    })
                    self.previous_position = current_position
                    self.sample_count += 1
                    print(f'Recording started - Waypoint 1: x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw={yaw:.3f}')
                else:
                    # Compute distance from last waypoint
                    distance = np.linalg.norm(current_position - self.previous_position)

                    # Only record if moved enough
                    if distance >= self.min_distance_threshold:
                        self.cumulative_distance += distance
                        self.waypoints.append({
                            'x': x,
                            'y': y,
                            'z': z,
                            's': self.cumulative_distance,
                            'yaw': yaw
                        })
                        self.previous_position = current_position
                        self.sample_count += 1

                        # Print progress every 10 waypoints
                        if self.sample_count % 10 == 0:
                            print(f'Waypoint {self.sample_count}: s={self.cumulative_distance:.3f}m, yaw={yaw:.3f}rad')

                # Send neutral commands back to simulator
                self.vehicle.throttle_command = 0.0
                self.vehicle.steering_command = 0.0
                json_msg = self.vehicle.generate_commands(verbose=False)

                try:
                    self.sio.emit('Bridge', data=json_msg)
                except Exception as e:
                    print(f'Error sending commands: {e}')

    def _save_waypoints(self):
        """Save recorded waypoints to CSV file"""
        if len(self.waypoints) == 0:
            print('\nNo waypoints recorded!')
            return

        print(f'\nSaving {len(self.waypoints)} waypoints to {self.output_file}...')

        with open(self.output_file, 'w', newline='') as csvfile:
            fieldnames = ['x', 'y', 'z', 's', 'yaw']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for waypoint in self.waypoints:
                writer.writerow(waypoint)

        print(f'✓ Saved successfully!')
        print(f'\nTrajectory Statistics:')
        print(f'  Total waypoints: {len(self.waypoints)}')
        print(f'  Total distance:  {self.cumulative_distance:.3f} m')
        print(f'  Output file:     {self.output_file}')

    def run(self, host='', port=4567):
        """Start the recording server"""
        print('='*80)
        print('AutoDRIVE Track Waypoint Recorder')
        print('='*80)
        print(f'\nStarting server on {host if host else "localhost"}:{port}...')
        print('Waiting for simulator connection...\n')

        app = socketio.Middleware(self.sio, self.app)

        try:
            eventlet.wsgi.server(eventlet.listen((host, port)), app)
        except KeyboardInterrupt:
            print('\n\nRecording stopped by user.')
            self._save_waypoints()
        except Exception as e:
            print(f'\nError: {e}')
            self._save_waypoints()

################################################################################

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Record AutoDRIVE track waypoints')
    parser.add_argument('--vehicle-id', type=str, default='V1',
                        help='Vehicle ID (default: V1)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV filename (default: track_waypoints_TIMESTAMP.csv)')
    parser.add_argument('--port', type=int, default=4567,
                        help='Server port (default: 4567)')
    parser.add_argument('--min-distance', type=float, default=0.05,
                        help='Minimum distance between waypoints in meters (default: 0.05)')

    args = parser.parse_args()

    recorder = TrajectoryRecorder(
        vehicle_id=args.vehicle_id,
        output_file=args.output
    )
    recorder.min_distance_threshold = args.min_distance

    recorder.run(port=args.port)
