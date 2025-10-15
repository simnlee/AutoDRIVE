#!/usr/bin/env python

"""
Diagnostic script to inspect all data sent from AutoDRIVE simulator
Run this script and start the simulator to see what data is available
"""

# Import libraries
import socketio
import eventlet
from flask import Flask
import json

################################################################################

# Initialize the server
sio = socketio.Server()

# Flask (web) app
app = Flask(__name__)

# Flag to print only once
data_printed = False

# Registering "connect" event handler for the server
@sio.on('connect')
def connect(sid, environ):
    print('='*80)
    print('AutoDRIVE Simulator Connected!')
    print('SID:', sid)
    print('='*80)
    print('\nWaiting for data from simulator...\n')

# Registering "disconnect" event handler for the server
@sio.on('disconnect')
def disconnect(sid):
    print('\n' + '='*80)
    print('AutoDRIVE Simulator Disconnected!')
    print('='*80)

# Registering "Bridge" event handler for the server
@sio.on('Bridge')
def bridge(sid, data):
    global data_printed

    if data and not data_printed:
        print('='*80)
        print('RECEIVED DATA FROM SIMULATOR')
        print('='*80)

        # Print all available keys
        print('\n' + '-'*80)
        print('AVAILABLE DATA KEYS:')
        print('-'*80)
        sorted_keys = sorted(data.keys())
        for i, key in enumerate(sorted_keys, 1):
            print(f'{i:3d}. {key}')

        # Print sample values for each key (excluding large data like images)
        print('\n' + '-'*80)
        print('SAMPLE VALUES:')
        print('-'*80)
        for key in sorted_keys:
            value = data[key]

            # Skip base64 encoded images (they're too long)
            if 'Camera' in key or 'Image' in key:
                print(f'\n{key}:')
                print(f'  [Base64 encoded image data, length: {len(value)} chars]')

            # For array-like data, show the type and sample
            elif isinstance(value, str) and ' ' in value:
                try:
                    # Try to parse as space-separated numbers
                    nums = value.split()
                    if len(nums) > 10:
                        print(f'\n{key}:')
                        print(f'  [Array with {len(nums)} elements]')
                        print(f'  First 5: {" ".join(nums[:5])}')
                        print(f'  Last 5:  {" ".join(nums[-5:])}')
                    else:
                        print(f'\n{key}:')
                        print(f'  {value}')
                except:
                    print(f'\n{key}:')
                    print(f'  {value}')

            # For simple values
            else:
                print(f'\n{key}:')
                print(f'  {value}')

        print('\n' + '='*80)
        print('DATA INSPECTION COMPLETE')
        print('='*80)
        print('\nPress Ctrl+C to exit\n')

        # Mark as printed so we only do this once
        data_printed = True

    # Send back neutral commands to keep the connection alive
    try:
        # Assume vehicle ID is V1 (adjust if needed)
        json_msg = {
            'V1 Throttle': '0.0',
            'V1 Steering': '0.0'
        }
        sio.emit('Bridge', data=json_msg)
    except Exception as e:
        print(f'Error sending response: {e}')

################################################################################

if __name__ == '__main__':
    print('='*80)
    print('AutoDRIVE Simulator Data Inspector')
    print('='*80)
    print('\nStarting Socket.IO server on port 4567...')
    print('Please start the AutoDRIVE simulator and connect to this script.\n')

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
