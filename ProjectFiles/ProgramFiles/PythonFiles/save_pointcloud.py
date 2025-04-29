import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import math
import asyncio
import websockets
import json
import threading

# Global configurable variables.
SAMPLING_FACTOR = 2      # Process every SAMPLING_FACTOR-th point.
TIME_DELAY = 5.0         # Time delay (in seconds) between sending messages over the WebSocket.
GLOBAL_ROTATION_OFFSET = 0.0  # Fixed rotation (in degrees) applied to every segment.
                               # Adjust this if you need to align your sensor's frame with Unity.

class PointCloudSender(Node):
    def __init__(self):
        super().__init__('pointcloud_sender')
        # Create an event that gets set when Unity is connected.
        self.unity_connected_event = threading.Event()
        
        # Subscribe to the PointCloud2 topic.
        self.subscription = self.create_subscription(
            PointCloud2,
            '/cloud_unstructured_segments',
            self.listener_callback,
            10
        )
        self.max_segments = 12
        self.collected = 0
        self.all_points = []          # Temporary storage for accumulating points in one cycle.
        self.current_point_cloud = [] # Latest complete point cloud to be sent.

        # Start the WebSocket server in a separate thread.
        self.ws_thread = threading.Thread(target=self.start_websocket_server, daemon=True)
        self.ws_thread.start()
    
    def listener_callback(self, msg):
        # Wait until a Unity client has connected before processing data.
        if not self.unity_connected_event.is_set():
            self.get_logger().info("Waiting for Unity connection. Discarding incoming point cloud segment.")
            return

        self.get_logger().info(f"Received segment {self.collected + 1}/{self.max_segments}...")
        segment_points = []
        # Process only every SAMPLING_FACTOR-th point.
        for i, p in enumerate(pc2.read_points(msg, skip_nans=True)):
            if i % SAMPLING_FACTOR == 0:
                x, y, z = p[0], p[1], p[2]
                segment_points.append([x, y, z])
        np_points = np.array(segment_points)
        
        # Apply a fixed, global rotation to all scans.
        # This ensures that every segment retains the same absolute orientation.
        angle_rad = math.radians(GLOBAL_ROTATION_OFFSET)
        rotation_matrix = np.array([
            [math.cos(angle_rad), -math.sin(angle_rad), 0],
            [math.sin(angle_rad),  math.cos(angle_rad), 0],
            [0,                   0,                   1]
        ])
        transformed_points = np.dot(np_points, rotation_matrix.T)

        # Accumulate the (transformed) points.
        self.all_points.extend(transformed_points.tolist())
        self.collected += 1

        # When a complete cycle is collected, update the point cloud and reset.
        if self.collected >= self.max_segments:
            self.get_logger().info('Complete point cloud cycle collected. Updating point cloud data for transmission...')
            self.current_point_cloud = self.all_points.copy()
            self.all_points = []
            self.collected = 0

    def start_websocket_server(self):
        # Start the asynchronous WebSocket server.
        asyncio.run(self.websocket_server())
    
    async def websocket_server(self):
        async def handler(websocket, path):
            # Once a client connects, mark Unity as connected.
            self.get_logger().info("Unity client connected via WebSocket.")
            self.unity_connected_event.set()
            try:
                while True:
                    # Convert the latest point cloud to JSON.
                    json_data = json.dumps(self.current_point_cloud)
                    await websocket.send(json_data)
                    self.get_logger().info("Sent point cloud JSON to Unity.")
                    await asyncio.sleep(TIME_DELAY)
            except websockets.exceptions.ConnectionClosed:
                self.get_logger().info("Unity client disconnected.")
                # Clear the event so that new data is not captured until a client reconnects.
                self.unity_connected_event.clear()
        
        # Listen on port 5525 as requested.
        port = 5525
        server = await websockets.serve(handler, '0.0.0.0', port)
        self.get_logger().info(f"WebSocket server started at ws://0.0.0.0:{port}.")
        # Run the server indefinitely.
        await asyncio.Future()

def main(args=None):
    rclpy.init(args=args)
    sender = PointCloudSender()
    try:
        rclpy.spin(sender)
    except KeyboardInterrupt:
        sender.get_logger().info("Interrupt received, shutting down.")
    finally:
        sender.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
