import cv2
import numpy as np
import pyrealsense2 as rs

# Start and configure pipeline
def configure_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    return pipeline, config

def start_pipeline(pipeline, config):
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 3)  # 0: Default, 1: Short range, 2: Mid range, 3: Long range 
    return profile

def main():
    pipeline, config = configure_pipeline()
    start_pipeline(pipeline, config)

    # Set up align and pointcloud objects
    align_to = rs.stream.color
    align = rs.align(align_to)
    pc = rs.pointcloud()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Generate point cloud
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03),
                cv2.COLORMAP_JET)

            # Display the images
            images = np.hstack((color_image, depth_colormap))
            cv2.imshow('RealSense', images)

            # Handle keyboard events
            key = cv2.waitKey(1)
            if key == ord("e"):
                points.export_to_ply('./output.ply', color_frame)
            elif key in (27, ord("q")) or cv2.getWindowProperty('RealSense', cv2.WND_PROP_AUTOSIZE) < 0:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
