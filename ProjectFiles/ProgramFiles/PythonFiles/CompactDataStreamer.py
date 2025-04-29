import socket
import struct
import os
import time
import math

# === CONFIG ===
UDP_IP = "0.0.0.0"
UDP_PORT = 3000
OUTPUT_FOLDER = r"C:\Users\Quinn\Documents\multiscan100_DataCollection\SavedData"
SAVE_INTERVAL = 3  # seconds between frames

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# === SOCKET SETUP ===
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(10)

# === ANGLES PER LAYER (SICK MultiScan default estimates in degrees) ===
VERTICAL_ANGLES = [
    -15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0,
    1.0, 3.0, 5.0, 7.0
]

# === SAVE OBJ ===
def save_obj(points):
    timestamp = int(time.time())
    path = os.path.join(OUTPUT_FOLDER, f"compact_fullscan_{timestamp}.obj")
    with open(path, 'w') as f:
        for (x, y, z) in points:
            f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
    print(f"[OBJ SAVED] {path} with {len(points)} points")

# === PARSE PACKET ===
def parse_compact_packet(packet):
    try:
        if len(packet) < 32:
            print("[SKIP] Packet too short.")
            return None

        header = struct.unpack_from('<IIQQII', packet, 0)
        start_of_frame = header[0]
        command_id = header[1]
        telegram_counter = header[2]
        timestamp = header[3]
        telegram_version = header[4]
        size_module_0 = header[5]

        if start_of_frame != 0x02020202:
            print("[SKIP] Invalid start of frame header.")
            return None

        print(f"[HEADER] Frame: {telegram_counter}, SizeModule0: {size_module_0}")

        if len(packet) < 32 + size_module_0:
            print("[SKIP] Incomplete module data.")
            return None

        module_data = packet[32:32 + size_module_0]
        print(f"[DEBUG] First 16 bytes of module: {module_data[:16].hex(' ')}")

        if len(module_data) < 16:
            print("[SKIP] Module too small for metadata.")
            return None

        layer_index = struct.unpack_from('<I', module_data, 0)[0]
        echo_count = struct.unpack_from('<I', module_data, 8)[0]
        next_module_size = struct.unpack_from('<I', module_data, 12)[0]

        print(f"[META] Layer: {layer_index}, Echoes: {echo_count}, NextModuleSize: {next_module_size}")

        points = []
        data = module_data[16:]
        num_points = len(data) // 4
        vertical_angle = math.radians(VERTICAL_ANGLES[layer_index % len(VERTICAL_ANGLES)])

        for idx in range(num_points):
            word = struct.unpack_from('<I', data, idx * 4)[0]

            distance_mm = word & 0xFFFFF
            reflectivity = (word >> 20) & 0x3F
            azimuth_raw = (word >> 26) & 0x3FF  # 10-bit azimuth value

            if distance_mm == 0:
                continue

            distance_m = distance_mm / 1000.0
            azimuth_deg = (azimuth_raw / 1024.0) * 360.0
            azimuth_rad = math.radians(azimuth_deg)

            # CORRECTED CONVERSION: Align with SICK coordinate frame
            x = distance_m * math.cos(vertical_angle) * math.cos(azimuth_rad)
            y = distance_m * math.cos(vertical_angle) * math.sin(azimuth_rad)
            z = distance_m * math.sin(vertical_angle)
            points.append((x, y, z))

        print(f"[POINTS] Parsed {len(points)} points from layer {layer_index}")
        return telegram_counter, points

    except Exception as e:
        print("[ERROR in parse_compact_packet]", e)
        return None

# === MAIN LOOP ===
def main():
    last_save_time = 0
    seen_frames = set()
    all_points = []

    print("[INFO] Listening for Compact-format LiDAR data (with SICK framing)...")

    while True:
        try:
            packet, _ = sock.recvfrom(65535)
            result = parse_compact_packet(packet)
            if not result:
                continue

            frame_id, points = result
            if frame_id in seen_frames or len(points) == 0:
                continue

            seen_frames.add(frame_id)
            all_points.extend(points)

            now = time.time()
            if now - last_save_time >= SAVE_INTERVAL:
                save_obj(all_points)
                last_save_time = now

        except socket.timeout:
            print("[TIMEOUT] No data received.")
        except KeyboardInterrupt:
            print("[EXIT] Stopped by user. Saving full point cloud...")
            save_obj(all_points)
            break
        except Exception as e:
            print("[ERROR]", e)

if __name__ == "__main__":
    main()
