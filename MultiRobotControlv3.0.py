import cv2
import numpy as np
import serial
import time
import math

# =========================
# Serial → Arduino
# =========================
arduino = serial.Serial(port='COM12', baudrate=115200, timeout=0)
time.sleep(2)

# =========================
# Waypoints per robot ID
# =========================
waypoints = {
    1: [(300, 150), (350, 150), (400, 150), (450, 150)],
    2: [(300, 350), (350, 350), (400, 350), (450, 350)],  # NEW: mid row
    3: [(300, 550), (350, 550), (400, 550), (450, 550)],
}

# =========================
# Per-robot runtime state
# =========================
robot_state = {
    rid: {
        "wp_idx": 0,          # current waypoint index
        "nxtwp_flag": 0,      # 0 = not advanced yet for this arrival; 1 = advanced
        "target_px": None,    # current target in pixels for this robot
        "center_px": (None, None),  # detected bot center (pixels)
        "pose": (None, None),  # (rvec, tvec)
        "last_cmd": "",       # last command sent (for debug if needed)
    }
    for rid in waypoints.keys()
}

last_command = ""
last_sent_time = 0
send_interval = 0.05  # 50 ms

def send_command(cmd: str):
    global last_command, last_sent_time
    now = time.time()
    if cmd != last_command or (now - last_sent_time) > send_interval:
        arduino.write((cmd + '\n').encode())
        arduino.flush()
        last_command = cmd
        last_sent_time = now

# =========================
# Camera intrinsics
# =========================
fs = cv2.FileStorage("camera_calibration_1.yaml", cv2.FILE_STORAGE_READ)
camera_matrix = fs.getNode("camera_matrix").mat().astype(np.float64)
dist_coeffs   = fs.getNode("dist_coeff").mat().astype(np.float64)  # keep your key name
fs.release()

# =========================
# ArUco setup
# =========================
marker_length = 0.056  # meters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

# =========================
# Arena boundary (image pixels)
# =========================
boundary_img = np.array([[5, 715], [1275, 715], [1275, 5], [5, 5]], dtype=np.int32)

# =========================
# Mouse: set (or reset) the current target to the *current* waypoint index
# for each robot, without changing indices.
# =========================
def set_target(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        inside = cv2.pointPolygonTest(boundary_img, (x, y), False)
        if inside >= 0:
            for rid in robot_state.keys():
                idx = robot_state[rid]["wp_idx"]
                if idx < len(waypoints[rid]):
                    robot_state[rid]["target_px"] = waypoints[rid][idx]
                else:
                    robot_state[rid]["target_px"] = None
                robot_state[rid]["nxtwp_flag"] = 0  # allow advancing on arrival

# =========================
# Helper: pixel → world on Z=0 plane of the *marker* frame
# =========================
def pixel_to_world_on_plane(u, v, rvec, tvec, K, dist):
    """
    Map pixel (u,v) to the Z=0 plane of the ArUco marker (its local world).
    Returns np.array([Xw, Yw, Zw]) in meters (Zw≈0), or None on failure.
    """
    if rvec is None or tvec is None or u is None or v is None:
        return None
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, dist)  # normalized camera coords
    x_n, y_n = und[0, 0]

    R, _ = cv2.Rodrigues(rvec)               # world->camera
    ray_c = np.array([x_n, y_n, 1.0], dtype=np.float64)  # direction in camera
    ray_w = R.T @ ray_c                       # direction in world (marker) frame
    Cw   = (-R.T @ tvec).ravel()              # camera center in world frame

    denom = ray_w[2]
    if abs(denom) < 1e-9:
        return None
    t = -Cw[2] / denom
    Pw = Cw + t * ray_w
    return Pw  # [Xw, Yw, 0]

# =========================
# Control thresholds & commands
# =========================
STOP_DIST_M     = 0.02       # 2 cm
ANG_THRESH_DEG  = 6          # align within 6°

# Per-robot command strings
CMD = {
    1: {"FWD":"10001", "BCK":"10010", "LEFT":"10011", "RIGHT":"10100", "STOP":"10000"},
    2: {"FWD":"20001", "BCK":"20010", "LEFT":"20011", "RIGHT":"20100", "STOP":"20000"},
    3: {"FWD":"30001", "BCK":"30010", "LEFT":"30011", "RIGHT":"30100", "STOP":"30000"},
}

# Robot forward axis in its marker/world frame (up on print)
FORWARD_AXIS_MARKER = np.array([0.0, 1.0], dtype=np.float64)
TURN_SIGN = +1   # flip to -1 if turning direction is inverted

def signed_angle_deg(from_vec, to_vec):
    fx, fy = from_vec
    tx, ty = to_vec
    fn = math.hypot(fx, fy) or 1.0
    tn = math.hypot(tx, ty) or 1.0
    fx, fy = fx / fn, fy / fn
    tx, ty = tx / tn, ty / tn
    cross = fx * ty - fy * tx
    dot   = fx * tx + fy * ty
    return math.degrees(math.atan2(cross, dot))

# =========================
# Video
# =========================
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("Bot Tracking")
cv2.setMouseCallback("Bot Tracking", set_target)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)

def marker_center_px(corners_array):
    """Compute (cx, cy) from a (4,2) corners array."""
    x_min, y_min = np.min(corners_array, axis=0)
    x_max, y_max = np.max(corners_array, axis=0)
    return int((x_min + x_max) * 0.5), int((y_min + y_max) * 0.5)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.convertScaleAbs(frame, alpha=0.6, beta=-9)

    # Draw boundary & waypoint markers (current targets)
    cv2.polylines(frame, [boundary_img], True, (0, 255, 0), 2)
    for rid, st in robot_state.items():
        if st["target_px"] is not None:
            color = {1:(0,0,255), 2:(255,0,255), 3:(0,255,255)}.get(rid, (255,255,255))
            cv2.circle(frame, st["target_px"], 6, color, -1)
            cv2.putText(frame, f"Tgt{rid}(px)", (st["target_px"][0] + 8, st["target_px"][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    # Default STOP for each robot each frame
    pending_cmds = {rid: CMD[rid]["STOP"] for rid in robot_state.keys()}
    dbg_text = []

    if ids is not None and len(ids) > 0:
        # Pose estimation for each detected marker (keeping order aligned with corners)
        rvecs = []
        tvecs = []
        objPoints = np.array([
            [-marker_length/2,  marker_length/2, 0],
            [ marker_length/2,  marker_length/2, 0],
            [ marker_length/2, -marker_length/2, 0],
            [-marker_length/2, -marker_length/2, 0]
        ], dtype=np.float32)

        for corner in corners:
            success, rvec, tvec = cv2.solvePnP(objPoints, corner[0], camera_matrix, dist_coeffs)
            if success:
                rvecs.append(rvec.reshape(3,1))
                tvecs.append(tvec.reshape(3,1))
            else:
                rvecs.append(None)
                tvecs.append(None)

        id_to_idx = {int(idv): i for i, idv in enumerate(ids.flatten())}
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Populate per-robot pose + center if available
        for rid in robot_state.keys():
            st = robot_state[rid]
            if rid in id_to_idx:
                i = id_to_idx[rid]
                rvec = rvecs[i]
                tvec = tvecs[i]
                st["pose"] = (rvec, tvec)

                if rvec is not None and tvec is not None:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.7)
                pts = corners[i][0]
                cx, cy = marker_center_px(pts)
                st["center_px"] = (cx, cy)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            else:
                st["pose"] = (None, None)
                st["center_px"] = (None, None)

        # Control for each robot independently
        for rid, st in robot_state.items():
            rvec, tvec = st["pose"]
            cx, cy = st["center_px"]
            target_px = st["target_px"]

            Pw_bot = pixel_to_world_on_plane(cx, cy, rvec, tvec, camera_matrix, dist_coeffs) if (cx is not None) else None
            Pw_tgt = (pixel_to_world_on_plane(target_px[0], target_px[1], rvec, tvec, camera_matrix, dist_coeffs)
                      if (target_px is not None and rvec is not None) else None)

            if Pw_bot is not None and Pw_tgt is not None:
                v_to_tgt = np.array([float(Pw_tgt[0] - Pw_bot[0]), float(Pw_tgt[1] - Pw_bot[1])], dtype=np.float64)
                dist_m  = float(np.hypot(v_to_tgt[0], v_to_tgt[1]))
                ang_err = signed_angle_deg(FORWARD_AXIS_MARKER, v_to_tgt)

                dbg_text.append(f"ID{rid} dist={dist_m:.3f} m | ang={ang_err:.1f}°")

                if dist_m < STOP_DIST_M:
                    pending_cmds[rid] = CMD[rid]["STOP"]
                    if st["nxtwp_flag"] == 0:
                        st["wp_idx"] += 1
                        if st["wp_idx"] < len(waypoints[rid]):
                            st["target_px"] = waypoints[rid][st["wp_idx"]]
                        else:
                            st["target_px"] = None  # finished
                        st["nxtwp_flag"] = 1
                else:
                    st["nxtwp_flag"] = 0
                    signed_err = TURN_SIGN * ang_err
                    if abs(signed_err) > ANG_THRESH_DEG:
                        pending_cmds[rid] = CMD[rid]["LEFT"] if signed_err > 0 else CMD[rid]["RIGHT"]
                    else:
                        pending_cmds[rid] = CMD[rid]["FWD"]

    # Send all robot commands each loop
    for rid in sorted(pending_cmds.keys()):
        send_command(pending_cmds[rid])

    # HUD
    y = 40
    for rid in sorted(robot_state.keys()):
        cv2.putText(frame, f"Cmd{rid}: {pending_cmds[rid]}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 32
    for i, s in enumerate(dbg_text):
        cv2.putText(frame, s, (20, y + i*26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

    cv2.imshow("Bot Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        for rid in robot_state.keys():
            send_command(CMD[rid]["STOP"])
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
