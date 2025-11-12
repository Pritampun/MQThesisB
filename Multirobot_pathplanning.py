"""
Threads:
  A) planner_control_thread  – path planning + gating + control decisions
  B) vision_thread           – camera capture + ArUco detection + pose update
  C) comms_thread            – serial link to Arduino TX, rate-limited sending

Protocol:
  5-char commands rAAAA, e.g., 10001 (R1 FWD). First digit is robot ID.
Quit: press 'q' in the OpenCV preview window or Ctrl+C in terminal.
"""

import cv2
import numpy as np
import serial, time, math, random, sys, signal
from typing import Dict, Tuple, Optional, List
from threading import Thread, Event, Lock
from queue import Queue, Empty

# =========================
# ---- Config
# =========================
WIDTH, HEIGHT     = 1280, 720
MARGIN            = 40
WPS_PER_ROBOT     = 4
PATH_MIN_SEP      = 120.0
RANDOM_SEED       = 7

# Gating (pixel-domain planning)
NEAR_PIXELS       = 36
APPROACH_PIXELS   = 12
CLEAR_MARGIN      = 0.01

# Control (marker-frame)
MARKER_LENGTH_M   = 0.056
STOP_DIST_M       = 0.020
ANG_THRESH_DEG    = 6.0
ANG_RELEASE_DEG   = 3.0
FORWARD_AXIS_MARKER = np.array([0.0, 1.0], dtype=np.float64)
TURN_SIGN         = +1
LOST_TIMEOUT_S    = 0.30

# Serial → Arduino TX
SERIAL_PORT       = "COM12" # adjust as needed
BAUD              = 115200
SEND_INTERVAL_S   = 0.05      # rate-limit per robot
RESEND_PERIOD_S   = 0.12      # keep-alive
TX_QUEUE_MAXSIZE  = 100

# IDs
ARUCO_TO_ROBOT    = {1:1, 2:2, 3:3}  # map marker ID → robot ID (1..3)

CMD = {
    1: {"FWD":"10001", "BCK":"10010", "LEFT":"10011", "RIGHT":"10100", "STOP":"10000"},
    2: {"FWD":"20001", "BCK":"20010", "LEFT":"20011", "RIGHT":"20100", "STOP":"20000"},
    3: {"FWD":"30001", "BCK":"30010", "LEFT":"30011", "RIGHT":"30100", "STOP":"30000"},
}

# =========================
# ---- Geometry helpers (pixels)
# =========================
Point = Tuple[float, float]
Segment = Tuple[Point, Point]

def dist_px(a: Point, b: Point) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def seg_intersection_point(p: Point, p2: Point, q: Point, q2: Point) -> Optional[Point]:
    (x1, y1), (x2, y2) = p, p2
    (x3, y3), (x4, y4) = q, q2
    r = (x2 - x1, y2 - y1)
    s = (x4 - x3, y4 - y3)
    rxs = r[0]*s[1] - r[1]*s[0]
    q_p = (x3 - x1, y3 - y1)
    qpxr = q_p[0]*r[1] - q_p[1]*r[0]
    if abs(rxs) < 1e-9:
        return None
    t = (q_p[0]*s[1] - q_p[1]*s[0]) / rxs
    u = qpxr / rxs
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return (x1 + t * r[0], y1 + t * r[1])
    return None

def segments_from_waypoints(wps: List[Point]) -> List[Segment]:
    return list(zip(wps[:-1], wps[1:]))

def random_point(margin=MARGIN) -> Point:
    return (random.uniform(margin, WIDTH - margin),
            random.uniform(margin, HEIGHT - margin))

def generate_waypoints(n: int = 4, min_sep: float = 80.0) -> List[Point]:
    pts: List[Point] = []
    tries = 0
    while len(pts) < n and tries < 10000:
        tries += 1
        p = random_point()
        if all(dist_px(p, q) >= min_sep for q in pts):
            pts.append(p)
    while len(pts) < n:
        pts.append(random_point())
    return pts

def segment_lengths_px(path: List[Point]) -> List[float]:
    return [dist_px(path[i], path[i+1]) for i in range(len(path)-1)]

def param_t_on_segment(a: Point, b: Point, p: Point) -> float:
    ax, ay = a; bx, by = b; px, py = p
    vx, vy = (bx-ax, by-ay)
    L2 = vx*vx + vy*vy
    if L2 <= 1e-9: return 0.0
    t = ((px-ax)*vx + (py-ay)*vy) / L2
    return max(0.0, min(1.0, t))

# =========================
# ---- Crossing/queue helpers
# =========================
def build_pair_details(segsA: List[Segment], segsB: List[Segment],
                       nameA: str, nameB: str):
    mat = [[0]*len(segsB) for _ in range(len(segsA))]
    hits: List[Point] = []
    events_for_A: List[List[Tuple[str, int, float, Point]]] = [[] for _ in range(len(segsA))]
    events_for_B: List[List[Tuple[str, int, float, Point]]] = [[] for _ in range(len(segsB))]

    for i, (a0, a1) in enumerate(segsA):
        ax, ay = a0
        dax, day = (a1[0]-ax, a1[1]-ay)
        alen2 = dax*dax + day*day
        for j, (b0, b1) in enumerate(segsB):
            p = seg_intersection_point(a0, a1, b0, b1)
            if p is None: continue
            mat[i][j] = 1
            hits.append(p)

            tA = 0.0 if alen2 == 0 else ((p[0]-ax)*dax + (p[1]-ay)*day) / alen2
            tA = max(0.0, min(1.0, tA))
            bx, by = b0
            dbx, dby = (b1[0]-bx, b1[1]-by)
            blen2 = dbx*dbx + dby*dby
            tB = 0.0 if blen2 == 0 else ((p[0]-bx)*dbx + (p[1]-by)*dby) / blen2
            tB = max(0.0, min(1.0, tB))

            events_for_A[i].append((nameB, j, tA, p))
            events_for_B[j].append((nameA, i, tB, p))

    for i in range(len(events_for_A)):
        events_for_A[i].sort(key=lambda x: x[2])
    for j in range(len(events_for_B)):
        events_for_B[j].sort(key=lambda x: x[2])
    return mat, hits, events_for_A, events_for_B

def norm_key(a_name:str, a_seg:int, b_name:str, b_seg:int):
    return (a_name, a_seg, b_name, b_seg) if a_name < b_name else (b_name, b_seg, a_name, a_seg)

# =========================
# ---- Camera / ArUco helpers
# =========================
def pixel_to_world_on_plane(u, v, rvec, tvec, K, D):
    if rvec is None or tvec is None or u is None or v is None:
        return None
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, D)  # normalized
    x_n, y_n = und[0,0]
    R,_ = cv2.Rodrigues(rvec)
    ray_c = np.array([x_n, y_n, 1.0], dtype=np.float64)
    ray_w = R.T @ ray_c
    Cw   = (-R.T @ tvec).ravel()
    denom = ray_w[2]
    if abs(denom) < 1e-9: return None
    t = -Cw[2] / denom
    return (Cw + t*ray_w)

def signed_angle_deg(from_vec, to_vec):
    fx, fy = from_vec; tx, ty = to_vec
    fn = math.hypot(fx, fy) or 1.0
    tn = math.hypot(tx, ty) or 1.0
    fx, fy = fx/fn, fy/fn; tx, ty = tx/tn, ty/tn
    cross = fx*ty - fy*tx
    dot   = fx*tx + fy*ty
    return math.degrees(math.atan2(cross, dot))

def marker_center_px(corners_array):
    x_min, y_min = np.min(corners_array, axis=0)
    x_max, y_max = np.max(corners_array, axis=0)
    return int((x_min + x_max)*0.5), int((y_min + y_max)*0.5)

# =========================
# ---- Shared state
# =========================
class Shared:
    def __init__(self):
        self.shutdown = Event()
        self.lock = Lock()

        # camera / calibration
        fs = cv2.FileStorage("camera_calibration_1.yaml", cv2.FILE_STORAGE_READ)
        self.K = fs.getNode("camera_matrix").mat().astype(np.float64)
        dn = fs.getNode("dist_coeffs")
        if dn.empty(): dn = fs.getNode("dist_coeff")
        self.D = dn.mat().astype(np.float64)
        fs.release()

        # vision (per robot)
        self.pose: Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {1:(None,None),2:(None,None),3:(None,None)}
        self.center_px: Dict[int, Tuple[Optional[int], Optional[int]]] = {1:(None,None),2:(None,None),3:(None,None)}
        self.last_seen: Dict[int, float] = {1:0.0,2:0.0,3:0.0}

        # planning
        random.seed(RANDOM_SEED)
        self.waypoints: Dict[int, List[Point]] = {
            1: generate_waypoints(WPS_PER_ROBOT, PATH_MIN_SEP),
            2: generate_waypoints(WPS_PER_ROBOT, PATH_MIN_SEP),
            3: generate_waypoints(WPS_PER_ROBOT, PATH_MIN_SEP),
        }
        self.segments: Dict[int, List[Segment]] = {r: segments_from_waypoints(self.waypoints[r]) for r in [1,2,3]}
        self.seg_lens: Dict[int, List[float]]   = {r: segment_lengths_px(self.waypoints[r]) for r in [1,2,3]}

        # pair events for queues
        m12, h12, e1v2, e2v1 = build_pair_details(self.segments[1], self.segments[2], "R1", "R2")
        m13, h13, e1v3, e3v1 = build_pair_details(self.segments[1], self.segments[3], "R1", "R3")
        m23, h23, e2v3, e3v2 = build_pair_details(self.segments[2], self.segments[3], "R2", "R3")
        self.events_by_seg: Dict[int, List[List[Tuple[str,int,float,Point]]]] = {
            1: [[] for _ in range(len(self.segments[1]))],
            2: [[] for _ in range(len(self.segments[2]))],
            3: [[] for _ in range(len(self.segments[3]))],
        }
        for i in range(len(self.segments[1])): self.events_by_seg[1][i] = sorted(e1v2[i]+e1v3[i], key=lambda x:x[2])
        for i in range(len(self.segments[2])): self.events_by_seg[2][i] = sorted(e2v1[i]+e2v3[i], key=lambda x:x[2])
        for i in range(len(self.segments[3])): self.events_by_seg[3][i] = sorted(e3v1[i]+e3v2[i], key=lambda x:x[2])

        # queue structures
        self.crossing_q: Dict[Tuple[str,int,str,int], List[Tuple[str,int]]] = {}
        self.ticket = 0

        # per-robot progress on plan
        self.wp_idx: Dict[int,int] = {1:0,2:0,3:0}
        self.event_ptr: Dict[int,Dict[int,int]] = {1:{},2:{},3:{}}
        self.segment_enq_done: Dict[int,set] = {1:set(),2:set(),3:set()}
        self.enqueued_keys: Dict[int,set] = {1:set(),2:set(),3:set()}

        # last commands for hysteresis + comms
        self.last_cmd: Dict[int,str] = {1:CMD[1]["STOP"],2:CMD[2]["STOP"],3:CMD[3]["STOP"]}

        # queues
        self.tx_queue: "Queue[Tuple[int,str]]" = Queue(maxsize=TX_QUEUE_MAXSIZE)

def ensure_in_queue(sh: Shared, key, who):
    q = sh.crossing_q.setdefault(key, [])
    if not any(w == who for (w, _) in q):
        sh.ticket += 1
        q.append((who, sh.ticket))
        q.sort(key=lambda x: x[1])

def pop_if_head(sh: Shared, key, who):
    q = sh.crossing_q.get(key, [])
    if q and q[0][0] == who:
        q.pop(0)
    if not q:
        sh.crossing_q.pop(key, None)

def q_head(sh: Shared, key):
    q = sh.crossing_q.get(key, [])
    return q[0][0] if q else "none"

# =========================
# ---- Vision thread (ArUco)
# =========================
def vision_thread(sh: Shared):
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    objPoints = np.array([
        [-MARKER_LENGTH_M/2,  MARKER_LENGTH_M/2, 0],
        [ MARKER_LENGTH_M/2,  MARKER_LENGTH_M/2, 0],
        [ MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0],
        [-MARKER_LENGTH_M/2, -MARKER_LENGTH_M/2, 0]
    ], dtype=np.float32)

    while not sh.shutdown.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        now = time.time()
        if ids is not None and len(ids) > 0:
            # Estimate poses
            rvecs, tvecs = [], []
            for corner in corners:
                okp, rvec, tvec = cv2.solvePnP(objPoints, corner[0], sh.K, sh.D)
                rvecs.append(rvec if okp else None)
                tvecs.append(tvec if okp else None)

            id2idx = {int(x): i for i, x in enumerate(ids.flatten())}

            with sh.lock:
                for ar_id, k in id2idx.items():
                    if ar_id not in ARUCO_TO_ROBOT:
                        continue
                    rid = ARUCO_TO_ROBOT[ar_id]
                    rvec, tvec = rvecs[k], tvecs[k]
                    pts = corners[k][0]
                    cx, cy = marker_center_px(pts)

                    sh.pose[rid] = (rvec, tvec)
                    sh.center_px[rid] = (cx, cy)
                    sh.last_seen[rid] = now

        # Minimal preview + quit
        cv2.putText(frame, "Press q to quit", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Vision (ArUco)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            sh.shutdown.set()
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# ---- Planner + Control thread
# =========================
def planner_control_thread(sh: Shared):
    # Helper for command decision with hysteresis
    def pick_cmd(rid: int, ang_deg: float, arrived: bool) -> str:
        if arrived:
            return CMD[rid]["STOP"]
        last = sh.last_cmd[rid]
        thresh = ANG_THRESH_DEG if (last.endswith("001") or last.endswith("000")) else ANG_RELEASE_DEG
        if abs(ang_deg) > thresh:
            return CMD[rid]["LEFT"] if ang_deg > 0 else CMD[rid]["RIGHT"]
        else:
            return CMD[rid]["FWD"]

    while not sh.shutdown.is_set():
        now = time.time()
        # default STOP if unseen recently
        pending: Dict[int, str] = {1:CMD[1]["STOP"],2:CMD[2]["STOP"],3:CMD[3]["STOP"]}

        with sh.lock:
            for rid in [1,2,3]:
                # finished path?
                if sh.wp_idx[rid] >= len(sh.waypoints[rid]):
                    pending[rid] = CMD[rid]["STOP"]
                    continue

                # vision available?
                cx, cy = sh.center_px[rid]
                rvec, tvec = sh.pose[rid]
                if (cx is None) or (cy is None) or (now - sh.last_seen[rid] > LOST_TIMEOUT_S):
                    pending[rid] = CMD[rid]["STOP"]
                    continue

                # current segment
                segs = sh.segments[rid]
                lens = sh.seg_lens[rid]
                events = sh.events_by_seg[rid]
                i = sh.wp_idx[rid]
                if i >= len(segs):
                    pending[rid] = CMD[rid]["STOP"]
                    continue

                # Multi-reserve: enqueue once
                if i not in sh.segment_enq_done[rid]:
                    for (oname, oseg, _, _) in events[i]:
                        key = norm_key(f"R{rid}", i, oname, oseg)
                        if key not in sh.enqueued_keys[rid]:
                            ensure_in_queue(sh, key, f"R{rid}")
                            sh.enqueued_keys[rid].add(key)
                    sh.segment_enq_done[rid].add(i)

                # progress along segment (pixel projection)
                a, b = segs[i]
                t_on = param_t_on_segment(a, b, (cx, cy))
                s_along = t_on * lens[i]

                # Gate near crossing if not head
                ptr = sh.event_ptr[rid].get(i, 0)
                gated = False
                if ptr < len(events[i]):
                    oname, oseg, t_cross, _ = events[i][ptr]
                    key = norm_key(f"R{rid}", i, oname, oseg)
                    s_cross = t_cross * lens[i]
                    dpx = s_cross - s_along
                    if q_head(sh, key) != f"R{rid}" and dpx <= NEAR_PIXELS:
                        pending[rid] = CMD[rid]["STOP"]
                        gated = True

                # Not gated → steer to end waypoint in marker world
                if not gated:
                    tgt_px = sh.waypoints[rid][i+1]  # end of current segment
                    Pw_bot = pixel_to_world_on_plane(cx, cy, rvec, tvec, sh.K, sh.D) if rvec is not None else None
                    Pw_tgt = pixel_to_world_on_plane(tgt_px[0], tgt_px[1], rvec, tvec, sh.K, sh.D) if rvec is not None else None
                    if (Pw_bot is None) or (Pw_tgt is None):
                        pending[rid] = CMD[rid]["STOP"]
                    else:
                        v = np.array([float(Pw_tgt[0]-Pw_bot[0]), float(Pw_tgt[1]-Pw_bot[1])], dtype=np.float64)
                        d = float(np.hypot(v[0], v[1]))
                        ang = signed_angle_deg(FORWARD_AXIS_MARKER, v) * TURN_SIGN

                        if d < STOP_DIST_M:
                            pending[rid] = CMD[rid]["STOP"]
                            # advance to next segment
                            sh.wp_idx[rid] += 1
                            # cleanup queues tied to finished segment
                            for (oname, oseg, _, _) in events[i]:
                                key = norm_key(f"R{rid}", i, oname, oseg)
                                if q_head(sh, key) == f"R{rid}":
                                    pop_if_head(sh, key, f"R{rid}")
                                sh.enqueued_keys[rid].discard(key)
                            sh.event_ptr[rid][i] = 0
                        else:
                            pending[rid] = pick_cmd(rid, ang, arrived=False)

                # If head and passed crossing → pop
                ptr = sh.event_ptr[rid].get(i, 0)
                if ptr < len(events[i]):
                    oname, oseg, t_cross, _ = events[i][ptr]
                    key = norm_key(f"R{rid}", i, oname, oseg)
                    if q_head(sh, key) == f"R{rid}" and t_on >= (t_cross + CLEAR_MARGIN - 1e-6):
                        pop_if_head(sh, key, f"R{rid}")
                        sh.event_ptr[rid][i] = ptr + 1

        # Push commands to TX queue (non-blocking)
        for rid, cmd in pending.items():
            try:
                sh.tx_queue.put_nowait((rid, cmd))
            except:
                pass

        # small pacing
        time.sleep(0.01)

# =========================
# ---- Communication thread
# =========================
def comms_thread(sh: Shared):
    def open_serial():
        try:
            s = serial.Serial(port=SERIAL_PORT, baudrate=BAUD, timeout=0)
            time.sleep(1.0)
            print(f"[SER] Connected {SERIAL_PORT}")
            return s
        except Exception as e:
            print(f"[SER] Open fail: {e}")
            return None

    ser = open_serial()
    last_sent: Dict[int, Tuple[str, float]] = {1:(CMD[1]["STOP"],0.0), 2:(CMD[2]["STOP"],0.0), 3:(CMD[3]["STOP"],0.0)}

    def try_send(rid: int, cmd: str):
        nonlocal ser
        now = time.time()
        last_cmd, last_t = last_sent[rid]
        if (cmd != last_cmd) or (now - last_t) >= SEND_INTERVAL_S:
            payload = (cmd + "\n").encode()
            try:
                if ser is None or not ser.is_open:
                    ser = open_serial()
                if ser:
                    ser.write(payload)
                    ser.flush()
                    last_sent[rid] = (cmd, now)
                    with sh.lock:
                        sh.last_cmd[rid] = cmd
            except Exception as e:
                print(f"[SER] Write err: {e}")
                try:
                    if ser: ser.close()
                except: pass
                ser = None

    while not sh.shutdown.is_set():
        # 1) Drain queue quickly
        drained = []
        try:
            while True:
                drained.append(sh.tx_queue.get_nowait())
        except Empty:
            pass

        # keep only the last command per rid this cycle
        latest: Dict[int,str] = {}
        for rid, cmd in drained:
            latest[rid] = cmd

        # 2) Send any updates
        for rid, cmd in latest.items():
            try_send(rid, cmd)

        # 3) Keep-alive resend
        now = time.time()
        for rid in [1,2,3]:
            _, t0 = last_sent[rid]
            if now - t0 >= RESEND_PERIOD_S:
                try_send(rid, sh.last_cmd[rid])

        time.sleep(0.01)

    # shutdown
    try:
        if ser and ser.is_open: ser.close()
    except: pass

# =========================
# ---- Entrypoint
# =========================
def main():
    sh = Shared()

    # graceful Ctrl+C
    def on_sigint(sig, frame):
        print("\n[SYS] SIGINT → shutting down...")
        sh.shutdown.set()
    signal.signal(signal.SIGINT, on_sigint)

    th_v = Thread(target=vision_thread, args=(sh,), daemon=True)
    th_p = Thread(target=planner_control_thread, args=(sh,), daemon=True)
    th_c = Thread(target=comms_thread, args=(sh,), daemon=True)

    th_v.start(); th_p.start(); th_c.start()

    try:
        while not sh.shutdown.is_set():
            time.sleep(0.1)
    finally:
        sh.shutdown.set()
        th_v.join(timeout=2)
        th_p.join(timeout=2)
        th_c.join(timeout=2)
        print("[SYS] Stopped.")

if __name__ == "__main__":
    main()
