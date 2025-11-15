import pygame
import random
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Set

# =========================================================
# Config
# =========================================================
WIDTH, HEIGHT = 1280, 720
BOT_WIDTH = 50  # Bot width
BOT_HEIGHT = 35  # Bot height
BOT_MAX_DIM = max(BOT_WIDTH, BOT_HEIGHT)
MARGIN = BOT_MAX_DIM + 10  # Margin from edges (bot size + buffer)
BG = (255, 255, 255)
R_COLORS = {
    "R1": (0, 120, 255),   # blue
    "R2": (40, 200, 40),   # green
    "R3": (255, 170, 0),   # amber
}
WP_COLOR = (255, 30, 30)
HIT_COLOR = (220, 20, 60)
TEXT_COLOR = (0, 0, 0)

SPEED = {"R1": 120.0, "R2": 110.0, "R3": 100.0}  # px/s
POINT_R = 7
INTER_R = 5
LABEL_OFFSET = (10, -18)
RANDOM_SEED = 7

# Gating behavior
# - Bots only brake when they are NEAR a blocked crossing (in pixels).
# - When braking, they stop APPROACH_PIXELS before the crossing.
NEAR_PIXELS      = BOT_MAX_DIM + 20   # start respecting a block when within this many pixels of the crossing
APPROACH_PIXELS  = BOT_MAX_DIM        # come to a stop this many pixels BEFORE the crossing (one bot length)
CLEAR_MARGIN_PX  = BOT_MAX_DIM * 1.5  # pixels to travel past crossing before releasing queue (1.5x bot size)

# =========================================================
# Geometry
# =========================================================
Point = Tuple[float, float]
Segment = Tuple[Point, Point]

def dist(a: Point, b: Point) -> float:
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

def generate_waypoints_poisson(n: int = 4, radius: Optional[float] = None) -> List[Point]:
    """
    Generate waypoints using Poisson disk sampling.
    Ensures points are separated by at least 'radius' distance.
    
    Args:
        n: Number of waypoints to generate
        radius: Minimum separation distance (defaults to 2x bot size)
    """
    if radius is None:
        radius = BOT_MAX_DIM * 2  # Default radius based on bot size
    
    # Poisson disk sampling parameters
    k = 30  # Number of attempts before rejection
    cell_size = radius / math.sqrt(2)
    grid_width = int(math.ceil((WIDTH - 2 * MARGIN) / cell_size))
    grid_height = int(math.ceil((HEIGHT - 2 * MARGIN) / cell_size))
    grid: List[List[Optional[Point]]] = [[None for _ in range(grid_height)] for _ in range(grid_width)]
    
    def grid_coords(p):
        return (int((p[0] - MARGIN) / cell_size), int((p[1] - MARGIN) / cell_size))
    
    def is_valid(p):
        if p[0] < MARGIN or p[0] >= WIDTH - MARGIN or p[1] < MARGIN or p[1] >= HEIGHT - MARGIN:
            return False
        gx, gy = grid_coords(p)
        # Check neighboring cells
        for i in range(max(0, gx - 2), min(grid_width, gx + 3)):
            for j in range(max(0, gy - 2), min(grid_height, gy + 3)):
                neighbor = grid[i][j]
                if neighbor and dist(p, neighbor) < radius:
                    return False
        return True
    
    points = []
    active = []
    
    # Start with a random point
    first = random_point()
    points.append(first)
    active.append(first)
    gx, gy = grid_coords(first)
    grid[gx][gy] = first
    
    # Generate points using Poisson disk sampling
    while active and len(points) < n:
        idx = random.randint(0, len(active) - 1)
        p = active[idx]
        found = False
        
        for _ in range(k):
            # Generate random point in annulus between radius and 2*radius
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(radius, 2 * radius)
            new_p = (p[0] + r * math.cos(angle), p[1] + r * math.sin(angle))
            
            if is_valid(new_p):
                points.append(new_p)
                active.append(new_p)
                gx, gy = grid_coords(new_p)
                grid[gx][gy] = new_p
                found = True
                break
        
        if not found:
            active.pop(idx)
    
    # If we didn't get enough points, fill with random ones
    while len(points) < n:
        p = random_point()
        if all(dist(p, q) >= radius for q in points):
            points.append(p)
    
    return points[:n]

def generate_waypoints(n: int = 4, min_sep: float = 80.0) -> List[Point]:
    """Legacy function - now uses Poisson distribution"""
    return generate_waypoints_poisson(n, min_sep)

def generate_waypoints_from(anchor: Tuple[float, float], n: int = 4, min_sep: float = 80.0) -> List[Point]:
    """First waypoint is the given anchor; others use Poisson distribution."""
    if n < 1:
        return []
    
    if min_sep is None:
        min_sep = BOT_MAX_DIM * 2
    
    pts: List[Point] = [anchor]
    remaining = generate_waypoints_poisson(n - 1, min_sep)
    
    # Filter out points too close to anchor
    for p in remaining:
        if dist(p, anchor) >= min_sep:
            pts.append(p)
    
    # Fill remaining if needed
    tries = 0
    while len(pts) < n and tries < 5000:
        tries += 1
        p = random_point()
        if all(dist(p, q) >= min_sep for q in pts):
            pts.append(p)
    
    return pts[:n]

def segment_lengths(path: List[Point]) -> List[float]:
    return [dist(path[i], path[i+1]) for i in range(len(path)-1)]

# ---------- Pairwise intersection details ----------
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
            if p is None:
                continue
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

def compute_intersection_flags(mat):
    if not mat:
        return [], []
    r = len(mat); c = len(mat[0])
    f_rows = [1 if any(mat[i][j] for j in range(c)) else 0 for i in range(r)]
    f_cols = [1 if any(mat[i][j] for i in range(r)) else 0 for j in range(c)]
    return f_rows, f_cols

# =========================================================
# Pygame setup / drawing
# =========================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Three-Bot Waypoint Simulation (FIFO Multi-Reserve + Near-Crossing Gating)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 16)

def label_waypoint(surface, wp: Point, idx: int):
    txt = f"w{idx}({int(wp[0])},{int(wp[1])})"
    surface.blit(font.render(txt, True, TEXT_COLOR),
                 (wp[0] + LABEL_OFFSET[0], wp[1] + LABEL_OFFSET[1]))

def draw_path(surface, wps: List[Point], color):
    if len(wps) >= 2:
        pygame.draw.lines(surface, color, False, wps, 3)
    for i, p in enumerate(wps, start=1):
        pygame.draw.circle(surface, WP_COLOR, (int(p[0]), int(p[1])), POINT_R)
        label_waypoint(surface, p, i)

def draw_intersections(surface, hits: List[Point]):
    for pt in hits:
        pygame.draw.circle(surface, HIT_COLOR, (int(pt[0]), int(pt[1])), INTER_R)

def render_matrix(surface, title, mat, top_left):
    x0, y0 = top_left
    lh = 18
    surface.blit(font.render(title, True, (90, 90, 90)), (x0, y0))
    y = y0 + lh
    for i, row in enumerate(mat):
        row_str = " ".join(str(v) for v in row)
        surface.blit(font.render(f"{i+1}: {row_str}", True, TEXT_COLOR), (x0, y))
        y += lh

def render_flags(surface, label, flags, pos):
    surface.blit(font.render(f"{label}: {flags}", True, TEXT_COLOR), pos)

def render_traffic(surface, t1, t2, t3, pos):
    x0, y0 = pos
    surface.blit(font.render(f"Traffic R1: {t1}", True, (200, 200, 255)), (x0, y0))
    surface.blit(font.render(f"Traffic R2: {t2}", True, (200, 255, 200)), (x0, y0+18))
    surface.blit(font.render(f"Traffic R3: {t3}", True, (255, 220, 180)), (x0, y0+36))

# =========================================================
# Bot & Traffic
# =========================================================
class BotState:
    def __init__(self, name: str, wps: List[Point], speed: float):
        self.name = name
        self.wps = wps
        self.segs = segments_from_waypoints(wps)
        self.lens = segment_lengths(wps)
        self.speed = speed
        self.seg_idx = 0
        self.s_along = 0.0
        self.pos = wps[0]
        self.done = (len(self.segs) == 0)
        # per segment: list of (other_name, other_seg, t_on_me, point)
        self.events_by_seg: List[List[Tuple[str, int, float, Point]]] = [[] for _ in range(len(self.segs))]
        self.event_ptr: Dict[int, int] = {}
        self.segment_enq_done: Set[int] = set()
        self.enqueued_keys: Set[Tuple[str, int, str, int]] = set()

    def at_end(self):
        return self.seg_idx >= len(self.segs)

    def alpha(self):
        if self.at_end(): return 1.0
        L = self.lens[self.seg_idx]
        if L <= 1e-9: return 1.0
        a = self.s_along / L
        return max(0.0, min(1.0, a))

    def update_pos(self):
        if self.at_end():
            self.pos = self.wps[-1]; self.done = True; return
        a, b = self.segs[self.seg_idx]
        A = self.alpha()
        self.pos = (a[0] + (b[0]-a[0])*A, a[1] + (b[1]-a[1])*A)

# Crossing queues: key = (minName, minSeg, maxName, maxSeg) -> list of (name, ticket)
CROSSING_Q: Dict[Tuple[str,int,str,int], List[Tuple[str,int]]] = {}
TICKET = 0  # global monotonic ticket for unbiased arrival ordering

def norm_key(a_name:str, a_seg:int, b_name:str, b_seg:int):
    if a_name < b_name:
        return (a_name, a_seg, b_name, b_seg)
    else:
        return (b_name, b_seg, a_name, a_seg)

def q_head(key):
    q = CROSSING_Q.get(key, [])
    return q[0][0] if q else "none"

def ensure_in_queue(key, who):
    global TICKET
    q = CROSSING_Q.setdefault(key, [])
    if not any(w == who for (w, _) in q):
        TICKET += 1
        q.append((who, TICKET))
        q.sort(key=lambda x: x[1])  # FIFO by ticket

def pop_if_head(key, who):
    q = CROSSING_Q.get(key, [])
    if q and q[0][0] == who:
        q.pop(0)
    if not q:
        CROSSING_Q.pop(key, None)

def derive_live_flags(bots: Dict[str, BotState]) -> Dict[str, List[int]]:
    # For each queue, whoever is head blocks the other robot's segment
    flags = {name: [0]*len(bot.segs) for name, bot in bots.items()}
    for (n1, s1, n2, s2), q in CROSSING_Q.items():
        if not q: 
            continue
        head = q[0][0]
        if head == n1:
            flags[n2][s2] = 1
        elif head == n2:
            flags[n1][s1] = 1
        else:  # unexpected
            flags[n1][s1] = 1
            flags[n2][s2] = 1
    return flags

# =========================================================
# Scenario builder (3 robots) + Rebuild utilities
# =========================================================
def new_scenario(seed=None):
    if seed is not None:
        random.seed(seed)

    # Minimum separation should account for bot size
    min_separation = BOT_MAX_DIM * 3  # 3x bot size for safe spacing
    
    W = { "R1": generate_waypoints(4, min_separation),
          "R2": generate_waypoints(4, min_separation),
          "R3": generate_waypoints(4, min_separation) }

    bots = {
        "R1": BotState("R1", W["R1"], SPEED["R1"]),
        "R2": BotState("R2", W["R2"], SPEED["R2"]),
        "R3": BotState("R3", W["R3"], SPEED["R3"]),
    }

    mats, hits, static_flags = rebuild_all_pair_data(bots)
    return bots, {n: bots[n].wps for n in bots}, {n: bots[n].segs for n in bots}, mats, hits, static_flags

def rebuild_all_pair_data(bots: Dict[str, BotState]):
    names = ["R1","R2","R3"]
    S = {n: bots[n].segs for n in names}

    m12, h12, e1v2, e2v1 = build_pair_details(S["R1"], S["R2"], "R1", "R2")
    m13, h13, e1v3, e3v1 = build_pair_details(S["R1"], S["R3"], "R1", "R3")
    m23, h23, e2v3, e3v2 = build_pair_details(S["R2"], S["R3"], "R2", "R3")

    # Reset per-bot structures
    for n in names:
        b = bots[n]
        b.events_by_seg = [[] for _ in range(len(b.segs))]
        b.event_ptr.clear()
        b.segment_enq_done.clear()
        b.enqueued_keys.clear()

    # Attach combined events per bot
    for i in range(len(bots["R1"].segs)):
        bots["R1"].events_by_seg[i].extend(e1v2[i])
        bots["R1"].events_by_seg[i].extend(e1v3[i])
        bots["R1"].events_by_seg[i].sort(key=lambda x: x[2])
    for i in range(len(bots["R2"].segs)):
        bots["R2"].events_by_seg[i].extend(e2v1[i])
        bots["R2"].events_by_seg[i].extend(e2v3[i])
        bots["R2"].events_by_seg[i].sort(key=lambda x: x[2])
    for i in range(len(bots["R3"].segs)):
        bots["R3"].events_by_seg[i].extend(e3v1[i])
        bots["R3"].events_by_seg[i].extend(e3v2[i])
        bots["R3"].events_by_seg[i].sort(key=lambda x: x[2])

    # Static flags
    f12_A, f12_B = compute_intersection_flags(m12)
    f13_A, f13_B = compute_intersection_flags(m13)
    f23_A, f23_B = compute_intersection_flags(m23)
    static_flags = {
        "R1_vs_R2": {"R1": f12_A, "R2": f12_B},
        "R1_vs_R3": {"R1": f13_A, "R3": f13_B},
        "R2_vs_R3": {"R2": f23_A, "R3": f23_B},
    }

    mats = {"12": m12, "13": m13, "23": m23}
    hits_all = h12 + h13 + h23
    return mats, hits_all, static_flags

def clear_all_queues():
    CROSSING_Q.clear()
    global TICKET
    TICKET = 0

def regenerate_bot(bots: Dict[str, BotState], name: str):
    """
    Start the new path from the bot's last position (its final waypoint),
    rebuild pair intersections, and clear queues to avoid stale locks.
    """
    b = bots[name]
    anchor = b.pos  # last waypoint when done
    min_separation = BOT_MAX_DIM * 3  # 3x bot size for safe spacing
    new_wps = generate_waypoints_from(anchor, n=4, min_sep=min_separation)

    # Reset this bot with new path
    b.wps = new_wps
    b.segs = segments_from_waypoints(new_wps)
    b.lens = segment_lengths(new_wps)
    b.seg_idx = 0
    b.s_along = 0.0
    b.pos = new_wps[0]          # equals anchor
    b.done = (len(b.segs) == 0)
    b.events_by_seg = [[] for _ in range(len(b.segs))]
    b.event_ptr.clear()
    b.segment_enq_done.clear()
    b.enqueued_keys.clear()

    # Intersections updated → rebuild for all bots
    mats, hits, static_flags = rebuild_all_pair_data(bots)
    clear_all_queues()
    return mats, hits, static_flags

# =========================================================
# Stepping (per bot)
# =========================================================
def step_bot(bot: BotState, dt: float):
    if bot.done or bot.at_end():
        bot.done = True; return

    i = bot.seg_idx
    L = bot.lens[i]
    if L < 1e-9:
        bot.s_along = L
        bot.update_pos()
        bot.seg_idx += 1
        bot.done = bot.at_end()
        return

    events = bot.events_by_seg[i]  # (other_name, other_seg, t_on_me, point)
    ptr = bot.event_ptr.get(i, 0)

    # Multi-reserve: enqueue for all crossings on this segment once
    if i not in bot.segment_enq_done:
        for other_name, other_seg, _, _ in events:
            key = norm_key(bot.name, i, other_name, other_seg)
            if key not in bot.enqueued_keys:
                ensure_in_queue(key, bot.name)
                bot.enqueued_keys.add(key)
        bot.segment_enq_done.add(i)

    # If next crossing exists and I'm not head, only gate when NEAR the crossing
    gated = False
    if ptr < len(events):
        other_name, other_seg, t_cross, _ = events[ptr]
        key = norm_key(bot.name, i, other_name, other_seg)
        s_cross = t_cross * L
        dist_to_cross = s_cross - bot.s_along
        if q_head(key) != bot.name and dist_to_cross <= NEAR_PIXELS:
            stop_s = max(0.0, s_cross - APPROACH_PIXELS)  # stop APPROACH_PIXELS before crossing
            next_s = bot.s_along + bot.speed * dt
            bot.s_along = min(stop_s, next_s)            # strict clamp
            bot.update_pos()
            gated = True

    if gated:
        return

    # Move forward normally
    bot.s_along = min(L, bot.s_along + bot.speed * dt)
    bot.update_pos()

    # If head and cleared the crossing, pop & advance pointer
    ptr = bot.event_ptr.get(i, 0)
    if ptr < len(events):
        other_name, other_seg, t_cross, _ = events[ptr]
        key = norm_key(bot.name, i, other_name, other_seg)
        s_cross = t_cross * L
        # Check if bot has traveled CLEAR_MARGIN_PX past the crossing
        if q_head(key) == bot.name and bot.s_along >= (s_cross + CLEAR_MARGIN_PX):
            pop_if_head(key, bot.name)
            bot.event_ptr[i] = ptr + 1

    # End of segment cleanup
    if bot.s_along >= L - 1e-9:
        for other_name, other_seg, _, _ in events:
            key = norm_key(bot.name, i, other_name, other_seg)
            if q_head(key) == bot.name:
                pop_if_head(key, bot.name)
            bot.enqueued_keys.discard(key)
        bot.event_ptr[i] = 0
        bot.seg_idx += 1
        bot.s_along = 0.0
        bot.done = bot.at_end()
        bot.update_pos()

# =========================================================
# Init scenario
# =========================================================
seed_counter = RANDOM_SEED
bots, waypoints, segments, mats, hits, static_flags = new_scenario(seed_counter)
clear_all_queues()

paused = False

# =========================================================
# Main loop
# =========================================================
running = True
while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                # Global regenerate all 3 with new seed
                seed_counter += 1
                bots, waypoints, segments, mats, hits, static_flags = new_scenario(seed_counter)
                clear_all_queues()
                paused = False

    if not paused:
        # unbiased stepping order each frame
        for name in random.sample(["R1","R2","R3"], 3):
            step_bot(bots[name], dt)

        # On-the-fly per-bot regeneration as soon as it finishes its path
        for name in ["R1","R2","R3"]:
            if bots[name].done:
                mats, hits, static_flags = regenerate_bot(bots, name)
                # refresh dicts used by renderer
                waypoints = {n: bots[n].wps for n in bots}
                segments  = {n: bots[n].segs for n in bots}

    # ---- Render ----
    screen.fill(BG)

    # Paths & waypoints
    for name in ["R1","R2","R3"]:
        draw_path(screen, waypoints[name], R_COLORS[name])

    # Intersections
    draw_intersections(screen, hits)

    # Bots as rectangles (4-wheel bot representation)
    BOT_WIDTH = 50
    BOT_HEIGHT = 35
    WHEEL_WIDTH = 12  # Length of wheel (along the side)
    WHEEL_HEIGHT = 6  # How much wheel sticks out
    
    for name in ["R1","R2","R3"]:
        p = bots[name].pos
        bot = bots[name]
        
        # Draw main bot body (rectangle centered at position)
        rect_x = int(p[0] - BOT_WIDTH / 2)
        rect_y = int(p[1] - BOT_HEIGHT / 2)
        
        # Bot chassis - dark gray base
        pygame.draw.rect(screen, (60, 60, 60), (rect_x, rect_y, BOT_WIDTH, BOT_HEIGHT))
        
        # Inner colored body (slightly smaller)
        inner_margin = 4
        pygame.draw.rect(screen, R_COLORS[name], 
                        (rect_x + inner_margin, rect_y + inner_margin, 
                         BOT_WIDTH - 2*inner_margin, BOT_HEIGHT - 2*inner_margin))
        
        # Draw 4 wheels as black rectangles on the TOP and BOTTOM sides
        wheel_color = (40, 40, 40)
        wheel_x_offset = 8  # Distance from left/right edge to place wheels
        
        # Top side wheels
        # Top-left wheel
        pygame.draw.rect(screen, wheel_color, 
                        (rect_x + wheel_x_offset, rect_y - WHEEL_HEIGHT // 2, 
                         WHEEL_WIDTH, WHEEL_HEIGHT))
        # Top-right wheel
        pygame.draw.rect(screen, wheel_color, 
                        (rect_x + BOT_WIDTH - WHEEL_WIDTH - wheel_x_offset, rect_y - WHEEL_HEIGHT // 2, 
                         WHEEL_WIDTH, WHEEL_HEIGHT))
        # Top-right wheel
        pygame.draw.rect(screen, wheel_color, 
                        (rect_x + BOT_WIDTH - WHEEL_WIDTH - wheel_x_offset, rect_y - WHEEL_HEIGHT // 2, 
                         WHEEL_WIDTH, WHEEL_HEIGHT))
        
        # Bottom side wheels
        # Bottom-left wheel
        pygame.draw.rect(screen, wheel_color, 
                        (rect_x + wheel_x_offset, rect_y + BOT_HEIGHT - WHEEL_HEIGHT // 2, 
                         WHEEL_WIDTH, WHEEL_HEIGHT))
        # Bottom-right wheel
        pygame.draw.rect(screen, wheel_color, 
                        (rect_x + BOT_WIDTH - WHEEL_WIDTH - wheel_x_offset, rect_y + BOT_HEIGHT - WHEEL_HEIGHT // 2, 
                         WHEEL_WIDTH, WHEEL_HEIGHT))
        
        # Draw detection marker box (like AprilTag in the image)
        marker_size = 20
        marker_x = int(p[0] - marker_size / 2)
        marker_y = int(p[1] - marker_size / 2)
        
        # White border
        pygame.draw.rect(screen, (255, 255, 255), 
                        (marker_x - 2, marker_y - 2, marker_size + 4, marker_size + 4))
        # Gray marker
        pygame.draw.rect(screen, (100, 100, 100), 
                        (marker_x, marker_y, marker_size, marker_size))
        
        # Bot number in center
        font_marker = pygame.font.Font(None, 24)
        label_surf = font_marker.render(name[-1], True, (255, 255, 255))
        label_rect = label_surf.get_rect(center=(int(p[0]), int(p[1])))
        screen.blit(label_surf, label_rect)

    # Pair matrices (top-left grid)
    render_matrix(screen, "R1 vs R2", mats["12"], (10, 10))
    render_matrix(screen, "R1 vs R3", mats["13"], (10, 10 + 4*18))
    render_matrix(screen, "R2 vs R3", mats["23"], (10, 10 + 8*18))

    # Static intersection flags per pair
    y0 = 10 + 12*18
    render_flags(screen, "Intersect Flags R1 (vs R2)", static_flags["R1_vs_R2"]["R1"], (10, y0))
    render_flags(screen, "Intersect Flags R2 (vs R1)", static_flags["R1_vs_R2"]["R2"], (10, y0+18))
    render_flags(screen, "Intersect Flags R1 (vs R3)", static_flags["R1_vs_R3"]["R1"], (10, y0+36))
    render_flags(screen, "Intersect Flags R3 (vs R1)", static_flags["R1_vs_R3"]["R3"], (10, y0+54))
    render_flags(screen, "Intersect Flags R2 (vs R3)", static_flags["R2_vs_R3"]["R2"], (10, y0+72))
    render_flags(screen, "Intersect Flags R3 (vs R2)", static_flags["R2_vs_R3"]["R3"], (10, y0+90))

    # Live traffic flags (from queue heads)
    live = derive_live_flags(bots)
    render_traffic(screen, live["R1"], live["R2"], live["R3"], (10, y0 + 120))

    help1 = font.render("Space: pause/resume  |  R: regenerate ALL  |  Esc: quit", True, (80, 80, 80))
    screen.blit(help1, (10, HEIGHT - 28))

    pygame.display.flip()

pygame.quit()
