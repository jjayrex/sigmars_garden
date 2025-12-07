use std::collections::{HashMap, HashSet, BTreeMap, hash_map::DefaultHasher, };
use std::fs;
use std::hash::{Hash, Hasher};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use std::thread;

use anyhow::{Result, bail, anyhow, Context};
use hexx::*;
use serde::{Serialize, Deserialize};

use enigo::{Enigo, Mouse, Button, Coordinate, Settings, Direction};
use rdev::{listen, Event, EventType, Key};

#[derive(Debug)]
struct SolverStats {
    start: Instant,
    nodes: u64,
    max_depth: usize,
    best_remaining: usize,
    last_log: Instant,
    log_every_nodes: u64,
    log_every_time: Duration,
}

impl SolverStats {
    fn new(initial_remaining: usize) -> Self {
        let now = Instant::now();
        Self {
            start: now,
            nodes: 0,
            max_depth: 0,
            best_remaining: initial_remaining,
            last_log: now,
            log_every_nodes: 50_000,
            log_every_time: Duration::from_secs(1),
        }
    }

    fn on_node(
        &mut self,
        depth: usize,
        remaining: usize,
        moves_len: usize,
        seen_len: usize,
    ) {
        self.nodes += 1;
        if depth > self.max_depth {
            self.max_depth = depth;
        }
        if remaining < self.best_remaining {
            self.best_remaining = remaining;
        }

        let now = Instant::now();
        let time_due = now.duration_since(self.last_log) >= self.log_every_time;
        let node_due = self.nodes % self.log_every_nodes == 0;

        if time_due || node_due {
            let elapsed = now.duration_since(self.start);
            eprintln!(
                "[solver] nodes={} depth={} max_depth={} remaining={} best_remaining={} moves={} seen={} elapsed={:.2?}",
                self.nodes,
                depth,
                self.max_depth,
                remaining,
                self.best_remaining,
                moves_len,
                seen_len,
                elapsed
            );
            self.last_log = now;
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Marble {
    Fire,
    Water,
    Air,
    Earth,
    Salt,
    Vitae,
    Mors,
    Quicksilver,
    Lead,
    Tin,
    Iron,
    Copper,
    Silver,
    Gold,
}

impl Marble {
    fn is_cardinal(self) -> bool {
        matches!(self, Marble::Fire | Marble::Water | Marble::Air | Marble::Earth)
    }

    fn is_metal(self) -> bool {
        matches!(
            self,
            Marble::Lead | Marble::Tin | Marble::Iron | Marble::Copper | Marble::Silver | Marble::Gold
        )
    }

    fn metal_index(self) -> Option<usize> {
        match self {
            Marble::Lead => Some(0),
            Marble::Tin => Some(1),
            Marble::Iron => Some(2),
            Marble::Copper => Some(3),
            Marble::Silver => Some(4),
            Marble::Gold => Some(5),
            _ => None,
        }
    }
}

pub const METAL_ORDER: [Marble; 6] = [
    Marble::Lead,
    Marble::Tin,
    Marble::Iron,
    Marble::Copper,
    Marble::Silver,
    Marble::Gold,
];

const NEIGHBOR_OFFSETS_CW: [(i32, i32); 6] = [
    ( 1,  0),
    ( 1, -1),
    ( 0, -1),
    (-1,  0),
    (-1,  1),
    ( 0,  1),
];

#[inline]
fn neighbor_at(h: Hex, i: usize) -> Hex {
    let (dx, dy) = NEIGHBOR_OFFSETS_CW[i];
    hex(h.x + dx, h.y + dy)
}

fn affected_marbles(a: Hex, b: Hex, marbles: &HashMap<Hex, Marble>) -> Vec<Hex> {
    let mut set = HashSet::new();

    for base in [a, b] {
        for i in 0..6 {
            let n = neighbor_at(base, i);
            if marbles.contains_key(&n) {
                set.insert(n);
            }
        }
    }

    set.into_iter().collect()
}

fn local_newly_freed_gain(
    tiles_set: &HashSet<Hex>,
    before: &HashMap<Hex, Marble>,
    after: &HashMap<Hex, Marble>,
    a: Hex,
    b: Hex,
) -> usize {
    let candidates = affected_marbles(a, b, before);

    let mut gain = 0;
    for h in candidates {
        let was_free = is_marble_free(h, tiles_set, before);
        let now_free = is_marble_free(h, tiles_set, after);
        if !was_free && now_free {
            gain += 1;
        }
    }
    gain
}

#[inline]
fn is_empty_neighbor(n: Hex, tiles_set: &HashSet<Hex>, marbles: &HashMap<Hex, Marble>) -> bool {
    !tiles_set.contains(&n) || !marbles.contains_key(&n)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Deserialize)]
struct ScreenPoint {
    x: i32,
    y: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawBoardState {
    tiles: Option<Vec<Hex>>,
    marbles: Option<HashMap<String, Marble>>,
}

#[derive(Debug, Clone)]
struct BoardState {
    tiles: Vec<Hex>,
    marbles: HashMap<Hex, Marble>,
    screen_points: HashMap<Hex, ScreenPoint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Move {
    a: Hex,
    b: Hex,
}

#[derive(Debug, Clone)]
struct SearchState {
    marbles: HashMap<Hex, Marble>,
    next_metal_index: usize, // 0..=6
}

fn main() -> Result<()> {
    // create_board()?;
    let board = load_board()?;

    let solution = solve_board(&board).ok_or_else(|| anyhow!("No solution found"))?;

    println!("Found solution with {} moves.", solution.len());
    println!("Press F8 to execute...");

    wait_for_f8()?;

    execute_solution(&board.screen_points, &solution)?;

    Ok(())
}

// fn create_grid() -> Result<Vec<Hex>> {
//     let center = hex(0, 0);
//     let radius = 5;
//
//     let board = shapes::hexagon(center, radius).collect();
//
//     Ok(board)
// }
//
// fn hex_key(h: Hex) -> Result<String> {
//     Ok(format!("{},{}", h.x, h.y))
// }

fn parse_hex_key(s: &str) -> Option<Hex> {
    let (a, b) = s.split_once(',')?;
    let x = a.parse().ok()?;
    let y = b.parse().ok()?;
    Some(Hex::new(x, y))
}

fn load_click_points(path: &str) -> Result<Vec<ScreenPoint>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read click points: {path}"))?;
    let pts: Vec<ScreenPoint> = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse click points json: {path}"))?;
    Ok(pts)
}

#[inline]
fn row_len_for_r(radius: i32, r: i32) -> i32 {
    (2 * radius + 1) - r.abs()
}

#[inline]
fn q_range_for_r(radius: i32, r: i32) -> (i32, i32) {
    let q_min = (-radius).max(-r - radius);
    let q_max = (radius).min(-r + radius);
    (q_min, q_max)
}

fn build_hex_to_screen_map_for_radius(
    points: &[ScreenPoint],
    radius: i32,
) -> Result<HashMap<Hex, ScreenPoint>> {
    let mut rows: BTreeMap<i32, Vec<ScreenPoint>> = BTreeMap::new();
    for &p in points {
        rows.entry(p.y).or_default().push(p);
    }

    let expected_row_count = (2 * radius + 1) as usize;
    anyhow::ensure!(
        rows.len() == expected_row_count,
        "expected {expected_row_count} distinct y rows for radius {radius}, got {}",
        rows.len()
    );

    let mut row_vec: Vec<(i32, Vec<ScreenPoint>)> = rows
        .into_iter()
        .map(|(y, mut ps)| {
            ps.sort_by_key(|p| p.x);
            (y, ps)
        })
        .collect();

    // BTreeMap already gave ascending y
    let mut out = HashMap::new();

    for (idx, (_y, ps)) in row_vec.iter().enumerate() {
        let r = -radius + idx as i32;
        let expected_len = row_len_for_r(radius, r) as usize;

        anyhow::ensure!(
            ps.len() == expected_len,
            "row for r={r} expected length {expected_len}, got {}",
            ps.len()
        );

        let (q_min, q_max) = q_range_for_r(radius, r);
        let mut q = q_min;

        for p in ps {
            anyhow::ensure!(q <= q_max, "q overflow for r={r}");
            out.insert(hex(q, r), *p);
            q += 1;
        }
    }

    // Total tiles sanity check
    let expected_tiles = 1 + 3 * radius * (radius + 1);
    anyhow::ensure!(
        out.len() as i32 == expected_tiles,
        "expected {expected_tiles} points for radius {radius}, got {}",
        out.len()
    );

    Ok(out)
}

// fn create_board() -> Result<()> {
//     let base = create_grid()?;
//     let mut marbles = HashMap::new();
//     marbles.insert(hex_key(Hex::new(0,0))?, Marble::Fire);
//     marbles.insert(hex_key(Hex::new(0,1))?, Marble::Water);
//     marbles.insert(hex_key(Hex::new(-1,0))?, Marble::Air);
//
//     let board = RawBoardState { tiles: Some(base), marbles: Some(marbles) };
//
//     let template = serde_json::to_string_pretty(&board)?;
//
//     fs::write("template.json", template)?;
//     Ok(())
// }

fn load_board() -> Result<BoardState> {
    let text = fs::read_to_string("board.json")
        .context("failed to read board.json")?;

    let raw: RawBoardState = serde_json::from_str(&text)
        .context("failed to parse board.json")?;

    let tiles = raw.tiles.unwrap_or_default();

    let mut marbles_hex: HashMap<Hex, Marble> = HashMap::new();
    if let Some(marbles) = raw.marbles {
        for (k, v) in marbles {
            let h = parse_hex_key(&k).with_context(|| format!("invalid hex key '{k}'"))?;
            marbles_hex.insert(h, v);
        }
    }

    // If tiles missing, infer tiles from marble keys.
    let tiles = if tiles.is_empty() && !marbles_hex.is_empty() {
        marbles_hex.keys().cloned().collect()
    } else {
        tiles
    };

    // Load click points and build mapping for your board shape.
    // If you later support other shapes, see the alternative variant below.
    let click_points = load_click_points("1920x1200_formatted.json")?;
    let mut screen_points = build_hex_to_screen_map_for_radius(&click_points, 5)?;

    // Optional strictness: ensure every tile has a screen point.
    // This catches mismatched JSON early.
    let tile_set: HashSet<Hex> = tiles.iter().cloned().collect();
    screen_points.retain(|h, _| tile_set.contains(h));

    anyhow::ensure!(
        screen_points.len() == tile_set.len(),
        "screen point count ({}) does not match tile count ({})",
        screen_points.len(),
        tile_set.len()
    );

    Ok(BoardState {
        tiles,
        marbles: marbles_hex,
        screen_points,
    })
}

fn is_marble_free(h: Hex, tiles_set: &HashSet<Hex>, marbles: &HashMap<Hex, Marble>) -> bool {
    // If there is no marble here, you may want false.
    if !marbles.contains_key(&h) {
        return false;
    }

    let mut empty = [false; 6];
    for i in 0..6 {
        let n = neighbor_at(h, i);
        empty[i] = is_empty_neighbor(n, tiles_set, marbles);
    }

    // Check any 3 consecutive around the ring.
    for i in 0..6 {
        let a = empty[i];
        let b = empty[(i + 1) % 6];
        let c = empty[(i + 2) % 6];
        if a && b && c {
            return true;
        }
    }

    false
}

fn free_marbles(
    tiles_set: &HashSet<Hex>,
    marbles: &HashMap<Hex, Marble>,
) -> Vec<Hex> {
    marbles
        .keys()
        .cloned()
        .filter(|&h| is_marble_free(h, tiles_set, marbles))
        .collect()
}

fn parse_points(list: &[&str]) -> Result<Vec<ScreenPoint>> {
    let mut out = Vec::with_capacity(list.len());
    for s in list {
        let (xs, ys) = s
            .split_once(',')
            .with_context(|| format!("invalid point '{s}', expected 'x,y'"))?;
        let x: i32 = xs.trim().parse().with_context(|| format!("bad x in '{s}'"))?;
        let y: i32 = ys.trim().parse().with_context(|| format!("bad y in '{s}'"))?;
        out.push(ScreenPoint { x, y });
    }
    Ok(out)
}

fn generate_hexes_row_major(row_counts: &[i32]) -> Vec<Hex> {
    let mid = (row_counts.len() as i32 - 1) / 2;
    let mut out = Vec::new();

    for (i, &len) in row_counts.iter().enumerate() {
        let r = i as i32 - mid;
        let start_q = -((len - 1) / 2);

        for dx in 0..len {
            let q = start_q + dx;
            out.push(hex(q, r));
        }
    }

    out
}

fn build_hex_to_screen_map(
    row_counts: &[i32],
    points: &[ScreenPoint],
) -> Result<HashMap<Hex, ScreenPoint>> {
    let expected: i32 = row_counts.iter().sum();
    if points.len() as i32 != expected {
        bail!(
            "point count mismatch: got {}, expected {} from row_counts",
            points.len(),
            expected
        );
    }

    let hexes = generate_hexes_row_major(row_counts);

    if hexes.len() != points.len() {
        bail!(
            "internal mismatch: hexes {} vs points {}",
            hexes.len(),
            points.len()
        );
    }

    let map = hexes
        .into_iter()
        .zip(points.iter().copied())
        .collect::<HashMap<Hex, ScreenPoint>>();

    Ok(map)
}

fn can_match_pair(
    m1: Marble,
    m2: Marble,
    next_metal_index: usize,
    remaining_count: usize,
) -> bool {
    use Marble::*;

    // Gold-last strict rule:
    // Only allow Gold with Quicksilver when it's the final pair.
    if m1 == Gold || m2 == Gold {
        // must be exactly 2 remaining marbles
        if remaining_count != 2 {
            return false;
        }
        // and the other must be Quicksilver
        return (m1 == Gold && m2 == Quicksilver) || (m2 == Gold && m1 == Quicksilver);
    }

    // Cardinal exact match
    if m1.is_cardinal() && m2.is_cardinal() {
        return m1 == m2;
    }

    // Salt rules
    if m1 == Salt && m2 == Salt {
        return true;
    }
    if (m1 == Salt && m2.is_cardinal()) || (m2 == Salt && m1.is_cardinal()) {
        return true;
    }

    // Vitae/Mors only opposite
    if (m1 == Vitae && m2 == Mors) || (m1 == Mors && m2 == Vitae) {
        return true;
    }

    // Metals with Quicksilver in ascending order
    let metal = if m1.is_metal() && m2 == Quicksilver { Some(m1) }
    else if m2.is_metal() && m1 == Quicksilver { Some(m2) }
    else { None };

    if let Some(m) = metal {
        if let Some(idx) = m.metal_index() {
            return idx == next_metal_index;
        }
    }

    false
}

fn advance_metal_index(marbles: &HashMap<Hex, Marble>, mut idx: usize) -> usize {
    while idx < METAL_ORDER.len() {
        let target = METAL_ORDER[idx];
        let still_present = marbles.values().any(|&m| m == target);
        if still_present {
            break;
        }
        idx += 1;
    }
    idx
}

fn initial_metal_index(marbles: &HashMap<Hex, Marble>) -> usize {
    advance_metal_index(marbles, 0)
}

fn possible_moves(
    tiles_set: &HashSet<Hex>,
    state: &SearchState,
) -> Vec<Move> {
    let free = free_marbles(tiles_set, &state.marbles);
    let remaining = state.marbles.len();

    let mut moves = Vec::new();

    for i in 0..free.len() {
        let a = free[i];
        let ma = state.marbles[&a];

        for j in (i + 1)..free.len() {
            let b = free[j];
            let mb = state.marbles[&b];

            if can_match_pair(ma, mb, state.next_metal_index, remaining) {
                moves.push(Move { a, b });
            }
        }
    }

    moves
}

fn apply_move(state: &SearchState, mv: Move) -> SearchState {
    let mut marbles = state.marbles.clone();
    marbles.remove(&mv.a);
    marbles.remove(&mv.b);

    let next_metal_index = advance_metal_index(&marbles, state.next_metal_index);

    SearchState { marbles, next_metal_index }
}

fn newly_freed_count(
    tiles_set: &HashSet<Hex>,
    before: &HashMap<Hex, Marble>,
    after: &HashMap<Hex, Marble>,
) -> usize {
    let free_before: HashSet<Hex> = free_marbles(tiles_set, before).into_iter().collect();
    let free_after: HashSet<Hex> = free_marbles(tiles_set, after).into_iter().collect();

    free_after.difference(&free_before).count()
}

fn solve_board(board: &BoardState) -> Option<Vec<Move>> {
    let tiles_set: HashSet<Hex> = board.tiles.iter().cloned().collect();

    let start = SearchState {
        marbles: board.marbles.clone(),
        next_metal_index: initial_metal_index(&board.marbles),
    };

    let mut path = Vec::new();
    let mut seen: HashSet<u64> = HashSet::new();
    let mut stats = SolverStats::new(start.marbles.len());

    dfs(&tiles_set, &start, &mut path, &mut seen, &mut stats)
}

fn state_key(state: &SearchState) -> Vec<(Hex, Marble)> {
    // Stable key for visited detection.
    // For small boards this is fine.
    let mut v: Vec<_> = state.marbles.iter().map(|(h, m)| (*h, *m)).collect();
    v.sort_by_key(|(h, _)| (h.x, h.y));
    v
}

fn dfs(
    tiles_set: &HashSet<Hex>,
    state: &SearchState,
    path: &mut Vec<Move>,
    seen: &mut HashSet<u64>,
    stats: &mut SolverStats,
) -> Option<Vec<Move>> {
    if state.marbles.is_empty() {
        return Some(path.clone());
    }

    let key = hash_state(state.next_metal_index, &state.marbles);
    if !seen.insert(key) {
        return None;
    }

    let mut moves = possible_moves(tiles_set, state);
    let remaining = state.marbles.len();

    stats.on_node(path.len(), remaining, moves.len(), seen.len());

    if moves.is_empty() {
        return None;
    }

    // Order moves by "newly freed" descending
    moves.sort_by_key(|mv| {
        let next = apply_move(state, *mv);
        let gain = local_newly_freed_gain(
            tiles_set,
            &state.marbles,
            &next.marbles,
            mv.a,
            mv.b,
        );
        std::cmp::Reverse(gain)
    });

    for mv in moves {
        let next = apply_move(state, mv);
        path.push(mv);

        if let Some(sol) = dfs(tiles_set, &next, path, seen, stats) {
            return Some(sol);
        }

        path.pop();
    }

    None
}

fn wait_for_f8() -> Result<()> {
    let (tx, rx) = mpsc::channel::<()>();

    // rdev::listen blocks the current thread, so run it in a dedicated thread
    let handle = thread::spawn(move || {
        let callback = move |event: Event| {
            if let EventType::KeyPress(Key::F8) = event.event_type {
                let _ = tx.send(());
            }
        };

        // Ignore errors here; we'll surface timeout-style failure below if needed.
        let _ = listen(callback);
    });

    // Wait until we receive the signal
    rx.recv().context("failed to receive F8 signal")?;

    // We cannot reliably stop rdev listener cleanly in all setups;
    // detach thread by not joining (it will likely keep running).
    let _ = handle;

    Ok(())
}

struct Clicker {
    enigo: Enigo,
}

impl Clicker {
    fn new() -> Self {
        // Settings is required by enigo 0.2
        let enigo = Enigo::new(&Settings::default()).expect("failed to init Enigo");
        Self { enigo }
    }

    fn move_and_click(&mut self, x: i32, y: i32) {
        // Absolute screen coordinates
        self.enigo.move_mouse(x, y, Coordinate::Abs).unwrap();
        self.enigo.button(Button::Left, Direction::Click).unwrap();
    }
}

fn execute_solution(
    screen_points: &HashMap<Hex, ScreenPoint>,
    moves: &[Move],
) -> Result<()> {
    if moves.is_empty() {
        return Ok(());
    }

    let mut clicker = Clicker::new();

    // You can tune these
    let pre_click_delay = Duration::from_millis(40);
    let between_pair_clicks = Duration::from_millis(120);
    let after_pair_delay = Duration::from_millis(360);

    for mv in moves {
        let pa = screen_points.get(&mv.a)
            .with_context(|| format!("missing screen point for hex {:?}", mv.a))?;
        let pb = screen_points.get(&mv.b)
            .with_context(|| format!("missing screen point for hex {:?}", mv.b))?;

        thread::sleep(pre_click_delay);
        clicker.move_and_click(pa.x, pa.y);

        thread::sleep(between_pair_clicks);
        clicker.move_and_click(pb.x, pb.y);

        thread::sleep(after_pair_delay);
    }

    Ok(())
}

fn hash_state(next_metal_index: usize, marbles: &HashMap<Hex, Marble>) -> u64 {
    let mut v: Vec<_> = marbles.iter().collect();
    v.sort_by_key(|(h, _)| (h.x, h.y));

    let mut hasher = DefaultHasher::new();
    next_metal_index.hash(&mut hasher);
    for (h, m) in v {
        h.x.hash(&mut hasher);
        h.y.hash(&mut hasher);
        (*m as u8).hash(&mut hasher); // see note below
    }
    hasher.finish()
}