use std::collections::{HashMap, HashSet, BTreeMap};
use std::fs;

use anyhow::{Result, bail, anyhow, Context};
use hexx::*;
use serde::{Serialize, Deserialize};

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

const ROW_COUNTS: [i32; 11] = [6, 7, 8, 9, 10, 11, 10, 9, 8, 7, 6];

#[inline]
fn neighbor_at(h: Hex, i: usize) -> Hex {
    let (dx, dy) = NEIGHBOR_OFFSETS_CW[i];
    hex(h.x + dx, h.y + dy)
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

fn main() -> Result<()> {
    // create_board()?;
    let board = load_board()?;

    println!("tiles: {}", board.tiles.len());
    println!("marbles: {}", board.marbles.len());
    println!("points: {}", board.screen_points.len());
    Ok(())
}

fn create_grid() -> Result<Vec<Hex>> {
    let center = hex(0, 0);
    let radius = 5;

    let board = shapes::hexagon(center, radius).collect();

    Ok(board)
}

fn hex_key(h: Hex) -> Result<String> {
    Ok(format!("{},{}", h.x, h.y))
}

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

fn create_board() -> Result<()> {
    let base = create_grid()?;
    let mut marbles = HashMap::new();
    marbles.insert(hex_key(Hex::new(0,0))?, Marble::Fire);
    marbles.insert(hex_key(Hex::new(0,1))?, Marble::Water);
    marbles.insert(hex_key(Hex::new(-1,0))?, Marble::Air);

    let board = RawBoardState { tiles: Some(base), marbles: Some(marbles) };

    let template = serde_json::to_string_pretty(&board)?;

    fs::write("template.json", template)?;
    Ok(())
}

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

/// Collect all free marbles on the board.
fn find_free_marbles(board: &BoardState) -> Vec<Hex> {
    let tiles_set: HashSet<Hex> = board.tiles.iter().cloned().collect();

    board.marbles
        .keys()
        .cloned()
        .filter(|&h| is_marble_free(h, &tiles_set, &board.marbles))
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