use std::collections::HashMap;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawBoardState {
    pub tiles: Option<Vec<Hex>>,
    pub marbles: Option<HashMap<String, Marble>>,
}

#[derive(Debug, Clone)]
pub struct BoardState {
    pub tiles: Vec<Hex>,
    pub marbles: HashMap<Hex, Marble>,
}

fn main() -> Result<()> {
    // create_board()?;
    let board = load_board()?;

    println!("tiles: {}", board.tiles.len());
    println!("marbles: {}", board.marbles.len());
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
    let text = fs::read_to_string("board.json")?;

    let raw: RawBoardState = serde_json::from_str(&text)?;

    let tiles = raw.tiles.unwrap_or_default();

    let mut marbles_hex: HashMap<Hex, Marble> = HashMap::new();
    if let Some(marbles) = raw.marbles {
        for (k, v) in marbles {
            let h = parse_hex_key(&k).with_context(|| "invalid hex key".to_string())?;
            marbles_hex.insert(h, v);
        }
    }

    // If tiles missing, infer tiles from marble keys.
    let tiles = if tiles.is_empty() && !marbles_hex.is_empty() {
        marbles_hex.keys().cloned().collect()
    } else {
        tiles
    };

    Ok(BoardState {
        tiles,
        marbles: marbles_hex,
    })
}