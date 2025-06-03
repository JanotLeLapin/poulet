#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    White,
    Black,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PieceType {
    Pawn,
    Bishop,
    Knight,
    Rook,
    Queen,
    King,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Piece {
    pub color: Color,
    pub piece_type: PieceType,
}

impl Piece {
    pub fn new(color: Color, piece_type: PieceType) -> Self {
        Piece { color, piece_type }
    }
}

pub type Square = Option<Piece>;

#[derive(Clone, Debug)]
pub struct Board(pub [Square; 64]);

impl Board {
    pub fn new() -> Self {
        let squares: [Square; 64] = [None; 64];
        Self(squares)
    }

    pub fn init(&mut self) {
        for i in 0..8 {
            let piece_type = match i {
                0 | 7 => PieceType::Rook,
                1 | 6 => PieceType::Knight,
                2 | 5 => PieceType::Bishop,
                3 => PieceType::Queen,
                4 => PieceType::King,
                _ => {
                    panic!();
                }
            };

            self.set_square(i, 0, Some(Piece::new(Color::White, piece_type)));
            self.set_square(i, 1, Some(Piece::new(Color::White, PieceType::Pawn)));

            self.set_square(i, 7, Some(Piece::new(Color::Black, piece_type)));
            self.set_square(i, 6, Some(Piece::new(Color::Black, PieceType::Pawn)));
        }

        for i in 0..8 {
            for j in 2..6 {
                self.set_square(i, j, None);
            }
        }
    }

    pub fn get_square(&self, x: u8, y: u8) -> Square {
        self.0[(y * 8 + x) as usize]
    }

    pub fn set_square(&mut self, x: u8, y: u8, square: Square) {
        self.0[(y * 8 + x) as usize] = square;
    }
}
