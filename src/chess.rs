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

#[derive(Clone, Debug)]
pub struct Game {
    pub board: Board,
    pub total_moves: usize,
    pub until_stalemate: usize,
    pub en_passant: u8,
    pub castling_rights: [bool; 2],
    pub turn: Color,
}

impl Default for Game {
    fn default() -> Self {
        let mut board = Board::new();
        board.init();

        Self {
            board,
            total_moves: 0,
            until_stalemate: 0,
            turn: Color::White,
            en_passant: 0,
            castling_rights: [true, true],
        }
    }
}

impl Game {
    fn pawn_legal_move(&self, src_x: u8, src_y: u8, dst_x: u8, dst_y: u8) -> bool {
        let square = match self.board.get_square(src_x, src_y) {
            Some(v) => v,
            None => return false,
        };

        let (direction, start_rank) = match square.color {
            Color::White => (1, 1),
            Color::Black => (-1, 6),
        };

        match self.board.get_square(dst_x, dst_y) {
            None => {
                if src_x != dst_x {
                    return false;
                }

                if (dst_y as i8 - src_y as i8) * direction < 0 {
                    return false;
                }

                let max = if src_y == start_rank { 2 } else { 1 };
                let dist = dst_y.abs_diff(src_y);

                if dist < 1 || dist > max {
                    return false;
                }

                if dist == 2
                    && None
                        != self
                            .board
                            .get_square(src_x, (src_y as i8 + direction) as u8)
                {
                    return false;
                }

                true
            }
            Some(Piece { color, .. }) => {
                if square.color == color {
                    return false;
                }

                if dst_x.abs_diff(src_x) != 1 || (dst_y as i8 - src_y as i8) != direction {
                    return false;
                }

                return true;
            }
        }
    }

    pub fn legal_move(&self, src_x: u8, src_y: u8, dst_x: u8, dst_y: u8) -> bool {
        if src_x >= 8 || src_y >= 8 || dst_x >= 8 || dst_y >= 8 {
            return false;
        }

        let square = match self.board.get_square(src_x, src_y) {
            Some(v) => v,
            None => return false,
        };

        match square.piece_type {
            PieceType::Pawn => self.pawn_legal_move(src_x, src_y, dst_x, dst_y),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Color::*;
    use PieceType::*;

    #[test]
    fn pawn_move() {
        let game = Game::default();

        assert!(game.legal_move(3, 6, 3, 4));
        assert!(!game.legal_move(3, 6, 3, 3));

        let game = setup_board(&[(1, 1, White, Pawn)]);
        assert!(!game.legal_move(1, 1, 2, 1));
        assert!(!game.legal_move(1, 1, 2, 2));
        assert!(!game.legal_move(1, 1, 0, 1));

        let game = setup_board(&[(1, 2, White, Pawn)]);
        assert!(!game.legal_move(1, 2, 1, 1));

        let game = setup_board(&[(1, 2, White, Pawn)]);
        assert!(!game.legal_move(1, 2, 1, 4));

        let game = setup_board(&[(1, 6, Black, Pawn)]);
        assert!(!game.legal_move(1, 6, 2, 5));

        let game = setup_board(&[(1, 5, Black, Pawn)]);
        assert!(!game.legal_move(1, 5, 1, 6));

        let game = setup_board(&[(1, 1, White, Pawn), (2, 2, Black, Knight)]);
        assert!(game.legal_move(1, 1, 2, 2));

        let game = setup_board(&[(1, 6, Black, Pawn), (2, 5, White, Bishop)]);
        assert!(game.legal_move(1, 6, 2, 5));

        let game = setup_board(&[(1, 1, White, Pawn), (2, 2, White, Queen)]);
        assert!(!game.legal_move(1, 1, 2, 2));

        let game = setup_board(&[(1, 1, White, Pawn), (1, 2, Black, Pawn)]);
        assert!(!game.legal_move(1, 1, 1, 2));

        let game = setup_board(&[(1, 6, Black, Pawn), (2, 5, Black, Pawn)]);
        assert!(!game.legal_move(1, 6, 2, 5));

        let game = setup_board(&[(1, 1, White, Pawn), (1, 2, White, Knight)]);
        assert!(!game.legal_move(1, 1, 1, 3));
    }

    fn setup_board(pieces: &[(u8, u8, Color, PieceType)]) -> Game {
        let mut board = Board::new();
        for &(x, y, color, piece_type) in pieces {
            board.set_square(x, y, Some(Piece { color, piece_type }));
        }

        Game {
            board,
            ..Default::default()
        }
    }
}
