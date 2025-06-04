#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    White,
    Black,
}

impl Into<usize> for Color {
    fn into(self) -> usize {
        match self {
            Self::White => 0,
            Self::Black => 1,
        }
    }
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

pub type Position = (u8, u8);

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
            Some(_) => {
                if dst_x.abs_diff(src_x) != 1 || (dst_y as i8 - src_y as i8) != direction {
                    return false;
                }

                return true;
            }
        }
    }

    fn bishop_legal_move(&self, src_x: u8, src_y: u8, dst_x: u8, dst_y: u8) -> bool {
        if src_x.abs_diff(dst_x) != src_y.abs_diff(dst_y) {
            return false;
        }

        let xstep: i8 = if dst_x < src_x { -1 } else { 1 };
        let ystep: i8 = if dst_y < src_y { -1 } else { 1 };

        for i in 1..src_x.abs_diff(dst_x) {
            match self.board.get_square(
                (src_x as i8 + (xstep * i as i8)) as u8,
                (src_y as i8 + (ystep * i as i8)) as u8,
            ) {
                Some(_) => return false,
                None => continue,
            }
        }

        true
    }

    fn rook_legal_move(&self, src_x: u8, src_y: u8, dst_x: u8, dst_y: u8) -> bool {
        let dist_x = src_x.abs_diff(dst_x);
        let dist_y = src_y.abs_diff(dst_y);
        let dist = dist_x + dist_y;

        if (dist_x != 0 || dist_y != dist) && (dist_y != 0 || dist_x != dist) {
            return false;
        }

        let xstep: i8 = if dist_x == 0 {
            0
        } else {
            if dst_x < src_x { -1 } else { 1 }
        };
        let ystep: i8 = if dist_y == 0 {
            0
        } else {
            if dst_y < src_y { -1 } else { 1 }
        };

        for i in 1..dist {
            match self.board.get_square(
                (src_x as i8 + (xstep * i as i8)) as u8,
                (src_y as i8 + (ystep * i as i8)) as u8,
            ) {
                Some(_) => return false,
                None => continue,
            }
        }

        true
    }

    fn queen_legal_move(&self, src_x: u8, src_y: u8, dst_x: u8, dst_y: u8) -> bool {
        self.bishop_legal_move(src_x, src_y, dst_x, dst_y)
            || self.rook_legal_move(src_x, src_y, dst_x, dst_y)
    }

    fn king_legal_move(&self, src_x: u8, src_y: u8, dst_x: u8, dst_y: u8) -> bool {
        if src_x.abs_diff(dst_x) <= 1 && src_y.abs_diff(dst_y) <= 1 {
            return true;
        }

        let square = match self.board.get_square(src_x, src_y) {
            Some(v) => v,
            None => return false,
        };

        if !self.castling_rights[square.color as usize] {
            return false;
        }

        if dst_y != src_y || src_x.abs_diff(dst_x) != 2 {
            return false;
        }

        let (direction, until) = if dst_x < src_x { (-1, 3) } else { (1, 2) };

        let rook = self
            .board
            .get_square((src_x as i8 + (until as i8 + 1) * direction) as u8, src_y);
        if Some(Piece::new(square.color, PieceType::Rook)) != rook {
            return false;
        }

        for i in 1..until + 1 {
            match self
                .board
                .get_square((src_x as i8 + i as i8 * direction) as u8, src_y)
            {
                Some(_) => {
                    return false;
                }
                None => continue,
            }
        }

        true
    }

    pub fn knight_legal_move(&self, src_x: u8, src_y: u8, dst_x: u8, dst_y: u8) -> bool {
        let dist_x = src_x.abs_diff(dst_x);
        let dist_y = src_y.abs_diff(dst_y);

        (dist_x == 2 && dist_y == 1) || (dist_x == 1 && dist_y == 2)
    }

    pub fn legal_move(&self, src_x: u8, src_y: u8, dst_x: u8, dst_y: u8) -> bool {
        if src_x >= 8 || src_y >= 8 || dst_x >= 8 || dst_y >= 8 {
            return false;
        }

        if src_x == dst_x && src_y == dst_y {
            return false;
        }

        let square = match self.board.get_square(src_x, src_y) {
            Some(v) => v,
            None => return false,
        };

        // if self.turn != square.color {
        //     return false;
        // }

        match self.board.get_square(dst_x, dst_y) {
            Some(Piece { color, .. }) => {
                if square.color == color {
                    return false;
                }
            }
            None => {}
        }

        match square.piece_type {
            PieceType::Pawn => self.pawn_legal_move(src_x, src_y, dst_x, dst_y),
            PieceType::Bishop => self.bishop_legal_move(src_x, src_y, dst_x, dst_y),
            PieceType::Rook => self.rook_legal_move(src_x, src_y, dst_x, dst_y),
            PieceType::Queen => self.queen_legal_move(src_x, src_y, dst_x, dst_y),
            PieceType::King => self.king_legal_move(src_x, src_y, dst_x, dst_y),
            PieceType::Knight => self.knight_legal_move(src_x, src_y, dst_x, dst_y),
        }
    }

    pub fn find_king(&self, color: Color) -> Option<Position> {
        for x in 0..8 {
            for y in 0..8 {
                if Some(Piece::new(color, PieceType::King)) == self.board.get_square(x, y) {
                    return Some((x, y));
                }
            }
        }

        None
    }

    pub fn is_check(&self, color: Color) -> bool {
        let (king_x, king_y) = match self.find_king(color) {
            Some(v) => v,
            None => {
                return false;
            }
        };

        for x in 0..8 {
            for y in 0..8 {
                let piece = match self.board.get_square(x, y) {
                    Some(v) => v,
                    None => continue,
                };

                if piece.color == color {
                    continue;
                }

                if self.legal_move(x, y, king_x, king_y) {
                    return true;
                }
            }
        }

        false
    }

    pub fn safe_move(&mut self, src_x: u8, src_y: u8, dst_x: u8, dst_y: u8) -> bool {
        if !self.legal_move(src_x, src_y, dst_x, dst_y) {
            return false;
        }

        let src = match self.board.get_square(src_x, src_y) {
            Some(v) => v,
            None => return false,
        };
        let dst = self.board.get_square(dst_x, dst_y);

        if PieceType::King == src.piece_type && src_x.abs_diff(dst_x) == 2 {
            let (direction, until) = if dst_x < src_x { (-1, 3) } else { (1, 2) };

            let mut is_check = false;

            for i in 0..until + 2 {
                self.board
                    .set_square((src_x as i8 + i * direction) as u8, src_y, Some(src));

                if self.is_check(src.color) {
                    is_check = true;
                    break;
                }
                self.board
                    .set_square((src_x as i8 + i * direction) as u8, src_y, None);
            }

            self.board.set_square(src_x, src_y, Some(src));

            !is_check
        } else {
            self.board.set_square(dst_x, dst_y, Some(src));
            self.board.set_square(src_x, src_y, None);

            let is_check = self.is_check(src.color);

            self.board.set_square(src_x, src_y, Some(src));
            self.board.set_square(dst_x, dst_y, dst);

            !is_check
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
        let mut game = Game::default();
        game.turn = Black;

        assert!(game.legal_move(3, 6, 3, 4));
        assert!(!game.legal_move(3, 6, 3, 3));

        let game = setup_board(&[(1, 1, White, Pawn)], White);
        assert!(!game.legal_move(1, 1, 2, 1));
        assert!(!game.legal_move(1, 1, 2, 2));
        assert!(!game.legal_move(1, 1, 0, 1));

        let game = setup_board(&[(1, 2, White, Pawn)], White);
        assert!(!game.legal_move(1, 2, 1, 1));

        let game = setup_board(&[(1, 2, White, Pawn)], White);
        assert!(!game.legal_move(1, 2, 1, 4));

        let game = setup_board(&[(1, 6, Black, Pawn)], Black);
        assert!(!game.legal_move(1, 6, 2, 5));

        let game = setup_board(&[(1, 5, Black, Pawn)], Black);
        assert!(!game.legal_move(1, 5, 1, 6));

        let game = setup_board(&[(1, 1, White, Pawn), (2, 2, Black, Knight)], White);
        assert!(game.legal_move(1, 1, 2, 2));

        let game = setup_board(&[(1, 6, Black, Pawn), (2, 5, White, Bishop)], Black);
        assert!(game.legal_move(1, 6, 2, 5));

        let game = setup_board(&[(1, 1, White, Pawn), (2, 2, White, Queen)], White);
        assert!(!game.legal_move(1, 1, 2, 2));

        let game = setup_board(&[(1, 1, White, Pawn), (1, 2, Black, Pawn)], White);
        assert!(!game.legal_move(1, 1, 1, 2));

        let game = setup_board(&[(1, 6, Black, Pawn), (2, 5, Black, Pawn)], Black);
        assert!(!game.legal_move(1, 6, 2, 5));

        let game = setup_board(&[(1, 1, White, Pawn), (1, 2, White, Knight)], White);
        assert!(!game.legal_move(1, 1, 1, 3));
    }

    #[test]
    fn bishop_move() {
        let game = Game::default();
        assert!(!game.legal_move(1, 7, 2, 6));

        let game = setup_board(&[(4, 5, White, Bishop)], White);
        assert!(game.legal_move(4, 5, 7, 2));
        assert!(game.legal_move(4, 5, 2, 3));

        let game = setup_board(&[(1, 1, Black, Bishop), (3, 3, White, Knight)], Black);
        assert!(game.legal_move(1, 1, 2, 2));
        assert!(game.legal_move(1, 1, 3, 3));
        assert!(!game.legal_move(1, 1, 4, 4));

        let game = setup_board(&[(1, 1, Black, Bishop), (3, 3, Black, Knight)], Black);
        assert!(game.legal_move(1, 1, 2, 2));
        assert!(!game.legal_move(1, 1, 3, 3));
        assert!(!game.legal_move(1, 1, 4, 4));
    }

    #[test]
    fn rook_move() {
        let game = setup_board(&[(3, 2, White, Rook)], White);
        assert!(game.legal_move(3, 2, 6, 2));
        assert!(game.legal_move(3, 2, 3, 4));
        assert!(game.legal_move(3, 2, 1, 2));
        assert!(game.legal_move(3, 2, 3, 1));
        assert!(!game.legal_move(3, 2, 1, 4));

        let game = setup_board(&[(4, 3, Black, Rook), (6, 3, White, Rook)], Black);
        assert!(game.legal_move(4, 3, 2, 3));
        assert!(game.legal_move(4, 3, 5, 3));
        assert!(game.legal_move(4, 3, 6, 3));
        assert!(!game.legal_move(4, 3, 7, 3));

        let game = setup_board(&[(4, 3, White, Rook), (6, 3, White, Rook)], White);
        assert!(game.legal_move(4, 3, 2, 3));
        assert!(game.legal_move(4, 3, 5, 3));
        assert!(!game.legal_move(4, 3, 6, 3));
        assert!(!game.legal_move(4, 3, 7, 3));
    }

    #[test]
    fn king_move() {
        let game = setup_board(&[(4, 7, Black, King)], Black);
        assert!(game.legal_move(4, 7, 5, 7));
        assert!(game.legal_move(4, 7, 5, 6));
        assert!(game.legal_move(4, 7, 3, 6));
        assert!(!game.legal_move(4, 7, 4, 5));

        let game = setup_board(&[(4, 7, Black, King)], Black);
        assert!(!game.legal_move(4, 7, 6, 7));
        assert!(!game.legal_move(4, 7, 2, 7));

        let game = setup_board(&[(4, 7, Black, King), (7, 7, White, Rook)], Black);
        assert!(!game.legal_move(4, 7, 6, 7));
        assert!(!game.legal_move(4, 7, 2, 7));

        let game = setup_board(&[(4, 7, Black, King), (7, 7, Black, Rook)], Black);
        assert!(game.legal_move(4, 7, 6, 7));
        assert!(!game.legal_move(4, 7, 2, 7));

        let game = setup_board(&[(4, 7, Black, King), (0, 7, Black, Rook)], Black);
        assert!(!game.legal_move(4, 7, 6, 7));
        assert!(game.legal_move(4, 7, 2, 7));

        let mut game = setup_board(
            &[
                (4, 7, Black, King),
                (0, 7, Black, Rook),
                (7, 7, Black, Rook),
                (4, 5, White, Rook),
            ],
            Black,
        );
        assert!(!game.safe_move(4, 7, 6, 7));
        assert!(!game.safe_move(4, 7, 2, 7));

        let mut game = setup_board(
            &[
                (4, 7, Black, King),
                (0, 7, Black, Rook),
                (7, 7, Black, Rook),
                (3, 5, White, Rook),
            ],
            Black,
        );
        assert!(game.safe_move(4, 7, 6, 7));
        assert!(!game.safe_move(4, 7, 2, 7));

        let mut game = setup_board(
            &[
                (4, 7, Black, King),
                (0, 7, Black, Rook),
                (7, 7, Black, Rook),
                (0, 5, White, Rook),
            ],
            Black,
        );
        assert!(game.safe_move(4, 7, 6, 7));
        assert!(!game.safe_move(4, 7, 2, 7));

        let mut game = setup_board(
            &[
                (4, 7, Black, King),
                (0, 7, Black, Rook),
                (7, 7, Black, Rook),
                (7, 5, White, Rook),
            ],
            Black,
        );
        assert!(!game.safe_move(4, 7, 6, 7));
        assert!(game.safe_move(4, 7, 2, 7));

        let mut game = setup_board(
            &[
                (4, 7, Black, King),
                (0, 7, Black, Rook),
                (7, 7, Black, Rook),
                (6, 5, White, Rook),
                (3, 5, White, Rook),
            ],
            Black,
        );
        assert!(!game.safe_move(4, 7, 6, 7));
        assert!(!game.safe_move(4, 7, 2, 7));

        let mut game = setup_board(&[(2, 2, White, King), (3, 4, Black, King)], White);
        assert!(!game.safe_move(2, 2, 2, 3));
        assert!(game.safe_move(2, 2, 1, 3));
    }

    #[test]
    fn knight_move() {
        let game = setup_board(&[(2, 5, White, Knight)], White);
        assert!(game.legal_move(2, 5, 1, 7));
        assert!(game.legal_move(2, 5, 1, 3));
        assert!(!game.legal_move(2, 5, 2, 2));

        let game = setup_board(
            &[
                (5, 5, White, Knight),
                (4, 3, White, Pawn),
                (6, 3, Black, Queen),
            ],
            White,
        );
        assert!(!game.legal_move(5, 5, 4, 3));
        assert!(game.legal_move(5, 5, 6, 3));

        let mut game = setup_board(
            &[
                (5, 5, White, Knight),
                (4, 3, White, Pawn),
                (6, 3, Black, Queen),
                (7, 7, Black, Bishop),
                (4, 4, White, King),
            ],
            White,
        );
        assert!(!game.safe_move(5, 5, 4, 3));
        assert!(!game.safe_move(5, 5, 6, 3));
    }

    fn setup_board(pieces: &[(u8, u8, Color, PieceType)], turn: Color) -> Game {
        let mut board = Board::new();
        for &(x, y, color, piece_type) in pieces {
            board.set_square(x, y, Some(Piece { color, piece_type }));
        }

        Game {
            board,
            turn,
            ..Default::default()
        }
    }
}
