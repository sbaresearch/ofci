fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use ofci_db::{establish_pool, tokenize_db};

    let vocab = std::env::args().nth(1).expect("Specify vocab path");
    let merge = std::env::args().nth(2).expect("Specify merges path");
    let pool = establish_pool()?;
    tokenize_db(pool, &vocab, &merge)
}
