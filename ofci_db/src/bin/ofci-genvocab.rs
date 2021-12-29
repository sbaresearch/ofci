fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use ofci_db::{establish_connection, fetch_contents, train_tokenizer};

    let out_path = std::env::args().nth(1).expect("Specify output path");
    let prefix = std::env::args().nth(2).expect("Specify prefix");
    let conn = establish_connection()?;
    let contents = fetch_contents(&conn)?;
    train_tokenizer(&out_path, &prefix, contents)
}
