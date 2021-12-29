fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use ofci_db::{establish_connection, fetch_tokens_filtered, dump_tokens_for_pretrain};

    let out_path = std::env::args().nth(1).expect("Specify output path");
    let exclude = std::env::args().nth(2);
    let conn = establish_connection()?;
    let tokens = fetch_tokens_filtered(&conn, exclude.as_deref())?;
    dump_tokens_for_pretrain(&out_path, tokens)
}
