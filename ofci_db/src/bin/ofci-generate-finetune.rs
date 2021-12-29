fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use ofci_db::{establish_pool, dump_tokens_for_finetune};

    let out_path = std::env::args().nth(1).expect("Specify output path");
    let group = std::env::args().nth(2);
    let pool = establish_pool()?;
    dump_tokens_for_finetune(pool, &out_path, group.as_deref())
}
