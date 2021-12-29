#[macro_use]
extern crate diesel;

use diesel::prelude::*;
use rayon::prelude::*;
use diesel::pg::PgConnection;
use diesel::r2d2::{ConnectionManager, Pool};
use dotenv::dotenv;
use std::env;

pub mod models;
pub mod schema;

const PADDING_TOKEN: i16 = 1;
const MAX_TOKENS: i32 = 512;
const FINETUNE_FUNCTION_COUNT: usize = 500_000;
const FALSE_PAIR_NUM: usize = 2;
const DECAY_NUM: f64 = 0.9999;

#[derive(Copy, Clone, Debug)]
enum ObfuscationLevels {
    O0     = 0,
    O1     = 1,
    O2     = 2,
    O3     = 3,
    Bcf    = 4,
    Cff    = 5,
    Ibr    = 6,
    Spl    = 7,
    Sub    = 8,
    Virt   = 9,
    VirtEa = 10,
    Ea     = 11
}

#[derive(Hash, Eq, PartialEq)]
struct FunctionPairStub {
    fn1: i32,
    fn2: i32
}

#[derive(Hash, Eq, PartialEq, Clone)]
struct FunctionMeta {
    tool: i32,
    name: i32
}

struct FunctionCategoryData {
    category: ObfuscationLevels,
    fn_id: i32,
    fragment_count: u16
}

#[derive(Clone)]
enum TraceMatch {
    NoMatch,
    First,
    Second
}

#[derive(Clone)]
struct FunctionPair {
    fn1: i32,
    fn2: i32,
    label: i8,
    fragment_count: u16,
    trace_match: TraceMatch
}

fn db_to_obf(db_id: i32) -> ObfuscationLevels {
    match db_id {
        0  => ObfuscationLevels::O0,
        1  => ObfuscationLevels::O1,
        2  => ObfuscationLevels::O2,
        3  => ObfuscationLevels::O3,
        4  => ObfuscationLevels::Bcf,
        5  => ObfuscationLevels::Cff,
        6  => ObfuscationLevels::Ibr,
        7  => ObfuscationLevels::Spl,
        8  => ObfuscationLevels::Sub,
        9  => ObfuscationLevels::Virt,
        10 => ObfuscationLevels::VirtEa,
        11 => ObfuscationLevels::Ea,
        _  => ObfuscationLevels::O0, 
    }
}

pub fn establish_connection(
) -> Result<PgConnection, Box<dyn std::error::Error + Send + Sync>> {
    dotenv()?;

    let database_url = env::var("DATABASE_URL")?;
    PgConnection::establish(&database_url)
        .map_err(|e| e.into())
}

pub fn establish_pool(
) -> Result<
    Pool<ConnectionManager<PgConnection>>,
    Box<dyn std::error::Error + Send + Sync>
> {
    dotenv()?;

    let database_url = env::var("DATABASE_URL")?;
    let manager = ConnectionManager::<PgConnection>::new(database_url);
    Pool::builder().build(manager).map_err(|e| e.into())
}

pub fn fetch_contents(
    conn: &PgConnection
) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
    use schema::functions::dsl::*;
    functions
        .select(schema::functions::contents)
        .load(conn)
        .map_err(|e| e.into())
}

pub fn fetch_tokens_for_fn(
    conn: &PgConnection,
    fn_id: i32
) -> Result<
    String,
    Box<dyn std::error::Error + Send + Sync>
> {
    use schema::functions::dsl::*;
    let tok = functions
        .select(schema::functions::tokens)
        .find(fn_id)
        .first::<Option<String>>(conn)?;

    Ok(tok.unwrap())
}


pub fn fetch_tokens(
    conn: &PgConnection,
    exclude_group: Option<&str>
) -> Result<
    Vec<(i32, String)>,
    Box<dyn std::error::Error + Send + Sync>
> {
    use schema::functions::dsl::*;

    use indicatif::{ProgressBar, ParallelProgressIterator};
    use indicatif::ProgressStyle;


    let q = functions
        .select((
            schema::functions::token_count,
            schema::functions::tokens));
    let tok = if let Some(x) = exclude_group {
        let group = schema::group_names::dsl::group_names
            .select(schema::group_names::id)
            .filter(schema::group_names::name.eq(x))
            .first::<i32>(conn)?;

        q.filter(schema::functions::project_group.ne(group))
            .load::<(Option<i32>, Option<String>)>(conn)
    } else {
        q.load::<(Option<i32>, Option<String>)>(conn)
    }?;

    let len = tok.len() as u64;
    println!("Number of functions: {}", len);

    let progress = ProgressBar::new(len);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {msg:<40!} {wide_bar} {percent:>18!}%"),
    );
    progress
        .set_message(&format!("Unwrapping tokenizations ({:.2} Mo)", len / 1_000_000));
    progress.set_draw_delta(len / 100); // Redraw only every 2%

    let tok = tok
        .into_par_iter()
        .progress_with(progress)
        .map(|(i, s)| (i.unwrap(), s.unwrap()))
        .collect();

    Ok(tok)
}

pub fn fetch_tokens_filtered(
    conn: &PgConnection,
    exclude_group: Option<&str>
) -> Result<
    Vec<(i32, String)>,
    Box<dyn std::error::Error + Send + Sync>
> {
    use schema::functions::dsl::*;

    use indicatif::{ProgressBar, ParallelProgressIterator};
    use indicatif::ProgressStyle;


    let q = functions
        .select((
            schema::functions::token_count,
            schema::functions::tokens))
        // Blacklisted function names
        .filter(schema::functions::name.ne(22105)) // _init
        .filter(schema::functions::name.ne(85)) // HikariFunctionWrapper
        .filter(schema::functions::name.ne(22221)) // frame_dummy
        .filter(schema::functions::name.ne(22220)) // __do_global_dtors_aux
        .filter(schema::functions::name.ne(22219)) // register_tm_clones
        .filter(schema::functions::name.ne(22218)) // deregister_tm_clones
        .filter(schema::functions::name.ne(27057)) // __libc_csu_fini
        .filter(schema::functions::name.ne(22216)) // _start
        .filter(schema::functions::name.ne(27055)) // __libc_csu_init
        .filter(schema::functions::token_count.gt(59));

    let tok = if let Some(x) = exclude_group {
        let group = schema::group_names::dsl::group_names
            .select(schema::group_names::id)
            .filter(schema::group_names::name.eq(x))
            .first::<i32>(conn)?;

        q.filter(schema::functions::project_group.ne(group))
            .load::<(Option<i32>, Option<String>)>(conn)
    } else {
        q.load::<(Option<i32>, Option<String>)>(conn)
    }?;

    let len = tok.len() as u64;
    println!("Number of functions: {}", len);

    let progress = ProgressBar::new(len);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {msg:<40!} {wide_bar} {percent:>18!}%"),
    );
    progress
        .set_message(&format!("Unwrapping tokenizations ({:.2} Mo)", len / 1_000_000));
    progress.set_draw_delta(len / 100); // Redraw only every 2%

    let tok = tok
        .into_par_iter()
        .progress_with(progress)
        .map(|(i, s)| (i.unwrap(), s.unwrap()))
        .collect();

    Ok(tok)
}

fn fetch_function_meta(
    conn: &PgConnection,
    group: &Option<&str>,
    validation: &Option<&str>,
    output_location: &str,
) -> Result<
    Vec<Vec<FunctionCategoryData>>,
    Box<dyn std::error::Error + Send + Sync>
> {
    use std::collections::HashMap;
    use std::io::{Write, BufRead};
    use rand::seq::SliceRandom;

    let base_query = schema::functions::dsl::functions
        .select((
            schema::functions::id,
            schema::functions::name,
            schema::functions::tool,
            schema::functions::category,
            schema::functions::token_count
        ))
        // Blacklisted function names
        .filter(schema::functions::name.ne(1)) // _init
        .filter(schema::functions::name.ne(16608)) // HikariFunctionWrapper
        .filter(schema::functions::name.ne(1040)) // frame_dummy
        .filter(schema::functions::name.ne(1039)) // __do_global_dtors_aux
        .filter(schema::functions::name.ne(1038)) // register_tm_clones
        .filter(schema::functions::name.ne(1037)) // deregister_tm_clones
        .filter(schema::functions::name.ne(7691)) // __libc_csu_fini
        .filter(schema::functions::name.ne(5596)) // _start
        .filter(schema::functions::name.ne(7690)); // __libc_csu_init

    let function_meta = match group {
        Some(name_id) => {
            let g = schema::group_names::dsl::group_names
                .select(schema::group_names::id)
                .filter(schema::group_names::name.eq(name_id))
                .first::<i32>(conn)?;

            base_query
                .filter(schema::functions::project_group.eq(g))
                .load::<(i32, i32, i32, i32, Option<i32>)>(conn)?
        },
        None => {
            base_query
                .load::<(i32, i32, i32, i32, Option<i32>)>(conn)?
        }
    };

    let mut funcmap = HashMap::new();
    for (fn_id, name, tool, category, token_count) in function_meta.into_iter() {
        let token_count = token_count.unwrap();
        let mut fragment_count = (token_count / MAX_TOKENS) as u16;
        fragment_count += (token_count % MAX_TOKENS > 0) as u16;
        let category = db_to_obf(category);

        let meta_tag = FunctionMeta { name, tool };
        let cat_info = FunctionCategoryData { fn_id, category, fragment_count };
        
        let func_entry = funcmap.entry(meta_tag).or_insert(Vec::new());
        func_entry.push(cat_info);
    }

    println!("Number of unique functions: {}", funcmap.len());
    let training_size = (funcmap.len() as f32 * 0.3).ceil() as usize;

    let training_funcs = if let Some(f) = validation {
        let mut path = std::path::PathBuf::new();
        path.push(output_location);
        path.push(f);
        path.set_extension("txt");

        let input = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(input);
        for line in reader.lines() {
            let l = line?;
            let mut s = l.split("-");
            let name: i32 = s.next().unwrap().parse()?;
            let tool: i32 = s.next().unwrap().parse()?;
            funcmap.remove(&FunctionMeta { name, tool });
        }

        println!("Number of evaluation functions: {}", funcmap.len());
        funcmap.into_values().into_iter().collect()
    } else {
        println!("Number of training functions: {}", training_size);
        let meta_tags = funcmap.keys().cloned().collect::<Vec<_>>();
        let rng = &mut rand::thread_rng();
        let mut training_funcs = Vec::with_capacity(training_size);

        let mut path = std::path::PathBuf::new();
        path.push(output_location);
        match group {
            None => path.push("training-funcs"),
            Some(s) => path.push(&format!("training-funcs-{}", s))
        }
        path.set_extension("txt");
        let output = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(output);

        // Write chosen funcs to file and keep them in a separate vec
        for meta_tag in meta_tags.choose_multiple(rng, training_size) {
            // Unwrap here can't fail because we're just working with
            // tags that are in the map anyways
            training_funcs.push(funcmap.remove(&meta_tag).unwrap());
            writeln!(writer, "{}-{}", meta_tag.name, meta_tag.tool)?;
        }

        training_funcs
    };
    
    Ok(training_funcs)
}

pub fn train_tokenizer(
    output_location: &str,
    prefix: &str,
    function_contents: Vec<String>
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
    use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::{AddedToken, TokenizerBuilder, Model};

    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(1000)
        .min_frequency(0)
        .special_tokens(vec![
            AddedToken::from("<s>".to_owned(), true),
            AddedToken::from("<pad>".to_owned(), true),
            AddedToken::from("</s>".to_owned(), true),
            AddedToken::from("<unk>".to_owned(), true),
            AddedToken::from("<mask>".to_owned(), true),
        ])
        .build();

    let mut byte_level = ByteLevel::default();
    byte_level.add_prefix_space = false;
    byte_level.trim_offsets = false;

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into()
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()?;

    tokenizer
        .train(&mut trainer, function_contents.into_iter())?
        .get_model()
        .save(&std::path::Path::new(output_location), Some(prefix))?;

    Ok(())
}

pub fn tokenize_db(
    pool: Pool<ConnectionManager<PgConnection>>,
    vocabulary: &str,
    merges: &str
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use tokenizers::{TokenizerBuilder, Model};
    use tokenizers::models::bpe::BPE;
    use tokenizers::processors::roberta::RobertaProcessing;
    use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use indicatif::{ProgressBar, ParallelProgressIterator};
    use indicatif::ProgressStyle;
    use schema::functions::dsl::*;

    let bpe_builder = BPE::from_file(vocabulary, merges);
    let bpe = bpe_builder
        .unk_token("<unk>".to_owned())
        .build()?;

    let mut byte_level = ByteLevel::default();
    byte_level.add_prefix_space = false;
    byte_level.trim_offsets = false;

    let post_processor = RobertaProcessing::new(
        ("</s>".to_owned(), bpe.token_to_id("</s>").unwrap()),
        ("<s>".to_owned(), bpe.token_to_id("<s>").unwrap()))
        .add_prefix_space(false)
        .trim_offsets(false);

    let tokenizer = TokenizerBuilder::new()
        .with_model(bpe)
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into()
        ])))
        .with_pre_tokenizer(Some(byte_level))
        .with_post_processor(Some(post_processor))
        .with_decoder(Some(byte_level))
        .build()?;

    // Load all functions from the DB (there really needs to be an iter..)
    let conn = pool.get()?;
    let funcs = functions
        .select((schema::functions::id, schema::functions::contents))
        .load::<(i32, String)>(&conn)?;

    let len = funcs.len() as u64;
    let progress = ProgressBar::new(len);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {msg:<40!} {wide_bar} {percent:>18!}%"),
    );
    progress
        .set_message(&format!("Tokenizing functions ({:.2} Mo)", len / 1_000_000));
    progress.set_draw_delta(len / 100); // Redraw only every 2%

    funcs
        .into_par_iter()
        .progress_with(progress)
        .for_each(|(func_id, func_content)| {
            let p = pool.clone();
            let conn = p.get().unwrap();

            let enc = tokenizer.encode(func_content, true).unwrap();
            let tokenized = serde_json::to_string(enc.get_ids()).unwrap();
            diesel::update(functions.find(func_id))
                .set((
                    schema::functions::tokens.eq(&tokenized),
                    schema::functions::token_count.eq(enc.len() as i32)))
                .execute(&conn).unwrap();
        });

    Ok(())
}

pub fn dump_tokens_for_pretrain(
    output_location: &str,
    function_tokens: Vec<(i32, String)>
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use indicatif::{ProgressBar, ParallelProgressIterator};
    use indicatif::ProgressStyle;
    use ndarray::prelude::*;

    // Find number of tokens so that we can preallocate an ndarray
    let mut token_count = 0;
    let mut offsets = Vec::with_capacity(function_tokens.len());
    for (count, _) in function_tokens.iter() {
        offsets.push(token_count);
        token_count += (count / MAX_TOKENS) as usize;
        token_count += (count % MAX_TOKENS > 59) as usize;
    }
    println!("token_count: {}", token_count);

    // Allocate the whole array upfront, hoo boy
    let mut data_array = Array::from_elem(
        (token_count, MAX_TOKENS as usize),
        PADDING_TOKEN);

    let len = function_tokens.len() as u64;
    let progress = ProgressBar::new(len);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {msg:<40!} {wide_bar} {percent:>18!}%"),
    );
    progress
        .set_message(&format!("Deserializing tokens ({:.2} Mo)", len / 1_000_000));
    progress.set_draw_delta(len / 100); // Redraw only every 2%

    // We know that the array accesses are disjoint, but Rust
    // can't map this into a safe way atm
    let data_array_ptr: usize = data_array.as_mut_ptr() as _;
    function_tokens
        .into_par_iter()
        .zip_eq(offsets.into_par_iter())
        .progress_with(progress)
        .for_each(|((_, s), offset)| {
            let toks: Vec<i16> = serde_json::from_str(&s).unwrap();
            Array::from(toks)
                .axis_chunks_iter(Axis(0), MAX_TOKENS as usize)
                .enumerate()
                .for_each(|(i, c)| if c.len() > 59 {
                    let shape = (token_count, MAX_TOKENS as usize);

                    // DANGER ZONE
                    let mut da_view = unsafe {
                        let ptr = data_array_ptr as *mut i16;
                        ArrayViewMut::from_shape_ptr(shape, ptr)
                    };

                    // I really fucking hope this works
                    da_view
                        .slice_mut(s![offset + i, ..c.len()])
                        .assign(&c);
                });
        });

    let mut path = std::path::PathBuf::new();
    path.push(output_location);
    path.push("pretrain");
    path.set_extension("h5");

    // Don't split training/validation, it's easier for us
    // to handle this on the python side of things (array
    // shuffling is a bit of a pain in Rust atm)
    hdf5::File::create(&path)?
        .new_dataset::<i16>()
        .create("data", data_array.shape())?
        .write(&data_array)?;

    Ok(())
}

fn find_smallest_obf(
    rng: &mut rand::rngs::ThreadRng,
    obf_counters: &[f64; 12]
) -> (usize, usize) {
    use rand::seq::SliceRandom;
    let obf_indices: [usize; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    
    let obf1 = obf_indices.choose_weighted(rng, |i| obf_counters[*i]).unwrap();
    let obf2 = obf_indices.choose_weighted(rng, |i| obf_counters[*i]).unwrap();

    (*obf1, *obf2)
}

fn find_random_func(
    rng: &mut rand::rngs::ThreadRng,
    func_meta: &mut Vec<Vec<FunctionCategoryData>>,
    obf: usize,
) -> Option<(i32, u16)> {
    use rand::seq::SliceRandom;

    let func = match func_meta.choose(rng) {
        Some(f) => f,
        None => return None
    };

    let flen = func.len();
    for e in func.choose_multiple(rng, flen) {
        if e.category as usize == obf {
            return Some((e.fn_id, e.fragment_count));
        }
    }

    None
}

fn find_matching_pair(
    rng: &mut rand::rngs::ThreadRng,
    func_meta: &mut Vec<Vec<FunctionCategoryData>>,
    obf1: usize,
    obf2: usize
) -> Option<(FunctionPair, u16)> {
    use rand::seq::SliceRandom;

    let is_virt =
        obf1 == ObfuscationLevels::Virt as usize ||
        obf1 == ObfuscationLevels::VirtEa as usize ||
        obf2 == ObfuscationLevels::Virt as usize ||
        obf2 == ObfuscationLevels::VirtEa as usize;

    let func = match func_meta.choose(rng) {
        Some(f) => f,
        None => return None
    };

    // Quit early if no pair can be established
    let flen = func.len();
    if flen < 2 {
        return None;
    }

    let mut obf1_candidate = None;
    for e in func.choose_multiple(rng, flen) {
        if e.category as usize == obf1 {
            obf1_candidate = Some((e.fn_id, e.fragment_count));
            break;
        }
    }

    let (fn_id1, fcount1) = match obf1_candidate {
        None => return None,
        Some(info) => info
    };

    let mut obf2_candidate = None;
    for e in func.choose_multiple(rng, flen) {
        if e.category as usize == obf2 && e.fn_id != fn_id1 {
            obf2_candidate = Some((e.fn_id, e.fragment_count));
            break;
        }
    }

    let (fn_id2, fcount2) = match obf2_candidate {
        None => return None,
        Some(info) => info
    };

    // Check if we have a trace match
    let fmin = fcount1.min(fcount2);
    let (fragment_count, trace_match) = if is_virt && fmin == 1 {
        (fcount1.max(fcount2), if fcount1 > fcount2 {
            TraceMatch::Second
        } else {
            TraceMatch::First
        })
    } else {
        (fmin, TraceMatch::NoMatch)
    };

    Some((
        FunctionPair {
            fn1: fn_id1,
            fn2: fn_id2,
            fragment_count,
            trace_match,
            label: 1
        },
        fcount1
    ))
}

pub fn dump_tokens_for_finetune(
    pool: Pool<ConnectionManager<PgConnection>>,
    output_location: &str,
    group: Option<&str>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::collections::HashSet;
    use rand::thread_rng;
    use indicatif::{ProgressBar, ParallelProgressIterator};
    use indicatif::ProgressStyle;
    use ndarray::prelude::*;

    // Get a dump of all the functions
    println!("Fetching function meta...");
    let conn = pool.get()?;
    let mut func_meta = fetch_function_meta(&conn, &group, &None, output_location)?;

    // Calculate statistics
    let mut full_count: usize = 0;
    let mut base_counters = [0; 12];
    let mut obf_counters = [DECAY_NUM; 12];
    for v in func_meta.iter() {
        for d in v {
            base_counters[d.category as usize] += d.fragment_count as usize;
            full_count += d.fragment_count as usize;
        }
    }
    println!("Category statistics training: {}", full_count);
    for i in 0..12 {
        println!("    {:?}: {:.2}%, {}", db_to_obf(i as i32),
            (base_counters[i] as f32 / full_count as f32) * 100.0,
            base_counters[i]);

        // If we don't have any data, we never want the
        // obfuscation category to be selected
        if base_counters[i] == 0 {
            obf_counters[i] = 0.0;
        }
    }
    base_counters = [0; 12];

    // Create count for all obfuscation levels 
    let mut rng = thread_rng();
    let mut pairs: Vec<FunctionPair> = Vec::with_capacity(FINETUNE_FUNCTION_COUNT * 2);
    let mut pair_stubs: HashSet<FunctionPairStub> = HashSet::with_capacity(FINETUNE_FUNCTION_COUNT * 2);
    let mut fn_counter = 0;

    // Main dataset generation loop
    while fn_counter < FINETUNE_FUNCTION_COUNT {
        let (obf1, obf2) = find_smallest_obf(&mut rng, &obf_counters);
        if obf1 == obf2 {
            continue;
        }

        let (fp, fcount) = match find_matching_pair(
            &mut rng,
            &mut func_meta,
            obf1,
            obf2
        ) {
            Some(p) => p,
            None => continue
        };

        //println!("counter: {}, obf1: {}, obf2: {}", fn_counter, obf1, obf2);

        // Check if pair is already selected
        if pair_stubs.contains(&FunctionPairStub { fn1: fp.fn1, fn2: fp.fn2 }) {
            continue;
        }
        if pair_stubs.contains(&FunctionPairStub { fn1: fp.fn2, fn2: fp.fn1 }) {
            continue;
        }

        // Adjust counts according to matching strategy
        fn_counter += fp.fragment_count as usize;
        obf_counters[obf1 as usize] *= DECAY_NUM.powi(fp.fragment_count as i32);
        obf_counters[obf2 as usize] *= DECAY_NUM.powi(fp.fragment_count as i32);
        base_counters[obf1 as usize] += fp.fragment_count as usize;
        base_counters[obf2 as usize] += fp.fragment_count as usize;
        pairs.push(fp.clone());
        pair_stubs.insert(FunctionPairStub { fn1: fp.fn1, fn2: fp.fn2 });
        pair_stubs.insert(FunctionPairStub { fn1: fp.fn2, fn2: fp.fn1 });

        // Generate false pairs
        let mut false_counter = 0;
        while false_counter < FALSE_PAIR_NUM {
            let (obf, _) = find_smallest_obf(&mut rng, &obf_counters);
            let (fn2, fcount2) = match find_random_func(
                &mut rng,
                &mut func_meta,
                obf
            ) {
                Some(f) => f,
                None => continue
            };

            // Check if pair is already selected
            if pair_stubs.contains(&FunctionPairStub { fn1: fp.fn1, fn2 }) {
                continue;
            }
            if pair_stubs.contains(&FunctionPairStub { fn1: fn2, fn2: fp.fn1 }) {
                continue;
            }

            let is_virt =
                obf1 == ObfuscationLevels::Virt as usize ||
                obf1 == ObfuscationLevels::VirtEa as usize ||
                obf == ObfuscationLevels::Virt as usize ||
                obf == ObfuscationLevels::VirtEa as usize;

            let fmin = fcount.min(fcount2);
            let (fragment_count, trace_match) = if is_virt && fmin == 1 {
                (fcount.max(fcount2), if fcount > fcount2 {
                    TraceMatch::Second
                } else {
                    TraceMatch::First
                })
            } else {
                (fmin, TraceMatch::NoMatch)
            };

            let false_fp = FunctionPair {
                fn1: fp.fn1,
                fn2,
                fragment_count,
                trace_match,
                label: -1
            };

            // Adjust counts according to matching strategy
            false_counter += 1;
            fn_counter += false_fp.fragment_count as usize;
            obf_counters[obf1 as usize] *= DECAY_NUM.powi(false_fp.fragment_count as i32);
            obf_counters[obf as usize] *= DECAY_NUM.powi(false_fp.fragment_count as i32);
            base_counters[obf1 as usize] += false_fp.fragment_count as usize;
            base_counters[obf as usize] += false_fp.fragment_count as usize;
            pairs.push(false_fp.clone());
            pair_stubs.insert(FunctionPairStub { fn1: false_fp.fn1, fn2: false_fp.fn2 });
            pair_stubs.insert(FunctionPairStub { fn1: false_fp.fn2, fn2: false_fp.fn1 });
        }
    }

    // Generate token offsets for efficient assignment 
    let mut token_count = 0;
    let mut offsets = Vec::with_capacity(pairs.len());
    for fp in pairs.iter() {
        offsets.push(token_count);
        token_count += fp.fragment_count as usize;
    }

    println!("Category statistics fragments: {}", token_count);
    for i in 0..12 {
        println!("    {:?}: {:.2}%, {}, {:.3}", db_to_obf(i as i32),
            (base_counters[i] as f32 / (token_count * 2) as f32) * 100.0,
            base_counters[i], obf_counters[i]);

        // If we don't have any data, we never want the
        // obfuscation category to be selected
        if base_counters[i] == 0 {
            obf_counters[i] = 0.0;
        }
    }

    // Allocate all arrays upfront, hoo boy
    let mut fn1_array = Array::from_elem(
        (token_count, MAX_TOKENS as usize),
        PADDING_TOKEN);
    let mut fn2_array = Array::from_elem(
        (token_count, MAX_TOKENS as usize),
        PADDING_TOKEN);
    let mut label_array = Array::from_elem(
        token_count,
        0i8);

    let len = pairs.len() as u64;
    let progress = ProgressBar::new(len);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {msg:<40!} {wide_bar} {percent:>18!}%"),
    );
    progress
        .set_message(&format!("Retrieving pairs ({:.2} Th)", len / 1_000));
    progress.set_draw_delta(len / 100); // Redraw only every 2%

    // We know that the array accesses are disjoint, but Rust
    // can't map this into a safe way atm
    let fn1_array_ptr: usize = fn1_array.as_mut_ptr() as _;
    let fn2_array_ptr: usize = fn2_array.as_mut_ptr() as _;
    let label_array_ptr: usize = label_array.as_mut_ptr() as _;
    pairs 
        .into_par_iter()
        .zip_eq(offsets.into_par_iter())
        .progress_with(progress)
        .for_each(|(fp, offset)| {
            let conn = pool.get().unwrap();

            // Assign labels (i'm way too tired to write unsafe code atm..)
            {
                let label_shape = token_count;

                // DANGER ZONE
                let mut label_view = unsafe {
                    let ptr = label_array_ptr as *mut i8;
                    ArrayViewMut::from_shape_ptr(label_shape, ptr)
                };

                label_view
                    .slice_mut(s![offset..offset + fp.fragment_count as usize])
                    .fill(fp.label);
            }

            // Assign tokens for function 1
            if let TraceMatch::First = fp.trace_match {
                let s = fetch_tokens_for_fn(&conn, fp.fn1).unwrap();
                let toks: Vec<i16> = serde_json::from_str(&s).unwrap();
                let c = Array::from(toks);

                for i in 0..fp.fragment_count as usize {
                    let shape = (token_count, MAX_TOKENS as usize);

                    // DANGER ZONE
                    let mut da_view = unsafe {
                        let ptr = fn1_array_ptr as *mut i16;
                        ArrayViewMut::from_shape_ptr(shape, ptr)
                    };

                    // I really fucking hope this works
                    da_view
                        .slice_mut(s![offset + i, ..c.len()])
                        .assign(&c);
                }
            } else {
                let s = fetch_tokens_for_fn(&conn, fp.fn1).unwrap();
                let toks: Vec<i16> = serde_json::from_str(&s).unwrap();

                // Update according to matching strategy
                Array::from(toks)
                    .axis_chunks_iter(Axis(0), MAX_TOKENS as usize)
                    .enumerate()
                    .take(fp.fragment_count as usize)
                    .for_each(|(i, c)| {
                        let shape = (token_count, MAX_TOKENS as usize);

                        // DANGER ZONE
                        let mut da_view = unsafe {
                            let ptr = fn1_array_ptr as *mut i16;
                            ArrayViewMut::from_shape_ptr(shape, ptr)
                        };

                        // I really fucking hope this works
                        da_view
                            .slice_mut(s![offset + i, ..c.len()])
                            .assign(&c);
                    });
            }

            // Assign tokens for function 2
            if let TraceMatch::Second = fp.trace_match {
                let s = fetch_tokens_for_fn(&conn, fp.fn2).unwrap();
                let toks: Vec<i16> = serde_json::from_str(&s).unwrap();
                let c = Array::from(toks);

                for i in 0..fp.fragment_count as usize {
                    let shape = (token_count, MAX_TOKENS as usize);

                    // DANGER ZONE
                    let mut da_view = unsafe {
                        let ptr = fn2_array_ptr as *mut i16;
                        ArrayViewMut::from_shape_ptr(shape, ptr)
                    };

                    // I really fucking hope this works
                    da_view
                        .slice_mut(s![offset + i, ..c.len()])
                        .assign(&c);
                }
            } else {
                let s = fetch_tokens_for_fn(&conn, fp.fn2).unwrap();
                let toks: Vec<i16> = serde_json::from_str(&s).unwrap();

                // Update according to matching strategy
                Array::from(toks)
                    .axis_chunks_iter(Axis(0), MAX_TOKENS as usize)
                    .enumerate()
                    .take(fp.fragment_count as usize)
                    .for_each(|(i, c)| {
                        let shape = (token_count, MAX_TOKENS as usize);

                        // DANGER ZONE
                        let mut da_view = unsafe {
                            let ptr = fn2_array_ptr as *mut i16;
                            ArrayViewMut::from_shape_ptr(shape, ptr)
                        };

                        // I really fucking hope this works
                        da_view
                            .slice_mut(s![offset + i, ..c.len()])
                            .assign(&c);
                    });
            }
        });


    let mut path = std::path::PathBuf::new();
    path.push(output_location);
    match group {
        None => path.push("finetune"),
        Some(s) => path.push(&format!("finetune-{}", s))
    }
    path.set_extension("h5");

    // Don't split training/validation, it's easier for us
    // to handle this on the python side of things (array
    // shuffling is a bit of a pain in Rust atm)
    let f = hdf5::File::create(&path)?;
    f.new_dataset::<i16>()
        .create("fn1", fn1_array.shape())?
        .write(&fn1_array)?;
    f.new_dataset::<i16>()
        .create("fn2", fn2_array.shape())?
        .write(&fn2_array)?;
    f.new_dataset::<i8>()
        .create("labels", label_array.shape())?
        .write(&label_array)?;

    Ok(())
}

pub fn dump_tokens_for_inference(
    pool: Pool<ConnectionManager<PgConnection>>,
    output_location: &str,
    group: Option<&str>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use indicatif::{ProgressBar, ParallelProgressIterator};
    use indicatif::ProgressStyle;
    use ndarray::prelude::*;

    // Get a dump of all the functions
    println!("Fetching function meta...");
    let conn = pool.get()?;
    let func_meta: Vec<_> = fetch_function_meta(
        &conn,
        &group,
        &Some("training-funcs"),
        output_location
    )?.into_iter().flatten().collect();

    // Calculate statistics and find number of tokens so that we can
    // preallocate an ndarray
    let mut token_count: usize = 0;
    let mut base_counters = [0; 12];
    let mut offsets = Vec::with_capacity(func_meta.len());
    for d in func_meta.iter() {
        offsets.push(token_count);
        token_count += d.fragment_count as usize;
        base_counters[d.category as usize] += d.fragment_count as usize;
    }

    // Print statistics
    println!("Category statistics inference: {}", token_count);
    for i in 0..12 {
        println!("    {:?}: {:.2}%, {}", db_to_obf(i as i32),
            (base_counters[i] as f32 / token_count as f32) * 100.0,
            base_counters[i]);
    }

    // Allocate the whole array upfront, hoo boy
    let mut data_array = Array::from_elem(
        (token_count, MAX_TOKENS as usize),
        PADDING_TOKEN);
    let mut id_array = Array::from_elem(
        token_count,
        0i32);

    // Progress bar
    let len = func_meta.len() as u64;
    let progress = ProgressBar::new(len);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {msg:<40!} {wide_bar} {percent:>18!}%"),
    );
    progress
        .set_message(&format!("Retrieving pairs ({:.2} Th)", len / 1_000));
    progress.set_draw_delta(len / 100); // Redraw only every 2%

    // We know that the array accesses are disjoint, but Rust
    // can't map this in a safe way atm
    let data_array_ptr: usize = data_array.as_mut_ptr() as _;
    let id_array_ptr: usize = id_array.as_mut_ptr() as _;
    func_meta 
        .into_par_iter()
        .zip_eq(offsets.into_par_iter())
        .progress_with(progress)
        .for_each(|(FunctionCategoryData {
            fn_id,
            fragment_count,
            category: _
        }, offset)| {
            let conn = pool.get().unwrap();

            // Assign fn_id of the function to the output array
            {
                let id_shape = token_count;

                // DANGER ZONE
                let mut id_view = unsafe {
                    let ptr = id_array_ptr as *mut i32;
                    ArrayViewMut::from_shape_ptr(id_shape, ptr)
                };

                id_view
                    .slice_mut(s![offset..offset + fragment_count as usize])
                    .fill(fn_id);
            }

            // Assign tokens of the function to the output array
            {
                let s = fetch_tokens_for_fn(&conn, fn_id).unwrap();
                let toks: Vec<i16> = serde_json::from_str(&s).unwrap();

                Array::from(toks)
                    .axis_chunks_iter(Axis(0), MAX_TOKENS as usize)
                    .enumerate()
                    .take(fragment_count as usize)
                    .for_each(|(i, c)| {
                        let shape = (token_count, MAX_TOKENS as usize);

                        // DANGER ZONE
                        let mut da_view = unsafe {
                            let ptr = data_array_ptr as *mut i16;
                            ArrayViewMut::from_shape_ptr(shape, ptr)
                        };

                        // I really fucking hope this works
                        da_view
                            .slice_mut(s![offset + i, ..c.len()])
                            .assign(&c);
                    });
            }
        });

    let mut path = std::path::PathBuf::new();
    path.push(output_location);
    match group {
        None => path.push("inference-fragments"),
        Some(s) => path.push(&format!("inference-fragments-{}", s))
    }
    path.set_extension("h5");

    // Don't split training/validation, it's easier for us
    // to handle this on the python side of things (array
    // shuffling is a bit of a pain in Rust atm)
    let f = hdf5::File::create(&path)?;
    f.new_dataset::<i16>()
        .create("fragments", data_array.shape())?
        .write(&data_array)?;
    f.new_dataset::<i32>()
        .create("ids", id_array.shape())?
        .write(&id_array)?;

    Ok(())
}
