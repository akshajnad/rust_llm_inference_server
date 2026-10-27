//! Prints a GGUF file's metadata (minus large arrays) and tensor table.
//! `cargo run --release -p model --example gguf_info -- models/<file>.gguf`
use candle_core::quantized::gguf_file;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: gguf_info <file.gguf>");
    let mut f = std::fs::File::open(&path)?;
    let content = gguf_file::Content::read(&mut f)?;
    let mut keys: Vec<_> = content.metadata.keys().collect();
    keys.sort();
    for k in keys {
        let v = &content.metadata[k];
        match v.to_vec() {
            Ok(items) => println!("{k} = [{} items]", items.len()),
            Err(_) => println!("{k} = {v:?}"),
        }
    }
    let mut names: Vec<_> = content.tensor_infos.iter().collect();
    names.sort_by_key(|(n, _)| n.as_str());
    for (name, info) in names {
        if !name.starts_with("blk.") || name.starts_with("blk.0.") {
            println!("{name:32} {:?} {:?}", info.ggml_dtype, info.shape.dims());
        }
    }
    Ok(())
}
