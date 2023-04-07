use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::prelude::*;
use tokenizers::models::bpe::BPE;
use tokenizers::tokenizer::{Result, Tokenizer};

#[derive(Serialize, Deserialize, Debug)]
struct Params {
    dim: i32,
    multiple_of: i32,
    n_heads: i32,
    n_layers: i32,
    norm_eps: f64,
    vocab_size: i32,
}
#[derive(Serialize, Deserialize, Debug)]
struct ModelArgs {
    max_seq_len: i32,
    max_batch_size: i32,
    params: Params,
}

pub fn main() -> anyhow::Result<()> {
    let device = tch::Device::Cpu;
    let temperature = 0.8;
    let top_p = 0.95;
    let max_seq_len = 512;
    let max_batch_size = 32;
    let max_gen_len = 256;

    let args: Vec<_> = std::env::args().collect();
    let (llama_dir, model_dir, prompt) = match args.as_slice() {
        [_, l, m, p] => (l.to_owned(), m.to_owned(), p.to_owned()),
        _ => panic!("usage: main llama_dir 7B prompt"),
    };
    // let checkpoint = tch::Tensor::loadz_multi(format!(
    //     "{}/{}/{}",
    //     llama_dir, model_dir, "consolidated.00.pth"
    // ))?;
    let mut file = File::open(format!("{}/{}/{}", llama_dir, model_dir, "params.json"))
        .expect("File not found");

    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Something went wrong reading the file");
    let params: Params = serde_json::from_str(&contents).expect("Error parsing JSON");
    println!("{:?}", params);

    // for (name, tensor) in checkpoint.iter() {
    //     println!(": {name} {tensor:?}")
    // }

    let tokenizer =
        sentencepiece::SentencePieceProcessor::open(format!("{}/{}", llama_dir, "tokenizer.model"))
            .expect("Something went wrong reading the tokenizer model");

    let model_args = ModelArgs {
        max_seq_len,
        max_batch_size,
        params: Params {
            vocab_size: tokenizer.len() as i32,
            ..params
        },
    };

    let lengths = vec![model_args.max_seq_len, max_gen_len + prompt.len() as i32];
    let total_len = lengths.iter().min().unwrap();

    let mut prompt_tokens = tokenizer.encode(&prompt).unwrap();
    prompt_tokens.insert(
        0,
        sentencepiece::PieceWithId {
            piece: "".to_string(),
            id: tokenizer.bos_id().unwrap(),
            span: (0, 0),
        },
    );

    println!("{:?}", prompt_tokens);

    // let mut tokens = tch::Tensor::full(
    //     &vec![1, *total_len as i64],
    //     tokenizer.pad_id().or(Some(0)).unwrap() as i64,
    //     (tch::kind::Kind::Int64, device),
    // );
    // for (index, prompt_token) in prompt_tokens.iter().enumerate() {
    //     println!("{}", tokens.get(index as i64));
    // }

    let bpe_builder = BPE::from_file(
        &format!("{}/{}", llama_dir, "tokenizer.model"),
        &format!("{}/{}", llama_dir, "merge.txt"), // TODO: I still don't have this file
                                                   // I don't see how the python code of the tokenizers inside huggingface/transformers require this file
                                                   // Maybe there is another way of creating a BPE model without merge?
    );
    let bpe = bpe_builder
        .dropout(0.1)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();

    let mut tokenizer = Tokenizer::new(bpe);

    let encoding = tokenizer.encode("Hey there!", false).unwrap();
    println!("{:?}", encoding.get_tokens());

    // let output = model.forward_ts(&[image.unsqueeze(0)])?.softmax(-1);
    // for (probability, class) in imagenet::top(&output, 5).iter() {
    //     println!("{:50} {:5.2}%", class, 100.0 * probability)
    // }

    Ok(())
}
