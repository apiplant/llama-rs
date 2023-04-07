### llama-rs

Torch-based implementation of Llama. WIP

NOTHING WORKS, just the sentencepiece tokenizer.

Run with:
```
cargo run ../llama.cpp/models 7B "The first man on the moon was"
```

The target directory should contain
```
├── 7B
│   ├── consolidated.00.pth
│   ├── ggml-model-f16.bin
│   ├── ggml-model-q4_0.bin
│   └── params.json
├── merge.txt
└── tokenizer.model
```

Download the model from [here](https://github.com/facebookresearch/llama/pull/73/files).

Follow [these](https://github.com/ggerganov/llama.cpp#usage) instructions to quantize the model to 4-bits.
