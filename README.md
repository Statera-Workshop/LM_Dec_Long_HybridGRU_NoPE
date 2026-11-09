# LM_Dec_Long_HybridGRU_NoPE

HybridGRU_NoPE - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked (6 seed-repeat runs, see results)**
- Failure class: Note: recommended long-memory structure - all 14 training runs succeeded (recall and memory update), exact at 16x the training length (8192), 370 MB inference at 8192

## Provenance

- Source repository: (new) hybrid: GRU layer + x-transformers decoder
- Upstream / license: lucidrains/x-transformers (MIT), pip 2.31.7 + torch.nn.GRU

## Differences from the source ([Unified] protocol)

- 1 GRU layer (dim 512) added residually before a 6-layer nope decoder (same decoder as xt_nope)

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 20.74 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.696 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.418 / test 1.400 bits per byte
- LM-02 enwik8 (1B tokens): valid 1.336 / test 1.328 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 380 MB, inference 118 MB
- Peak GPU memory at length 8192 (batch 1): train 1375 MB, inference 370 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 1.00 (keys removed 0.20); 4 pairs @ 2048: 1.00 (keys removed 0.13); 16 pairs @ 512: 1.00 (keys removed 0.04); 16 pairs @ 2048: 1.00 (keys removed 0.04)
- LM-05 recall `lm05v3long`: 4 pairs @ 512: 1.00 (keys removed 0.11); 4 pairs @ 2048: 1.00 (keys removed 0.09); 4 pairs @ 8192: 1.00 (keys removed 0.07); 16 pairs @ 512: 1.00 (keys removed 0.04); 16 pairs @ 2048: 1.00 (keys removed 0.03); 16 pairs @ 8192: 1.00 (keys removed 0.03)
- LM-05 recall `lm05v3s2`: 4 pairs @ 512: 1.00 (keys removed 0.12); 4 pairs @ 2048: 1.00 (keys removed 0.09); 16 pairs @ 512: 1.00 (keys removed 0.04); 16 pairs @ 2048: 1.00 (keys removed 0.03)
- LM-05 recall `lm05v3s3`: 4 pairs @ 512: 0.99 (keys removed 0.16); 4 pairs @ 2048: 0.99 (keys removed 0.08); 16 pairs @ 512: 1.00 (keys removed 0.04); 16 pairs @ 2048: 1.00 (keys removed 0.03)
- LM-05 recall `lm05v3longs2`: 4 pairs @ 512: 1.00 (keys removed 0.07); 4 pairs @ 2048: 1.00 (keys removed 0.04); 4 pairs @ 8192: 1.00 (keys removed 0.04); 16 pairs @ 512: 1.00 (keys removed 0.03); 16 pairs @ 2048: 1.00 (keys removed 0.02); 16 pairs @ 8192: 1.00 (keys removed 0.03)
- LM-06 memory update `lm06`: @512: latest 1.00, unchanged 1.00, stale 0.00; @2048: latest 1.00, unchanged 1.00, stale 0.00
- LM-06 memory update `lm06long`: @512: latest 1.00, unchanged 1.00, stale 0.00; @2048: latest 1.00, unchanged 1.00, stale 0.00; @8192: latest 1.00, unchanged 1.00, stale 0.00
- LM-06 memory update `lm06s2`: @512: latest 1.00, unchanged 1.00, stale 0.00; @2048: latest 1.00, unchanged 1.00, stale 0.00
- LM-06 memory update `lm06s3`: @512: latest 1.00, unchanged 1.00, stale 0.00; @2048: latest 0.99, unchanged 1.00, stale 0.01
- LM-06 memory update `lm06longs2`: @512: latest 1.00, unchanged 1.00, stale 0.00; @2048: latest 1.00, unchanged 1.00, stale 0.00; @8192: latest 1.00, unchanged 1.00, stale 0.00

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check hyb_gru_nope
python run.py lm02 hyb_gru_nope      # enwik8 is downloaded to data/ on first use
```
