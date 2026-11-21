# Monster-1 Persian language model

**Current status: real-data preparation is pinned and reproducible; no completed, quality-validated model is available.** The API therefore reports `model_ready: false` and `/v1/chat` returns HTTP 503 until a genuine from-scratch run passes the data, validation, and provenance gates.

## Architecture

Monster-1 is a small causal Transformer initialized from scratch: 4 layers, hidden size 256, 4 attention heads, 1,024-token maximum context, and 5,522,944 parameters. The target SentencePiece vocabulary is exactly 4,096. `model.py` and `config.json` define the architecture; `tokenizer.py` trains a tokenizer only from a supplied real Persian corpus and never creates fallback text.

The legacy `monster1.pt` and `monster1.onnx` files in the current GitHub history are preserved for audit/compatibility, but they are **not trained Persian model artifacts**. The checkpoint is byte-for-byte identical to the output of an earlier workflow that used a five-line synthetic corpus and random token IDs. The legacy ONNX passes graph validation and fixed-length inference, but only accepts `[1, 16]`; it is not a variable-length production model and must not be used by the API.

## Data and training

Read [DATASET_REQUIREMENTS.md](DATASET_REQUIREMENTS.md) before supplying data. The minimum planning gate is 110,458,880 actual SentencePiece tokens for training, plus 1,000,000 separate tokens each for validation and test, with source, license, attribution, rights, and checksums documented. This is a planning threshold—not a quality guarantee.

The `.github/workflows/monster1.yml` workflow is manual-only and opts out by default. When explicitly enabled, it fetches the pinned corpus below, verifies all shard checksums, prepares real-data splits with a privacy scan, and runs the repository tests and preflight before training from random initialization. Synthetic data and `.github/workflows/monster1-synthetic-test.yml` are never inputs to this workflow. The job is capped at 350 minutes overall and the training loop at 300 minutes; incomplete runs do not produce a final checkpoint. No model/API provider is called.

This repository is public. Standard GitHub-hosted Linux runners are free for public repositories; for a private repository, the workflow runs only if its manual input confirms that at least 350 Linux 2-core minutes ($2.10 at the listed rate) remain in the month's shared included compute. That balance is not queried automatically. The workflow also requires confirmation that at least 100 MB of the GitHub Free 500 MB artifact-storage allowance remains, and caps model artifacts at 100 MiB; reports and model artifacts are retained for 7 days. Model files are never committed: they are uploaded only for a private repository, or if a separate rights/privacy review has explicitly authorized public release. In a public repository, an artifact can be downloaded by anyone with repository read access, so the public-release confirmation defaults to false.

## Current verified corpus and compute status

The source is `AIgenius1/persian-pretraining-dataset`, pinned to revision `a6906d5585c981b74442f4be186ab9098e26b363`. The publisher identifies it as Persian Wikipedia and declares CC BY-SA 4.0; all 46 shard sizes and SHA-256 values were checked, and article URLs are retained for attribution. Rights for publicly releasing downstream model weights still need separate review. The prepared splits contain 798,903 training, 44,210 validation, and 44,202 test articles. Exact SentencePiece counts are 295,868,792 training tokens, 16,627,727 validation tokens, and 16,202,149 test tokens. Corpus files and model artifacts are not stored in Git.

A CPU-only benchmark on actual training-text token windows found batch 32 fastest at about 5,030 tokens/second: the 110,458,880-token minimum alone is estimated at 6.10 hours, before corpus preparation, validation, and final test evaluation. Batch 64 was slower at 6.58 hours. This exceeds the workflow's 300-minute training cap, so a CI run may end incomplete; an incomplete run is not production-ready and the API stays disabled. A completed run and held-out loss gate are not by themselves a claim of general language quality.

## Isolated synthetic smoke test

Run `.github/workflows/monster1-synthetic-test.yml` manually and leave its mode set to `synthetic_test` to exercise the end-to-end pipeline without a real corpus. `scripts/generate_synthetic_data.py` makes a deterministic, locally generated Persian dataset for conversation, question/answer, simple math, logic, beginner Python, and simulated Persian commands. The separate runner trains a fresh instance of the existing architecture from random initialization, fits a SentencePiece tokenizer, checks the exact model/tokenizer vocabulary match, exports ONNX, and runs several Persian prompts. The default smoke run uses 300 optimizer steps (76,800 training tokens), only to make the synthetic demo more likely to emit visible text; it is far below the production data gate and does not imply useful language ability. The demo decoder skips whitespace-only token pieces, normalizes Persian/Arabic decimal digits to the synthetic corpus's ASCII digits, and stops on EOS; it never substitutes a canned response. It makes no model/API calls and never loads the legacy checkpoint.

All test outputs are isolated under `artifacts/synthetic_test/`, including `monster1.pt`, `monster1.onnx`, `tokenizer.model`, `tokenizer.vocab`, `training_report.json`, and `corpus_manifest.json`. The manifest/report explicitly label the data and model as synthetic and not for production. Because this small corpus naturally yields only about 360 unigram pieces, the tokenizer records the remaining IDs as unused, explicitly named test-reserve entries to keep the model head at the existing 4,096-wide configuration; these are not claimed as learned vocabulary and are masked during sample generation. This mode does not change or bypass the real-data token gate, does not write to `artifacts/model/`, and cannot make `engine_ready` true. ONNX parity and sample generation are mechanics checks only; they are not a language-quality result. GitHub Actions uploads the outputs and detailed failure report as a 30-day Artifact, including when a pipeline step fails.

### Large synthetic token-count stress corpus

If a large file is needed to test storage, hashing, or pipeline handling without finding an external corpus, first run the isolated `synthetic_test` pipeline to create its local tokenizer, then run `python scripts/generate_synthetic_scale.py`. By default this writes at least 110,458,880 training tokens and 1,000,000 tokens for each held-out split under `data/synthetic_scale/`, with a checksum and exact SentencePiece counts in its manifest. It deterministically repeats the small `synthetic_test` examples; it does **not** create 110 million tokens of new knowledge. The generated corpus is marked synthetic, is ignored by Git, makes no network/API calls, and is explicitly rejected by the real-data training gate. It must not be presented as a real Persian corpus or used to claim training quality.

## API and deployment layout

`api.py` is the independent FastAPI CPU inference service. It loads `monster1.pt`, `monster1.onnx`, `monster1_tokenizer.model`, and `model_manifest.json` from `artifacts/model/` only when checksums, vocabulary, dynamic ONNX input, real-data provenance, and completed-training metrics are valid. It runs inference locally with ONNX Runtime and does not call ChatGPT or any other external model API, so it has no third-party model-token quota or per-token API charge; it uses the machine's CPU and still needs a valid trained model. The API caps each request at 128 generated tokens as a local resource guardrail, not a provider quota. The current repository has no such verified model, so `/v1/chat` correctly remains HTTP 503 until eligible real-data training is completed. Run the local service with:

```bash
python -m pip install -r requirements.txt
uvicorn api:app --host 127.0.0.1 --port 8000
```

### Experimental synthetic demo API

To expose the existing synthetic smoke-test model through a separate local-only demo route, first generate its test artifacts with `python scripts/run_synthetic_test.py --mode synthetic_test`, then start the API as above. The demo is available at `GET /v1/demo/health`, `GET /v1/demo/model`, and `POST /v1/demo/chat`. Example request:

```bash
curl -X POST http://127.0.0.1:8000/v1/demo/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"سلام"}],"max_new_tokens":16}'
```

This route uses local ONNX Runtime on CPU and makes no external model/API calls. It is explicitly labeled `experimental`, `is_synthetic`, and `not_for_production`; its `engine_ready` field is always `false`. The generated responses may be poor or meaningless, and this endpoint does not change the readiness or behavior of the production `GET /v1/health`, `GET /v1/model`, or `POST /v1/chat` routes. It is not publicly deployed, and the Cloudflare Worker is unchanged.

If the tiny smoke-test model emits no displayable text, the API reports `generation_status: "empty_model_output"`, returns a plain diagnostic message, and sets `reply_is_model_output: false` rather than pretending the diagnostic came from the model.

The Cloudflare Worker in `src/index.js` remains a gateway only. It does not run PyTorch/ONNX. Set its `MODEL_API_URL` to a separately hosted Monster-1 FastAPI service before expecting inference; without that setting, chat correctly returns 503. No paid hosting or external inference API is configured.

## Local validation

```bash
python -m unittest discover -s tests -v
python train.py preflight
```

With no corpus, preflight writes `reports/training_preflight.json`, reports the exact missing data, and does not create weights or a tokenizer. Do not describe the legacy checkpoint or the test-only ONNX fixture as a trained model.
