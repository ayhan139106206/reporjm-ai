# Monster-1 training data requirements

The current repository contains no approved real Persian training corpus. Training is therefore blocked. The five-line sample used by an earlier workflow and the random token IDs it trained on are not acceptable data and must never be used as a substitute.

## Required corpus package

Provide one archive containing these UTF-8 files:

- `train.txt`: deduplicated Persian prose for next-token pretraining.
- `validation.txt`: a held-out Persian split used for model selection.
- `test.txt`: a separate held-out split used only for final reporting.
- `corpus_manifest.json`: source name and HTTPS source URL, license name and HTTPS license URL, language (`fa`), collection date, attribution requirements, `rights_to_train: true`, `privacy_reviewed: true`, and SHA-256 values for all three text files.

Example manifest shape (replace every example value with the actual provenance):

```json
{
  "source_name": "Actual dataset name",
  "source_url": "https://example.org/dataset",
  "license": "Actual license",
  "license_url": "https://example.org/license",
  "language": "fa",
  "collection_date": "YYYY-MM-DD",
  "rights_to_train": true,
  "privacy_reviewed": true,
  "attribution": "Required attribution text or none",
  "sha256_train": "64-character SHA-256",
  "sha256_validation": "64-character SHA-256",
  "sha256_test": "64-character SHA-256"
}
```

The corpus must be real, sourced text—not generated examples, random token IDs, or a handful of hand-written prompts. It must not contain private personal data or material whose license/terms prohibit training. Remove duplicates and keep the source/attribution information with the resulting model artifact.

## Minimum data gate

The current architecture has 5,522,944 parameters. This project uses a conservative planning target of 20 training tokens per parameter: **at least 110,458,880 SentencePiece tokens** in `train.txt`, plus at least **1,000,000 tokens each** in validation and test. This is an engineering gate, not a promise of assistant quality. The corpus must also be predominantly Persian and large/diverse enough to train a 4,096-piece SentencePiece vocabulary.

The pipeline counts tokens with the exact trained tokenizer. It will not create `monster1.pt` or `monster1.onnx` if the corpus or compute budget is below the gate. It will write a preflight report explaining the missing files, license/provenance fields, or token counts instead.

## Supplying data to the free workflow

Do not commit a large corpus to Git. For a manual GitHub Actions run, provide a direct HTTPS URL to the archive and its SHA-256 checksum. The workflow downloads a static file only, checks the checksum, extracts only the four expected files, and performs no model/API calls. If no URL/checksum is supplied, it stops at preflight and uploads the status report only.

## Compute constraint

A 5.5M-parameter model trained on roughly 110M tokens may exceed a single free CPU Actions run. The training workflow is manually gated and time-limited. A run that reaches its time/step budget before the token target is met is reported as incomplete; its checkpoint is not published as the final `monster1.pt` and no ONNX release is created.
