# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is a Vietnamese legal RAG (Retrieval-Augmented Generation) pipeline built for a competition (NEXTGEN). It answers up to 2000 legal questions (focused on SME/doanh nghiệp nhỏ và vừa law) by retrieving relevant Vietnamese legal articles, generating an answer with an LLM, self-verifying the answer against the retrieved context, and packaging the results into the competition's submission format. The codebase, comments, and logs are written in Vietnamese.

The pipeline is designed to run on Kaggle GPU notebooks (T4/P100), not as a long-running service — `main.py` is a batch script that processes a question file end-to-end and writes a submission zip.

## Commands

There is no test suite, linter, or build step configured in this repo. Common commands:

```bash
# Install dependencies (torch is expected to be pre-installed, e.g. on Kaggle; install separately locally)
pip install -r requirements.txt

# Run the full pipeline over a question file
python main.py --input R2AIStage1DATA.json

# Resume from output/checkpoint.json (skips already-processed question IDs)
python main.py --input R2AIStage1DATA.json --resume

# Adjust checkpoint frequency (default 50) and enable verbose/traceback logging
python main.py --input R2AIStage1DATA.json --resume --batch-size 50 --debug

# Rebuild the BM25 index standalone (writes data/bm25_corpus.pkl)
python -m src.index_bm25

# Re-export the BM25 corpus from the live Qdrant collection
python export_corpus.py
```

`kaggle_setup.py` is not an executable script to run locally — it's a template containing the shell/Python snippets to paste into Kaggle notebook cells (install deps, load secrets, copy source, run `main.py`, package the submission).

## Configuration

All tunable parameters live in `config/settings.py` (`Settings` class) — retrieval depth (`TOP_K_RAW`), RRF constant (`RRF_K`), reranker threshold/output size (`RERANKER_THRESHOLD`, `TOP_K_FINAL`), LLM generation params, retry behavior, and file paths. There are no separate dev/prod config files; behavior is changed by editing `Settings` directly.

Qdrant credentials (`QDRANT_URL`, `QDRANT_API_KEY`) are loaded from `.env` via `python-dotenv`. On Kaggle, secrets are injected via `kaggle_secrets.UserSecretsClient` instead (see `kaggle_setup.py`).

## Pipeline architecture

`main.py` wires together the modules in `src/` and drives a loop over all questions. Each question flows through these stages in `process_question()`:

1. **Hybrid Retrieval** (`src/hybrid_retriever.py`, `HybridRetriever.retrieve`) — Giai đoạn 2.
   - `MultiQueryExpander` expands the query into a few variants (rule-based synonym substitution; LLM-based expansion exists in code but is currently disabled/commented out).
   - For each variant: dense search against Qdrant Cloud using a 2-stage prefetch+rescore query (oversamples 10x in the prefetch stage for recall, falls back to an unfiltered search if a doc-number metadata filter returns nothing), and sparse search via BM25 (`src/index_bm25.py`).
   - Dense and sparse result lists across all variants are merged with Reciprocal Rank Fusion (`_rrf_merge`) into a single ranked candidate list (`Settings.TOP_K_RAW`).
   - BM25 tokenization (`tokenize_legal_text`) applies a domain-specific synonym map (e.g. SME abbreviations, tax acronyms) and a legal stopword list before tokenizing with `underthesea`.

2. **Reranking** (`src/reranker.py`, `LegalReranker.rerank`) — Giai đoạn 3.
   - Cross-encoder (`BAAI/bge-reranker-large`) scores (query, passage) pairs, filters by `RERANKER_THRESHOLD`, and keeps the top `TOP_K_FINAL`. Falls back to keeping the top `TOP_K_FINAL // 2` docs by score if the threshold filters out almost everything.

3. **Answer Generation** (`src/answer_generator.py`, `AnswerGenerator.generate`) — Giai đoạn 4.
   - LLM (`Settings.LLM_MODEL_NAME`, currently `Qwen/Qwen2.5-7B-Instruct`) loaded 4-bit quantized (bitsandbytes NF4). Context docs are packed into a prompt capped at `MAX_CONTEXT_CHARS`, with a system prompt forcing a single-paragraph Vietnamese answer (no markdown headers, no line breaks).
   - Output is post-processed to strip any `<think>...</think>` reasoning tags and collapse whitespace into one paragraph.
   - Also auto-derives `relevant_docs`/`relevant_articles` directly from the *context* metadata (not from the generated text) using the `doc_number|doc_name` / `doc_number|doc_name|article` pipe-delimited format the competition expects.

4. **Self-Verification** (`src/self_verifier.py`, `SelfVerifier.verify`) — Giai đoạn 5, anti-hallucination gate with 5 rules:
   - Rule 1 (hard): every "Điều X" cited in the answer must appear in the retrieved context text.
   - Rule 2 (hard): every legal document number cited must exist in `data/law_manifest.json`.
   - Rule 3 (warning only): document names should match the manifest's canonical name.
   - Rule 4 (warning only): numeric claims (percentages) not found in context are flagged.
   - Rule 5 (hard): the answer must contain at least one article reference *and* a citation trigger phrase (e.g. "Theo quy định tại Điều...").
   - If verification fails, `main.py` regenerates the answer once at a lower temperature (`Settings.LLM_REGEN_TEMPERATURE`) before giving up and keeping the best attempt.

5. **Post-Processing & Packaging** (`src/post_processor.py`, `PostProcessor`) — Giai đoạn 6.
   - Regex-extracts document numbers and "Điều/Khoản/Điểm" references straight from the final answer text (independent extraction path from the generator's metadata-based one in step 3 — `main.py` currently overwrites `relevant_docs`/`relevant_articles` with the generator's version after calling this).
   - Maps extracted document numbers to canonical strings via `data/law_manifest.json`, with a fallback heuristic (`_infer_doc_type_from_number`) when a number isn't in the manifest.
   - `validate_results()` checks the full result set for duplicate IDs, null fields, empty/short answers, and exactly 2000 records before submission.
   - `package_submission()` writes `output/results.json` and a timestamped `output/submission_<ts>.zip` containing it.

`src/evaluator.py` (`PipelineEvaluator`) runs alongside the loop purely for internal observability — it tracks retrieval/verification/regeneration stats per item and writes `logs/evaluation_report.md` (human-readable summary with auto-generated tuning recommendations) and `logs/detailed_log.json` (per-question stats) at the end of a run. It does not affect the submission output.

### Checkpointing

`main.py` writes `output/checkpoint.json` after every `--batch-size` questions. `--resume` loads it and skips already-processed IDs by `id`/`question_id`, so a crashed or interrupted Kaggle session can continue without reprocessing. On a per-question exception, a placeholder record (Vietnamese error message, empty `relevant_docs`/`relevant_articles`) is inserted so no ID is dropped from the final output.

## Data files

- `data/corpus_clean.json` — ~28k chunked legal article passages (`{"id", "text", "metadata"}`), the BM25 corpus. Regenerated from the live Qdrant collection via `export_corpus.py` if it ever needs refreshing; the dense index itself lives in Qdrant Cloud, not locally.
- `data/law_manifest.json` — dict keyed by document number (e.g. `"91/2015/QH13"`) to canonical metadata (`doc_id`, `document_type`, `law_name`, `btc_standard_string`). This is the ground truth used by both `SelfVerifier` (Rule 2/3) and `PostProcessor` for the BTC-standard output strings.
- `data/legal_documents_catalog.json` — list of ~131 higher-level law catalog entries with domain/priority/frequency metadata, mostly for human reference on which laws matter most for the question set.
- `data/bm25_corpus.pkl` — pickled, pre-tokenized BM25 index built from `corpus_clean.json` (built once via `src/index_bm25.py`, then loaded instead of rebuilt on subsequent runs).

## Working in this repo

- Keep new user-facing strings (system prompts, log messages, error messages) in Vietnamese to match the existing codebase.
- `relevant_docs`/`relevant_articles` have two independent extraction paths (generator metadata-based, and post-processor regex-based) — be aware of which one `main.py` actually uses (currently the generator's) before changing either in isolation.
- The submission format is fixed by the competition: each result record must have `id`, `question`, `answer`, `relevant_docs`, `relevant_articles`, and the final file must contain exactly 2000 records (`validate_results` enforces this).