# workblock-engine

A pure calculation library that transforms raw activity and keystroke telemetry into **workblocks** (temporal work sessions) and **atomic slices** (the smallest meaningful units of work within them).

No I/O. No cloud. No AI. Just deterministic computation: DataFrames in, DataFrames out.

## Run Tests

```bash
uv sync --group dev
uv run pytest
```

```python
from workblock_engine import process

result = process(activities_df, keystrokes_df)

result.workblocks       # pl.DataFrame — one row per work session
result.atomic_slices    # pl.DataFrame — one row per app/URL segment
result.skipped_users    # list[str]    — users that failed processing
```

## Table of Contents

- [Run Tests](#run-tests)
- [Why This Exists](#why-this-exists)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Concepts](#concepts)
- [End-to-End Algorithm](#end-to-end-algorithm)
  - [Stage 1: Data Cleaning](#stage-1-data-cleaning)
  - [Stage 2: Overlap Resolution](#stage-2-overlap-resolution)
  - [Stage 3: Keystroke Matching](#stage-3-keystroke-matching-4-phase)
  - [Stage 4: Workblock Construction](#stage-4-workblock-construction)
  - [Stage 5: Output Assembly](#stage-5-output-assembly)
- [Output Schemas](#output-schemas)
- [Design Decisions & Rationale](#design-decisions--rationale)
- [Equivalence Testing](#equivalence-testing)
- [Consumer Guide: What You Need to Build](#consumer-guide-what-you-need-to-build)
- [Development](#development)

---

## Why This Exists

This library was extracted from [productivity-insights](https://github.com/rustancorpuz/productivity-insights), a full-stack productivity analytics platform. The extraction was motivated by:

1. **Reusability** — The core calculation logic (clean, match, build) is useful independent of PI's cloud integration, AI layer, and report generation. Other systems should be able to consume it as a library.

2. **Testability** — A pure function with no side effects is trivially testable. No GCS mocks, no API stubs, no environment variables.

3. **Clear boundaries** — PI does too many things (data loading, cleaning, matching, building, enriching, formatting, AI analysis, GCS I/O). Extracting the deterministic calculation core makes both projects easier to reason about.

4. **Dependency minimization** — This library depends only on Polars. PI depends on Anthropic SDK, Google GenAI, GCS, Redis, and more. Consumers who just need workblock computation shouldn't pull all of that.

---

## Installation

```bash
pip install workblock-engine
# or
uv add workblock-engine
```

**Requirements:** Python 3.11+, Polars >= 1.30.0

---

## Quick Start

```python
import polars as pl
from workblock_engine import process

# Load your telemetry data
activities = pl.read_parquet("activities.parquet")
keystrokes = pl.read_parquet("keystrokes.parquet")

# Process all users
result = process(activities, keystrokes)

# Inspect results
print(f"Workblocks: {result.workblocks.height}")
print(f"Atomic slices: {result.atomic_slices.height}")
print(f"Skipped users: {result.skipped_users}")

# Filter to a specific user
alice_slices = result.atomic_slices.filter(pl.col("user_id") == "alice")
```

---

## Concepts

| Term | Definition |
|------|-----------|
| **Activity** | A timestamped record of a user interacting with an application (e.g., "Chrome from 09:00 to 09:15") |
| **Keystroke** | A timestamped record of keyboard input within an application |
| **Workblock** | A group of activities separated by no more than 60 minutes of idle time. Represents a continuous work session. |
| **Atomic Slice** | The smallest meaningful unit within a workblock: a contiguous time segment with a single primary application, bounded by context switches or minute boundaries. |
| **Synthetic Activity** | An activity created by the engine to account for keystrokes that don't fall within any real activity (gap-filling or orphan capture). |

---

## End-to-End Algorithm

The pipeline processes raw telemetry through five stages. Each user is processed independently — failures in one user don't affect others.

```
                    ┌─────────────────┐
                    │  Raw Activities  │
                    │  Raw Keystrokes  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  1. CLEANING    │  Normalize, deduplicate, extract app names
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  2. OVERLAP     │  Trim overlapping activity end times
                    │     RESOLUTION  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  3. KEYSTROKE   │  4-phase algorithm: exact → nearest →
                    │     MATCHING    │  gap-fill → orphan capture
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  4. WORKBLOCK   │  Group by 60-min gaps, split at midnight,
                    │     BUILDING    │  create atomic slices from time segments
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  5. OUTPUT      │  Schema validation, type casting,
                    │     ASSEMBLY    │  user_id/date injection
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ WorkblockResult │
                    │  .workblocks    │
                    │  .atomic_slices │
                    │  .skipped_users │
                    └─────────────────┘
```

### Stage 1: Data Cleaning

**ActivityCleaner** validates and normalizes activity data:

- Ensures required columns exist: `user_id`, `timestamp`, `name`, `title`, `url`, `task_name`
- Creates `exec` column from `name` if missing
- Converts string datetimes to UTC-aware `datetime` objects
- Creates `start`/`end` columns from `timestamp` if missing (defaults to ±10 min window)
- Extracts `app_exe` (last 2 path segments) and `app_filename` (final segment) from `exec`
- Calculates `duration_minutes` from `start`/`end`
- Deduplicates by `id` column if present
- Sorts by `user_id`, then `timestamp`

**KeystrokeCleaner** validates and normalizes keystroke data:

- Normalizes column names: `application`→`app`, `keystroke`→`keystrokes`, `keys`→`keystrokes`, `title`→`window_title`
- Creates `app` from `exec` path if `app` column is missing
- Converts string datetimes to UTC
- Extracts app name from exec path (last segment after `/` or `\`)
- Applies **keystroke compression**: collapses repeated keys (`<Cmd><Cmd><Cmd>` → `<Cmd>{3}`) and repeated characters (`aaaaa` → `a{5}`, threshold > 4)
- Calculates `keystroke_count` and estimates `duration_seconds` (÷ 3.3 characters per second)
- Generates `window_title` from URL or app name if missing
- Deduplicates by `id` if present, sorts by `user_id`, `timestamp`

### Stage 2: Overlap Resolution

Activities can overlap when tracking software reports concurrent foreground windows. The resolver ensures non-overlapping timelines:

1. Sort activities by `start` time
2. For each consecutive pair: if `activity[i].end > activity[i+1].start`, trim `activity[i].end` to `activity[i+1].start`
3. Recalculate `duration_minutes` after trimming

**Rationale:** Later-starting activities take priority. This preserves all activities while eliminating double-counted time. The alternative (dropping the earlier activity entirely) would lose data.

### Stage 3: Keystroke Matching (4-Phase)

The matching algorithm assigns every keystroke to an activity. It runs four phases with decreasing confidence, ensuring a 100% match rate.

```
 All Keystrokes
      │
      ├──▶ Phase 1: Exact Match ──────▶ matched (high confidence)
      │    (temporal overlap + same app)
      │
      ├──▶ Phase 2: Nearest Neighbor ─▶ matched (medium confidence)
      │    (closest activity, app-preferred, ≤5 min)
      │
      ├──▶ Phase 3: Gap Filling ──────▶ synthetic activities created
      │    (keystrokes in >5 min gaps between activities)
      │
      └──▶ Phase 4: Orphan Capture ───▶ synthetic activities created
           (everything remaining, grouped by app + ≤5 min proximity)
```

**Phase 1 — Exact Match (temporal + app)**

Cross-join activities with keystrokes. Keep matches where:
- Keystroke timestamp falls within `[activity.start, activity.end]`
- Keystroke app matches activity app (after normalization)

If a keystroke matches multiple activities, take the first match. This phase handles the common case where the user is typing in the active application.

**Phase 2 — Nearest Neighbor with App Preference**

For unmatched keystrokes from Phase 1:
- Calculate time distance to every activity (clamped to activity boundaries)
- Apply a 180-second penalty for cross-app matches
- Filter to ≤ 300 seconds (5 minutes) total distance
- Take the best (lowest score) match per keystroke

This handles keystrokes that occur slightly outside an activity's recorded window but clearly belong to it.

**Phase 3 — Gap Filling (Synthetic Activities)**

For keystrokes still unmatched:
- Find gaps between consecutive activities where `gap > 5 minutes`
- Collect unmatched keystrokes falling within each gap
- Group gap keystrokes by normalized app name
- Create one synthetic activity per app group with:
  - `start` = earliest keystroke time in group
  - `end` = latest keystroke time in group
  - `is_synthetic = true`, `synthetic_reason = "activity_gap"`

This accounts for work the user was doing that the activity tracker didn't capture (e.g., terminal work, background processes).

**Phase 4 — Orphan Capture (Synthetic Activities)**

For any keystrokes still remaining:
- Sort by app, then timestamp
- Group consecutive keystrokes where: same app AND time gap ≤ 5 minutes
- Create one synthetic activity per group with:
  - `is_synthetic = true`, `is_orphan = true`, `synthetic_reason = "orphaned_keystrokes"`

This is the catch-all. Every keystroke is now assigned to an activity (real or synthetic).

**Output enrichment:** After matching, each activity gains:
- `keystrokes` — JSON array of matched keystroke objects `[{id, time, text, app, url}]`
- `keystroke_count` — number of matched keystrokes
- `cross_app_keystrokes` — count of Phase 2 cross-app matches
- `is_synthetic`, `synthetic_reason` — provenance flags

### Stage 4: Workblock Construction

The WorkblockBuilder takes matched activities and produces workblocks + atomic slices.

#### Step 4a: Day-Boundary Splitting

Activities that span UTC midnight are split into per-day fragments. Associated keystroke timestamps are distributed to the appropriate fragment. This ensures each workblock belongs to a single calendar day.

#### Step 4b: Workblock Segmentation

Activities are grouped into workblocks using two rules:

1. **60-minute gap rule** — If the gap between consecutive activities exceeds 60 minutes, start a new workblock.
2. **Cross-day rule** — If an activity falls on a different calendar day AND the gap exceeds `cross_day_gap_minutes` (default: 180 min), force a new workblock even if the 60-min rule wouldn't trigger.

#### Step 4c: Time Segment Creation

Within each workblock, activities are decomposed into non-overlapping time segments:

1. Collect all boundary timestamps (every minute of every activity's span)
2. Sort and deduplicate boundaries
3. Create segments between consecutive boundaries
4. For each segment, identify all overlapping activities
5. Preserve zero-length segments for synthetic/orphan activities

This prevents double-counting when activities overlap within a workblock (which can happen with synthetic activities).

#### Step 4d: Atomic Slice Creation

Each time segment becomes an atomic slice:

1. **Primary activity selection** — If multiple activities overlap in a segment, select the one whose app matches the most keystrokes in that segment (keystroke app affinity voting). Falls back to the first activity if no keystrokes.
2. **Attribute aggregation** — Duration, keystroke list, concurrent activity count, multitasking flag, concurrent apps list.
3. **Keystroke deduplication** — Within a workblock, each keystroke is assigned to exactly one slice (by MD5 signature).

#### Step 4e: Consolidation & Filtering

1. **Consolidation** — Adjacent slices with the same app/URL, no keystrokes, same synthetic flags, and ≤1 second gap are merged into a single slice.
2. **Minimum duration filter** — Workblocks shorter than 5 minutes are dropped.
3. **Re-sequencing** — Slices are renumbered (1, 2, 3, ...) within each workblock.
4. **Active duration** — Each workblock's `active_duration_minutes` = sum of its slice durations.

### Stage 5: Output Assembly

The engine adds metadata and enforces output schemas:

- Injects `user_id` and `date` (derived from `start_time`) into every row
- Serializes `raw_keystrokes` to JSON strings
- Fills missing provenance columns (`is_synthetic`, `is_orphan`, `synthetic_reason`, `cross_app_keystrokes`) with defaults
- Casts all columns to their schema-defined types
- Concatenates per-user DataFrames into the final `WorkblockResult`

---

## Output Schemas

### Workblocks

| Column | Type | Description |
|--------|------|-------------|
| `workblock_id` | String | Deterministic ID: `wb:<user_hash>:<time_hash>:<sequence>` |
| `user_id` | String | User identifier |
| `date` | String | Calendar date (YYYY-MM-DD) |
| `start_time` | String | ISO 8601 start timestamp |
| `end_time` | String | ISO 8601 end timestamp |
| `duration_minutes` | Float64 | Wall-clock duration (end - start) |
| `active_duration_minutes` | Float64 | Sum of all slice durations (excludes gaps) |
| `total_activities` | Int64 | Number of raw activities in this workblock |

### Atomic Slices

| Column | Type | Description |
|--------|------|-------------|
| `workblock_id` | String | Parent workblock ID |
| `sequence` | Int64 | Order within workblock (1-indexed) |
| `user_id` | String | User identifier |
| `date` | String | Calendar date (YYYY-MM-DD) |
| `start_time` | String | ISO 8601 start timestamp |
| `end_time` | String | ISO 8601 end timestamp |
| `duration_minutes` | Float64 | Slice duration |
| `application` | String | Primary application name |
| `exec` | String | Executable path |
| `url` | String | URL (if browser activity) |
| `window_title` | String | Window title |
| `keystroke_count` | Int64 | Number of keystrokes in this slice |
| `raw_keystrokes` | String | JSON array of keystroke objects |
| `activity_count` | Int64 | Number of overlapping activities |
| `is_multitasking` | Boolean | True if `concurrent_activities > 1` |
| `concurrent_activities` | Int64 | Number of simultaneous activities |
| `concurrent_apps` | List(String) | List of concurrent app names |
| `is_synthetic` | Boolean | True if created from unmatched keystrokes |
| `is_orphan` | Boolean | True if from Phase 4 (orphan capture) |
| `synthetic_reason` | String | `"activity_gap"` or `"orphaned_keystrokes"` or `""` |
| `cross_app_keystrokes` | Int64 | Keystrokes matched cross-app (Phase 2) |

### Workblock ID Format

IDs are deterministic and reproducible from the same input:

```
wb:31my5iJtNx:V8MPp9k:00
   └──────────┘ └─────┘ └┘
    user hash    time    seq
    (base62)    (base62) (base62)
```

- **User hash**: MD5 of user_id → base62 (10 chars)
- **Time hash**: Milliseconds since epoch → base62
- **Sequence**: Zero-padded base62 block index

---

## Design Decisions & Rationale

### Pure function, no I/O

The entire library is a pure function: `(DataFrame, DataFrame) → WorkblockResult`. This means:
- No file system access, no network calls, no environment variables
- Trivially testable — no mocks needed
- Embeddable in any context (serverless, notebooks, CLI, web servers)

### No `.env`, no configuration files

productivity-insights requires a `.env` file with credentials and path configuration:

```bash
# productivity-insights requires all of these:
PRODUCTIVITY_INSIGHTS_OUTPUT_ROOT=gs://...
PRODUCTIVITY_INSIGHTS_INPUT_ROOT=gs://...
GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/...
GEMINI_API_KEY=...
UPSTASH_REDIS_REST_URL=...
UPSTASH_REDIS_REST_TOKEN=...
```

workblock-engine requires **none of this**. There is no `.env` file, no environment variable discovery, no credentials. The library takes DataFrames and returns DataFrames. Configuration that matters (like `cross_day_gap_minutes`) is passed as function parameters. Everything else — where the data comes from, where it goes, what API keys to use for downstream processing — is the consumer's concern.

This is a deliberate consequence of the pure-library design. Cloud configuration belongs in the orchestration layer, not the calculation layer.

### 100% keystroke match rate

Every keystroke is assigned to an activity. The 4-phase algorithm creates synthetic activities rather than dropping unmatched keystrokes. Rationale: dropped keystrokes are invisible data loss. Synthetic activities are explicit and flagged (`is_synthetic`, `is_orphan`), so consumers can filter them if they want, but the data is never silently lost.

### Overlap resolution by trimming, not dropping

When activities overlap, we trim the earlier activity's end time rather than dropping it. This preserves the information that the user was in that application, just with a shorter duration. The alternative (dropping or merging) would lose context about app transitions.

### 60-minute workblock gap

The 60-minute idle threshold for splitting workblocks is a product-level decision aligned with the PRD. It's not configurable in the public API to avoid fragmentation of the concept across consumers. The `cross_day_gap_minutes` parameter (default 180) controls only the midnight-boundary rule.

### Minute-level atomic slices

Atomic slices are bounded by minute boundaries (not just app transitions). This ensures consistent granularity for downstream analysis — a 30-minute coding session becomes 30 one-minute slices, each of which can be independently analyzed for keystrokes, multitasking, etc.

### Deterministic IDs

Workblock IDs are generated from `(user_id, start_time, sequence)` using MD5 + base62 encoding. Running the same input twice produces the same IDs. This enables idempotent processing and cross-system joins.

### Polars, not pandas

Polars was chosen for:
- Lazy evaluation and query optimization
- Zero-copy memory model
- Consistent handling of null values and temporal types
- Better performance on columnar operations

### No enrichment, no formatting

The engine deliberately excludes:
- Job/role enrichment
- Manifest (SOD/EOD) enrichment
- Usage statistics calculation
- Markdown report generation
- AI-powered narrative generation

These are consumer concerns. See [Consumer Guide](#consumer-guide-what-you-need-to-build).

---

## Equivalence Testing

This library is proven equivalent to [productivity-insights](https://github.com/rustancorpuz/productivity-insights) via golden-file testing:

1. **Golden generator** (`productivity-insights/scripts/generate_equivalence_golden.py`) runs the PI pipeline on deterministic fixture data and writes `golden_output.json` to both repos.

2. **This repo's test** (`tests/test_equivalence.py`) runs `process()` on the same fixture data and compares field-by-field against the golden reference.

3. **PI's test** (`tests/test_equivalence.py`) does the same from the PI side.

Both repos independently prove they match the golden reference without needing the other installed. When PI's behavior changes, regenerate the golden and both test suites catch any divergence.

**Known differences** (by design, excluded from comparison):
- `exec` field: WE preserves the original path; PI strips it during cleaning
- Provenance fields (`is_synthetic`, `is_orphan`, `synthetic_reason`, `cross_app_keystrokes`): WE-only
- `raw_keystrokes` internal format: WE uses JSON string; PI uses list of dicts
- `concurrent_apps` ordering: may differ between implementations

```bash
# Regenerate golden (run from productivity-insights repo)
uv run python scripts/generate_equivalence_golden.py

# Run equivalence tests (from this repo)
uv run pytest tests/test_equivalence.py -v
```

---

## Consumer Guide: What You Need to Build

If you use `workblock-engine` instead of the full `productivity-insights` pipeline, here is what you are responsible for — and what you gain by not having to deal with.

### What workblock-engine handles (you get this for free)

- Activity and keystroke cleaning/normalization
- Overlap resolution
- 4-phase keystroke matching with synthetic activity creation
- Workblock segmentation (60-min gap rule, midnight splitting)
- Atomic slice creation with minute-level granularity
- Multitasking detection
- Deterministic ID generation
- Schema-validated Polars DataFrame output

### What you need to build

#### 1. Data Loading

workblock-engine accepts Polars DataFrames. You need to load your telemetry into that format:

```python
# Your responsibility:
activities_df = pl.read_parquet("gs://bucket/activities.parquet")
keystrokes_df = pl.read_parquet("gs://bucket/keystrokes.parquet")

# Engine takes it from here:
result = process(activities_df, keystrokes_df)
```

PI's implementation uses `DataSourceManager` with auto-discovery from conventional GCS paths. You may need something similar or simpler depending on your storage.

#### 2. Enrichment (optional)

The engine output contains raw workblocks and slices. PI enriches these with:

| Enrichment | What it adds | PI module |
|------------|-------------|-----------|
| **Job data** | Role, team, region, org metadata per user | `WorkblockBuilder.enrich_with_jobs()` |
| **Manifests** | SOD/EOD declarations (area of focus, KPIs, self-reported goals) | `WorkblockBuilder.enrich_with_manifests()` |
| **Usage statistics** | Top 10 apps, domains, URLs with durations and percentages | `WorkblockBuilder.calculate_usage_statistics()` |

If you need these, you'll join the engine output with your own data sources.

#### 3. Report Formatting (optional)

The engine outputs structured DataFrames. If you need human-readable reports:

| Format | PI module | Description |
|--------|-----------|-------------|
| Markdown tables | `WorkblockFormatter` | Timeline tables with app, URL, keystrokes, enrichment |
| Narrative summaries | `WorkHorseLLM` | AI-generated narrative via Gemini |

#### 4. Multi-Day Consolidation (optional)

If you need to aggregate across multiple days:

| Capability | PI module | Description |
|-----------|-----------|-------------|
| Context bundling | `TokenBasedContextBuilder` | Discovers historical reports, assembles within token budget |
| Go2 briefs | `TokenBasedBriefGenerator` | Structured JSON insights with scores and recommendations |

These require the AI layer (Gemini API) and are the most complex parts of PI.

#### 5. Output Persistence

The engine returns in-memory DataFrames. You need to write them somewhere:

```python
result.workblocks.write_parquet("output/workblocks.parquet")
result.atomic_slices.write_parquet("output/slices.parquet")
```

PI uses `BatchResultsWriter` with date/user partitioning on GCS. Your needs may be simpler.

#### 6. Configuration & Credentials

PI manages configuration through environment variables (`.env` file) — GCS paths, API keys, Redis URLs. With workblock-engine, there is no `.env`. The mapping looks like this:

| PI environment variable | workblock-engine equivalent |
|------------------------|----------------------------|
| `PRODUCTIVITY_INSIGHTS_INPUT_ROOT` | Your data loading code decides where to read from |
| `PRODUCTIVITY_INSIGHTS_OUTPUT_ROOT` | Your persistence code decides where to write to |
| `GOOGLE_APPLICATION_CREDENTIALS` | Not needed — no GCS access |
| `GEMINI_API_KEY` | Not needed — no AI layer |
| `UPSTASH_REDIS_REST_URL/TOKEN` | Not needed — no rate limiting |

If you're building a full pipeline on top of workblock-engine, you'll likely need your own configuration for storage and API credentials — but the engine itself needs nothing.

#### 7. Error Handling

The engine catches per-user errors and reports them in `result.skipped_users`. You should:
- Log skipped users
- Decide whether to retry or alert
- Track success rates across runs

### What you DON'T need to worry about

- Keystroke compression and deduplication — handled internally
- Activity overlap conflicts — resolved before workblock building
- Unmatched keystrokes — synthetic activities ensure 100% coverage
- Timezone normalization — all timestamps forced to UTC
- Workblock ID generation — deterministic and collision-resistant
- Schema validation — output DataFrames are schema-enforced

---

## Migrating from ProductivityInsightsRunner

If your platform currently calls `ProductivityInsightsRunner` from productivity-insights (e.g., via a Dagster asset), here's what changes when you switch to workblock-engine.

### Before: ProductivityInsightsRunner

```python
# .env (required by PI's runner internally)
PRODUCTIVITY_INSIGHTS_INPUT_ROOT=gs://bucket/input
PRODUCTIVITY_INSIGHTS_OUTPUT_ROOT=gs://bucket/output
GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/credentials.json

# Your code
from productivity_insights import ProductivityInsightsRunner

runner = ProductivityInsightsRunner()
runner.run_for_date(date="2026-01-15", max_users=50)
# Runner reads from INPUT_ROOT, writes to OUTPUT_ROOT internally
# You have no control over the intermediate DataFrames
```

### After: workblock-engine

```python
# No .env needed by the engine. Your platform handles I/O directly.

import polars as pl
from workblock_engine import process

# YOU load the data (from GCS, BigQuery, local, wherever)
activities_df = pl.read_parquet("gs://bucket/input/activities.parquet")
keystrokes_df = pl.read_parquet("gs://bucket/input/keystrokes.parquet")

# Engine computes — no I/O, no env vars, no credentials
result = process(activities_df, keystrokes_df)

# YOU write the results (to GCS, Cloud SQL, BigQuery, wherever)
result.workblocks.write_parquet("gs://bucket/output/workblocks.parquet")
result.atomic_slices.write_parquet("gs://bucket/output/slices.parquet")
```

### Environment variable migration

| Variable | Before (PI Runner) | After (workblock-engine) |
|----------|-------------------|--------------------------|
| `PRODUCTIVITY_INSIGHTS_INPUT_ROOT` | Read by PI runner to find data | **Remove.** Your platform loads data directly. |
| `PRODUCTIVITY_INSIGHTS_OUTPUT_ROOT` | Read by PI runner to write results | **Remove.** Your platform writes results directly. |
| `GOOGLE_APPLICATION_CREDENTIALS` | Needed by PI for GCS access | Still needed by your platform for GCS — but the engine never touches it. |
| `GEMINI_API_KEY` | Used by PI's AI layer (WorkHorseLLM, brief generation) | **Remove** if you don't use PI's AI layer. Your platform has its own LLM integration. |
| `UPSTASH_REDIS_REST_URL/TOKEN` | Used by PI's rate limiter | **Remove.** The engine has no rate limiting. |
| `ENABLE_WORKBLOCK_USER_FILTER` | Your platform's user scoping | **Keep.** This is your concern — filter users before passing to `process()`. |
| `ANTHROPIC_API_KEY` | Your platform's LLM (L1 generation) | **Keep.** Unrelated to the engine — this is your AI layer. |
| `CLOUDSQL_*` | Your platform's persistence | **Keep.** Write engine results to Cloud SQL yourself. |
| `BQ_DATASET_SILVER` | Your platform's BigQuery sync | **Keep.** Sync engine results to BigQuery yourself. |

### What moves from PI to your platform

The key shift: **I/O and orchestration move from the library to the caller.**

1. **Data loading** — PI's `DataSourceManager` auto-discovers files from conventional GCS paths. With the engine, your Dagster assets load the data and pass DataFrames directly.

2. **User filtering** — PI's runner accepts `max_users`. With the engine, you filter the input DataFrames before calling `process()`.

3. **Output partitioning** — PI writes to `workblock_processing/date=YYYY-MM-DD/batch_N/user_ID/`. With the engine, you control the output layout.

4. **Enrichment** — PI's `WorkblockBuilder.enrich_with_jobs()` and `enrich_with_manifests()` run inside the runner. With the engine, you join enrichment data with the output DataFrames in your own code.

5. **Error handling** — PI's runner tracks failed users internally. With the engine, `result.skipped_users` gives you the list and you decide what to do.

### What you can delete

Once migrated, you can remove from your platform:

- The `productivity-insights` git dependency from `pyproject.toml`
- All `PRODUCTIVITY_INSIGHTS_*` environment variables from `.env` files
- Any `GEMINI_API_KEY` / `UPSTASH_*` variables that were only needed by PI's AI layer
- The vendored PI code (e.g., `tmp/vendor/productivity-insights/`)

---

## Scaling & Parallelization

Inside `process()`, users are processed **sequentially** in a for-loop. Each user is fully independent — no shared mutable state between iterations. This makes parallelization straightforward at the caller level.

### Why the engine doesn't parallelize internally

The library is a pure calculation with no opinion on your execution environment. You might be running on a single-core Cloud Run container, a multi-core Dagster worker, or a distributed Spark cluster. Baking `multiprocessing` or `concurrent.futures` into the library would conflict with callers that already manage their own parallelism (e.g., Dagster's `@multi_asset` fan-out, or Kubernetes job-per-user patterns).

### Strategy 1: Partition by user, call `process()` per partition

Split the input DataFrames by user and call `process()` in parallel. Each call handles one user (or a small batch of users).

```python
import polars as pl
from concurrent.futures import ProcessPoolExecutor
from workblock_engine import process, WorkblockResult

def process_user_batch(user_ids, all_activities, all_keystrokes):
    """Process a batch of users — runs in a separate process."""
    batch_activities = all_activities.filter(pl.col("user_id").is_in(user_ids))
    batch_keystrokes = all_keystrokes.filter(pl.col("user_id").is_in(user_ids))
    return process(batch_activities, batch_keystrokes)

# Split users into chunks
activities_df = pl.read_parquet("activities.parquet")
keystrokes_df = pl.read_parquet("keystrokes.parquet")
user_ids = activities_df["user_id"].unique().sort().to_list()
chunk_size = max(1, len(user_ids) // 8)  # 8 workers
chunks = [user_ids[i:i + chunk_size] for i in range(0, len(user_ids), chunk_size)]

# Fan out
with ProcessPoolExecutor(max_workers=8) as pool:
    futures = [
        pool.submit(process_user_batch, chunk, activities_df, keystrokes_df)
        for chunk in chunks
    ]
    results = [f.result() for f in futures]

# Merge results
all_workblocks = pl.concat([r.workblocks for r in results if not r.workblocks.is_empty()])
all_slices = pl.concat([r.atomic_slices for r in results if not r.atomic_slices.is_empty()])
all_skipped = [uid for r in results for uid in r.skipped_users]
```

### Strategy 2: Dagster fan-out (one asset per user or batch)

If you're using Dagster, the natural pattern is to partition your asset by user ID and let Dagster manage concurrency.

```python
from dagster import asset, DynamicPartitionsDefinition

user_partitions = DynamicPartitionsDefinition(name="user_id")

@asset(partitions_def=user_partitions)
def workblocks(context):
    user_id = context.partition_key
    activities = load_activities_for_user(user_id)  # your I/O
    keystrokes = load_keystrokes_for_user(user_id)  # your I/O
    result = process(activities, keystrokes)
    write_results(user_id, result)                   # your I/O
```

Dagster handles scheduling, retries, and concurrency limits. The engine stays a pure function inside each partition.

### Strategy 3: One `process()` call for everything (simplest)

For small-to-medium workloads (< 500 users, < 1M activities), the sequential loop inside `process()` is fast enough. A single call typically completes in seconds. Don't parallelize until you've measured and found it necessary.

```python
# Just call it — sequential is fine for most workloads
result = process(activities_df, keystrokes_df)
```

### Performance characteristics

| Workload | Typical time | Recommendation |
|----------|-------------|----------------|
| 1-50 users | < 5 seconds | Single `process()` call |
| 50-500 users | 5-60 seconds | Single call or 4-8 process workers |
| 500+ users | Minutes | Partition by user, fan out across workers |

The bottleneck is per-user processing (keystroke matching is O(activities × keystrokes) per user). Total time scales linearly with user count, making partitioning effective.

---

## Replay determinism

Reordering the same source rows produces the same workblocks, slices, and raw-keystroke JSON. Cleaning orders records by user, timestamp, source ID when present, and content before keeping the first record for each duplicate source ID. Null values sort first.

Activity overlap resolution orders equal starts by end time, then ID and content, so the longest interval survives. Exact interval ties keep the last record in that order. Keystroke matches with equal scores use the first activity in chronological order. Raw keystrokes with equal times are ordered by ID, app, text, and URL.

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run all tests
uv run pytest

# Run equivalence tests only
uv run pytest tests/test_equivalence.py -v

# Run with coverage
uv run pytest --cov=workblock_engine
```

### Project Structure

```
src/workblock_engine/
├── __init__.py              # Public API: process(), WorkblockResult
├── engine.py                # Orchestrator: per-user pipeline coordination
├── building/
│   ├── overlap.py           # Stage 2: Activity overlap resolution
│   └── workblock_builder.py # Stage 4: Workblock + atomic slice construction
├── cleaning/
│   ├── activity_cleaner.py  # Stage 1: Activity normalization
│   └── keystroke_cleaner.py # Stage 1: Keystroke normalization + compression
├── matching/
│   ├── keystroke_matcher.py # Stage 3: 4-phase keystroke matching
│   └── synthetic.py         # Stage 3: Synthetic activity creation (Phase 3 & 4)
└── utils/
    ├── app_utils.py         # App name normalization (exec path → canonical name)
    ├── id_utils.py          # Deterministic ID generation (base62, workblock IDs)
    ├── keystroke_compression.py  # Keystroke text compression
    └── time_utils.py        # Timezone-aware datetime utilities
```
