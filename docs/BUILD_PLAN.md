# AI-Photo-Org — Build Plan

Derived from **Technical Specification v0.3**. This document is the *how*; the spec is the *what*.
Where this plan disagrees with the spec, the disagreement is called out explicitly in
[§6 Gap register](#6-gap-register) — the spec is not silently overridden.

**Repo state at time of writing:** empty. `LICENSE` (MIT ✅ satisfies §3.1), `README.md` stub,
Python `.gitignore`. Nothing else exists.

> **Revision (2026-09-11, post-review):** four items below reflect owner calls made after the first
> pass of this plan. **G2** (test corpus) is deprioritized — not needed to start building, only to
> validate M3, so it no longer blocks Phase 0. **G4** (installer size) is resolved — the size budget
> was relaxed to <5 GB, which removes the forcing function for INT8 quantization entirely. **G1**
> (reference hardware) got a partial, evidence-based answer from a real (if limited) CI-style sandbox
> — see the update inside G1. **G16** was corrected after actually running the numbers instead of
> estimating them; the claimed win was smaller than stated, and running it surfaced a real bug.

---

## 1. How to read this

The spec's milestones M0–M7 are sound and are kept. This plan adds:

- a **Phase 0** in front of M0, for long-lead items the spec assumes exist but that nobody has built yet;
- concrete **work packages** per milestone with exit criteria you can actually check;
- a **package layout** that satisfies the headless-core rule (F-1.10 note) from the first commit;
- a **gap register** of 20 things the spec is missing, ranked by when they will hurt.

---

## 2. Phase 0 — Long lead items (before any benchmark)

M0's exit criteria are *"real images/sec measured on target hardware for CLIP + YuNet + SFace in
ONNX"*. Three preconditions for that sentence do not currently exist. All three have lead time, so
they start now, in parallel.

### P0.1 — Define the reference machine

The spec quotes "a 4-core desktop" in §7.1 but never names target hardware, which makes N-2
(≥1.7 img/s floor, ≥6 img/s target) unfalsifiable — any number can be explained away as "wrong PC".

**Do:** write `docs/REFERENCE_HARDWARE.md` naming one primary and two secondary machines, e.g.

| Tier | CPU | RAM | Library drive | Role |
|---|---|---|---|---|
| Primary | 4c/8t, AVX2, ~2019-era | 16 GB | SATA HDD | N-2 floor is measured here |
| Fast | 8c/16t, AVX2/AVX-512 | 32 GB | NVMe SSD | N-2 target, DirectML gain |
| Floor | 2c/4t laptop, iGPU | 8 GB | SATA SSD | N-5 RAM ceiling, N-4 cold start |

Every perf number in the repo cites a tier. No tier, no number.

> **What a disposable cloud sandbox can and can't tell you.** Tried this against a throwaway Linux
> container (4 vCPU, shared/virtualized, no GPU, virtualized block storage) to see how far a sandbox
> substitutes for a named machine. Two results:
> - A synthetic ONNX graph FLOP-matched to ViT-B/32's ~4.4 GFLOPs/image reported ~60,000 img/s under
>   `onnxruntime`. That number is meaningless — a single dense matmul hits a far higher fraction of
>   peak FLOPs than a real 12-layer transformer's actual op mix (attention, layernorm, many small
>   matmuls) ever does. **FLOP count alone is not a valid throughput proxy; only the real exported
>   ONNX graph gives trustworthy numbers** (P0.3).
> - What *did* transfer: portable, architecture-independent behavior like JPEG decode strategy
>   (§7.3.1, G16 below) can be measured anywhere Python runs.
> - What can't transfer at all: anything Windows-only (DirectML, `SetThreadExecutionState`, WMI
>   drive-type detection), and anything depending on real storage hardware — a virtualized cloud disk
>   has neither a real HDD's seek penalty nor a real SSD's queue depth behavior.
>
> Net: a sandbox is useful for catching methodology mistakes before they reach real hardware, and for
> validating logic that doesn't depend on the OS or the disk. It is not a substitute for P0.1's named
> machines, particularly for §7.2's HDD/SSD read strategy and §7.3.6's DirectML measurement.

### P0.2 — Acquire the test corpus (longest lead item in the project)

§12.4 says *"generate or assemble a real 50k-photo library early"* and §12.1 wants a labeled
200–500 photo accuracy set. Neither exists, and the face half is genuinely hard to source: you need
**repeat identities across many photos, in messy consumer conditions** — that is exactly the data
that is hardest to obtain ethically.

Three corpora are needed, and they are different problems:

| Corpus | Size | Purpose | Source strategy |
|---|---|---|---|
| **A — Scale** | 50k images | M1 scan, N-2/N-2b/N-2c, thumbnail cache sizing, SQLite behavior | Open Images / Unsplash Lite / Flickr Commons. Variety matters more than realism here. |
| **B — Faces** | 5–10k images, 30–60 identities | M3 clustering quality, threshold calibration | Hardest. Options: your own library + explicit consent from anyone identifiable; a friend/family library under written consent; public figure sets only as a smoke test (they are not representative of consumer photo conditions). |
| **C — Labeled accuracy** | 200–500 images | §12.1 precision/recall per threshold | Hand-labeled subset of B plus scene concepts from A. This is a day of tedium — do it once, early, and never guess at thresholds again. |

**Do:** decide corpus B's provenance *now* and write the consent/provenance note into the repo.
It gates M3, which is 4–6 months out, and it is the item most likely to silently slip.

> Corpus B must never be committed to the repo. Keep it out of tree, and add a
> `.gitignore` entry plus a `tests/data/README.md` explaining where it lives locally.

### P0.3 — Build the ONNX model artifact pipeline

The spec names **OpenCLIP ViT-B/32 (ONNX)** as if it were a file you download. It is not, quite.
Producing it is real work that appears in no milestone:

1. Export the **image tower** to ONNX (dynamic batch axis — §7.3.4 batches 16–32).
2. Export the **text tower** to ONNX. Easy to forget; without it there is no F-2.1 search at all.
3. Ship a **CLIP BPE tokenizer** that does *not* pull in HuggingFace `transformers`, because §9.5
   requires disabling its telemetry and model-hub auto-download. A ~1.5 MB vocab/merges file plus
   ~150 lines of pure-Python BPE is the clean answer, and removes a heavyweight dependency.
4. **Parity-check** the export: cosine similarity between exported-ONNX and reference-PyTorch
   embeddings must be ≥ 0.9999 on a fixed 100-image set. Benchmarking a subtly broken export is a
   classic way to lose a week in M0.
5. Produce **fp32, fp16, and INT8-dynamic** variants. See G4 — the installer size target forces this.
6. Emit SHA-256 for each artifact into `assets/models/manifest.json` for §9.3 load-time verification.

Lives in `tools/export_models.py`, run manually, output committed to a release asset (not to git —
these are hundreds of MB).

**Phase 0 exit:** reference hardware documented, corpus A in hand and corpus B sourcing decided,
ONNX artifacts built and parity-checked, checksums recorded.

---

## 3. Milestones

### M0 — Spike (throwaway, no GUI, no CI)

The spec is right that this is first and right that it gets deleted. Its job is to answer four
questions, not three — the fourth is new:

1. Do **reads** or **inference** dominate, per drive type?
2. Is **DirectML** worth wiring up? (measure on iGPU, not a discrete card)
3. Is a **heavier CLIP model** affordable? (Q14)
4. **What does INT8 quantization cost in accuracy and buy in throughput?** (new — see G4)

**Work packages**

- `tools/bench/scan_bench.py` — a single-file CLI. Walk a folder, time each stage separately:
  read / decode / preprocess / CLIP / YuNet / SFace / SQLite write.
- Measure `Image.draft()` JPEG DCT-scaled decode against full decode against EXIF-preview
  extraction. (G16 — `draft()` is likely the winner and is three lines.)
- Sweep ONNX Runtime `intra_op_num_threads` × decode pool size. The defaults oversubscribe (§7.3.5)
  and the interaction is the whole ballgame.
- Run the matrix: {HDD, SSD} × {1, 2, 4, 8 reader threads} × {fp32, fp16, int8} × {CPU, DirectML}.
- Run the INT8 accuracy check against corpus C: does quantized CLIP change the *ranking* of results?
  (Absolute cosine scores will shift; ranking stability is what matters.)

**Exit criteria** — spec's, plus: a recommended (precision, thread-count, reader-count) tuple per
drive type, written into `docs/M0_RESULTS.md`, which survives after the code is deleted.

---

### M1 — Index (also: the open-sourcing gate)

The biggest milestone. It carries the spec's M1 scope *plus* five structural decisions that are
cheap now and expensive at M4 (G5–G11).

**Work packages**

| # | Package | Notes |
|---|---|---|
| 1.1 | Package skeleton + headless-core rule | §4 below. `core/` imports zero Qt — enforced by a test that fails on `import PySide6` inside `core`. |
| 1.2 | SQLite layer: schema, WAL, migrations | Single writer connection, `busy_timeout`, commit every N photos (tune in M0, ~50–200). Migration framework from day one — `index_version` is already in the schema and will be used. |
| 1.3 | Walker | Directory-order traversal (§7.2), not globally sorted. Windows `\\?\` long-path prefixing. Unicode/emoji safe. |
| 1.4 | Drive-type detection | WMI `MSFT_PhysicalDisk.MediaType` on Windows; conservative single-reader fallback. Behind a `platform/` port so M7 is not a refactor. |
| 1.5 | Reader thread → bounded queue → decode pool | The single highest-leverage decision (§7.3.3). `concurrent.futures.ThreadPoolExecutor`, **not** `QThreadPool` — see G8. |
| 1.6 | Inference worker | One thread, one ORT session per model, batches of 16–32. |
| 1.7 | Resumable scan state | `scan_state` transitions; row inserted at discovery, completed only after all derived data is committed. Kill -9 test. |
| 1.8 | Incremental re-scan | New / modified **/ missing** — see G5. Content hash for move detection. |
| 1.9 | Sleep inhibition | `SetThreadExecutionState(ES_SYSTEM_REQUIRED \| ES_CONTINUOUS)`, cleared on completion/crash via context manager. Never `ES_DISPLAY_REQUIRED`. |
| 1.10 | Thumbnail cache | Sharded directories by hash prefix (`ab/cd/<hash>.webp`) — 50k files in one folder is a real Windows problem. Configurable size/quality, clear-cache action. |
| 1.11 | Headless CLI | Q18 answered "yes" — it is nearly free given 1.1 and it is how you test M1 before any GUI exists. |
| 1.12 | CI: pytest + ruff + mypy + the privacy test | Privacy test (§12.3) blocks merge from the first green build. |
| 1.13 | CI: Windows installer build on every push | So M6 is a slope, not a cliff (§13). |
| 1.14 | Public-repo readiness | LICENSE ✅, README with privacy stance and biometric disclosure (§9.1), THIRD_PARTY_NOTICES (G17), CONTRIBUTING. |
| 1.15 | **Submit the code-signing application** | SignPath OSS tier. Weeks of latency; starting at M6 is the classic mistake (§11.2). |

**Exit:** 50k photos (corpus A) indexed on the primary tier; resumable across a hard kill;
incremental re-scan of 500 new photos < 2 min (N-2b); repo public and CI green.

---

### M2 — Object search

**Work packages**

- Vector store: `float32` L2-normalized at write, memory-resident, **incrementally appendable**
  during an active scan (G7). Preallocated array + doubling, plus a `photo_id ↔ row` map.
- Query path: tokenize → text tower → normalize → one matmul → argsort. Target < 200 ms at 50k (N-3).
- **Prompt templating and ensembling** (G13): encode `"a photo of a {}"` and ~6 siblings, average,
  renormalize. Applies to both `concepts.json` and free-text queries. Measurable accuracy for free.
- `concepts.json` curation pass (Q13) — ~800 concepts. This determines how the app *feels*; budget
  real time for it, not an afternoon.
- Concept matrix encoded at build time by `tools/encode_concepts.py`, shipped as `.npy` (~1.6 MB).
- **Per-concept calibration** (G14): a single global score threshold across 800 concepts produces a
  sidebar where some categories match everything and others match nothing. Compute each concept's
  score distribution over a library sample and use a percentile cutoff.
- UI: thumbnail grid with virtualized scrolling, threshold slider with live count (F-2.3),
  category sidebar with counts (F-2.4).
- Cheap re-tag pass: recompute concepts from stored embeddings with no image decode (§5.2).

**Exit:** spec's — "mountain" and "house" work by typing *and* by clicking, at 50k, in < 200 ms.

---

### M3 — Faces (highest technical risk)

**Work packages**

- YuNet detect → 5-point align → SFace embed, in the existing inference worker.
- Aggressive prefilter **before** clustering: drop faces < 40×40 px and below a detector-confidence
  floor (§5.3). This is the highest-value knob for cluster quality.
- Blocked similarity: 1024-row blocks, not 4096 — a 4096×75000 float32 block is 1.2 GB and blows
  N-5's 3 GB ceiling on its own.
- **Clustering with chaining protection** (G12 — read this one carefully). Plain union-find over a
  thresholded similarity graph collapses into one giant blob at consumer-photo quality. Mitigate
  with: (a) **mutual k-NN edges** rather than raw threshold edges, (b) a strict clustering threshold
  (~0.45) decoupled from the looser matching threshold, (c) a cluster-size sanity cap that flags
  suspiciously large clusters for split rather than presenting them as a person.
  Budget a Chinese-Whispers fallback if union-find will not behave.
- Clustering is a **progress-bar job, not an instant one** (G15): ~1 minute at 75k on the primary
  tier, not "a few seconds".
- Person model: store **every** enrollment embedding, match on **max** similarity, never average (§5.3).
- Cluster-first naming UI (F-3.1), enrollment (F-3.2/3.3), merge/split (F-3.5), delete (F-3.9),
  ignore (F-3.10). Q16 (wizard vs. permanent People tab) is decided here — a permanent tab that
  *opens itself* after the first scan gets both.
- Plain-language accuracy limitation notice in the UI (§5.3).

**Exit:** 75k faces cluster within the RAM ceiling; the top 20–30 clusters are coherent enough that
a tester names them without hesitation; precision/recall recorded against corpus C at all three
thresholds.

---

### M4 — Actions

Viewer (F-4.1–4.5), OS handoff (F-5.1–5.3), export (F-6.1–6.8). Lowest-risk milestone.

Two notes: the export collision/cleanup logic (F-6.3/F-6.5) is where the bugs live — write the
tests first. And put all three OS-handoff commands behind `platform/reveal.py` from the start; M7
then adds two functions rather than hunting `os.startfile` calls through the UI.

---

### M5 — Polish

Thresholds framed as "more results / fewer mistakes" (F-3.7), merge/split refinement, corrections
feedback (F-3.8), every failure path surfaced as a GUI dialog with copy-details (§11.5), rotating
logs, first-run explainer, delete-all-data actions (§9.4).

**Exit:** the spec's — a non-technical tester completes setup and finds a person unaided. Do this
with a real person, watching silently, taking notes. It is the single most informative hour in the
project.

---

### M6 — Ship

PyInstaller `--onedir` + Inno Setup, per-user install, x64 only. Signing certificate should already
be in hand from M1.15. VirusTotal submission and false-positive reports (§11.3). Clean-VM test with
no Python present and networking disabled.

**Note:** N-6's original < 250 MB target has been relaxed to < 5 GB (owner decision, see G4), which
fp32 or fp16 weights clear comfortably. This is no longer a watch item.

---

### M7 — Ports

macOS then Linux. Note the cost the spec does not: macOS distribution needs an **Apple Developer ID
($99/yr) and notarization**, with its own lead time and its own CI plumbing. §11.2 covers Windows
signing only.

---

## 4. Package layout

The F-1.10 architecture note ("headless worker, zero Qt widget dependencies") is the highest-value
structural rule in the spec. Encode it as a directory boundary and a test, not as a good intention.

```
aiphotoorg/
  core/                  # ZERO Qt imports — enforced by test_no_qt_in_core.py
    db/          schema.py  migrations/  repo.py
    scan/        walker.py  reader.py  decoder.py  pipeline.py  state.py
    ml/          session.py  clip.py  tokenizer.py  faces.py  concepts.py  quantized.py
    search/      vector_store.py  query.py
    people/      cluster.py  match.py  persons.py
    export/      copier.py
    platform/    sleep.py  reveal.py  drivetype.py  paths.py     # all OS branching lives here
    events.py                                                    # plain callbacks, no Qt signals
  ui/                    # the ONLY package that imports PySide6
    adapters.py                                                  # core callbacks -> Qt signals
    grid/  viewer/  people/  settings/
  cli/                   # headless entry point (Q18)
tools/         export_models.py  encode_concepts.py  bench/
assets/        concepts.json  concept_matrix.npy  models/manifest.json
tests/         unit/  integration/  privacy/  data/README.md
docs/          BUILD_PLAN.md  REFERENCE_HARDWARE.md  M0_RESULTS.md
```

`core/events.py` is the piece people skip. The core emits plain Python callbacks; `ui/adapters.py`
translates them into Qt signals. Tray mode (F-1.10) and the CLI (Q18) then cost nothing.

---

## 5. Cross-cutting decisions to lock before M1 code

These are baked into stored data or into the module graph. Changing them later means a re-index or
a refactor.

| Decision | Recommendation | Why now |
|---|---|---|
| Embedding precision on disk | `float32`, L2-normalized at write | Normalizing at write makes every query a pure matmul. fp16 halves storage (100→50 MB) at negligible cosine error — worth measuring in M0, but pick one and stop. |
| `model_version` format | `clip-vitb32-laion2b-int8-v1` — must encode **quantization** | An fp32-indexed library and an int8-indexed library are not interchangeable. If quantization is not in the version string, a future change silently corrupts search. |
| SQLite journal mode | WAL, single writer connection, `busy_timeout=5000` | F-1.6 (search during scan) is impossible without it. |
| Commit batching | Every N completed photos, N tuned in M0 | Per-photo commits will dominate scan time on an HDD. |
| Thumbnail store | Sharded dirs `ab/cd/<hash>.webp` | 50k files in one directory is a Windows-specific performance cliff. |
| Threading primitive in core | `concurrent.futures`, never `QThreadPool` | §7.3.4 says QThreadPool; F-1.10 says zero Qt in the scanner. F-1.10 wins. |
| Missing-file semantics | Soft-delete: mark `missing_since`, never hard-delete | D-8 means "file gone" and "drive unplugged" look identical. |
| Prompt template | `"a photo of a {}"` + ensemble, applied to both concepts and queries | Must be identical at build time and query time or the spaces do not line up. |

---

## 6. Gap register

What the spec does not cover, ranked by when it bites. IDs are stable; reference them in issues.

### 6.1 Blocking — resolve in Phase 0 / M0

**G1 — No reference hardware is defined.**
§7.1 says "a 4-core desktop"; N-2 says "≥1.7 img/s, target ≥6". With no named machine, N-2 cannot
pass or fail. → P0.1.

*Update (2026-09-11):* tested how far a disposable cloud sandbox substitutes for this. It can catch
methodology mistakes (see the FLOP-proxy finding in P0.1) and validate OS-independent logic (see
G16's decode finding below), but cannot stand in for P0.1 itself — no Windows, no DirectML, no real
HDD/SSD I/O characteristics, shared/virtualized CPU. P0.1 still needs to happen on named, real
hardware.

**G2 — The test corpus does not exist and has the longest lead time in the project.**
§12.4 asks for a real 50k library "early" and §12.1 for a labeled set, but neither is a milestone
deliverable and neither has an owner or a source. The face corpus in particular (repeat identities,
consumer conditions, ethically sourced) is the hard one, and it gates M3 and every threshold number
in §5.3. → P0.2.

*Status (2026-09-11): deprioritized by owner decision.* Testable locally against a personal library
as the project matures rather than sourced up front; the project doesn't need to be accuracy-perfect
from the start. Demoted out of the Phase 0 blocking set — it still gates M3's threshold work and the
labeled accuracy set in §12.1, so it isn't gone, just no longer on the critical path to starting.

**G3 — Producing the ONNX artifacts is unplanned work.**
The spec treats "OpenCLIP ViT-B/32, ONNX" as a given. In practice: two exported graphs (image
*and* text towers), a CLIP BPE tokenizer implemented without `transformers` (which §9.5 requires
you to defang anyway), a parity check against reference PyTorch, and checksums for §9.3. → P0.3.

**G4 — N-6 (installer < 250 MB) was unreachable without INT8 quantization, which the spec never mentioned.**
OpenCLIP ViT-B/32 is ~151M params across both towers:

| Precision | Models on disk | + runtime (Python, Qt, ORT, NumPy, Pillow) | Verdict vs. original N-6 (250 MB) |
|---|---|---|---|
| fp32 | ~605 MB | ~755 MB | impossible |
| fp16 | ~302 MB | ~450 MB | fails |
| int8 dynamic | ~155 MB | ~340 MB uncompressed → ~200–250 MB installer | tight but plausible |

(SFace adds 37 MB, YuNet 0.34 MB, concept matrix 1.6 MB in all cases.)

*Status (2026-09-11): RESOLVED by owner decision — N-6 relaxed from < 250 MB to < 5 GB.* This removes
the forcing function entirely: fp32 (~755 MB installed) and fp16 (~450 MB installed) both clear the
new budget with room to spare, even alongside the optional non-commercial `buffalo_l` face model or
a larger CLIP variant.

**Revised recommendation:** default to **fp16** — it halves size and RAM versus fp32 at negligible
accuracy cost, with no real downside, so there's no reason not to take it. **INT8 downgrades from a
size requirement to a throughput experiment.** It's still worth measuring in M0 (it plausibly doubles
CPU throughput, the cheapest path to N-2's ≥6 img/s target), but it no longer gates whether the app
can ship, and no longer needs an accuracy validation pass before a decision can be made either way.
`model_version` must still record precision (`clip-vitb32-laion2b-fp16-v1`, etc.) — an fp16-indexed
and int8-indexed library remain non-interchangeable regardless of which is the default.

### 6.2 Structural — decide at M1, expensive afterwards

**G5 — Deleted and moved files are absent from F-1.5.**
"Only new or modified files" says nothing about a photo that no longer exists. Under D-8 (external
drives) you cannot distinguish "deleted" from "drive unplugged", so hard-deleting rows is wrong. Add
a soft-delete (`missing_since`) requirement, and promote F-1.8 content hashing from *Could* to
*Should* so a moved file is recognized as a move instead of being re-embedded as a new photo.

**G6 — SQLite concurrency model is unspecified, and F-1.6 depends on it.**
WAL mode, one writer connection, `busy_timeout`, and batched commits. Also: per-photo commits are a
throughput trap on HDD.

**G7 — §6.3 says "load embeddings into RAM once at startup", but F-1.6 requires searching a library that is actively growing.**
The vector store must be incrementally appendable mid-scan. Re-reading from SQLite per query blows
N-3. Preallocate and double, keep a `photo_id ↔ row` map.

**G8 — §7.3.4 and the F-1.10 note contradict each other.**
"Decode in a `QThreadPool`" vs. "keep all scan logic in a headless worker with zero Qt dependencies".
Resolve in favor of F-1.10: `concurrent.futures` in `core/`. This also makes Q18's CLI free.

**G9 — Thumbnail cache layout is sized (~500 MB) but not designed.**
50k WebP files in one directory is a Windows performance cliff. Shard by hash prefix.

**G10 — No migration strategy.**
`index_version` and `model_version` exist in the schema but nothing says what happens when they
change. Needs: a migration runner at M1, and a defined UX for "CLIP model changed → full re-index"
(§7.1 flags the pain but proposes no handling).

**G11 — Platform-specific code needs a port boundary from M1, not at M7.**
Sleep inhibition (F-1.9) is a Windows API; macOS needs `IOPMAssertionCreateWithName`, Linux needs
`systemd-inhibit`/D-Bus. Same for reveal-in-folder and drive-type detection. One `core/platform/`
package now makes M7 additive.

### 6.3 Accuracy and ML

**G12 — Union-find over a thresholded graph will chain, and this is the top M3 risk.**
If A≈B and B≈C but A≉C, union-find merges all three. Across 75k consumer-quality faces this reliably
produces one enormous junk cluster containing several real people. §5.3 names Chinese Whispers as
"the same family" — it is not, it is meaningfully more chaining-resistant. Mitigations: mutual k-NN
edges, a strict clustering threshold decoupled from the matching threshold, and a size-based
sanity check. Plan for the fallback rather than discovering this at M3.

**G13 — No prompt templating for the CLIP text tower.**
Encoding the bare word `"mountain"` measurably underperforms `"a photo of a mountain"`, and an
ensemble of ~7 templates is better still. Applies to `concepts.json` and to free-text queries, and
must be identical in both or the spaces do not align. Near-zero cost, real accuracy.

**G14 — The concept sidebar needs per-concept calibration.**
§5.1 correctly notes CLIP scores are not probabilities — but §5.2 then stores "top 10 concepts above
threshold" using what appears to be a single global threshold. Across 800 concepts that yields a
sidebar where some categories match half the library and others are permanently empty. Since F-2.4
is a *Must* and the sidebar is the app's first impression, calibrate per concept against a sample of
the user's own library.

**G15 — The clustering cost estimate is optimistic by roughly an order of magnitude.**
§5.3 calls 75k×75k at 128 dims "a few seconds of NumPy". It is ~1.4 TFLOP plus ~22 GB of streamed
intermediates — realistically 30–90 s on the primary tier. Fine as a job; wrong as a UI assumption.
Give it a progress bar and a cancel. Also, the block size must be ~1024 rows, not 4096, to respect
N-5's 3 GB ceiling.

**G16 — `Image.draft()` is a simpler version of §7.3.1, but "no correctness caveats" was wrong — it has one, and the win is smaller than claimed.**
Extracting the embedded EXIF preview means handling missing previews, mismatched orientation,
different crops, and a "is it big enough for face detection" rule. Pillow's `draft()` does DCT-domain
JPEG downscaling to 1/2, 1/4, or 1/8 during decode, which avoided those caveats on paper.

*Update (2026-09-11): measured instead of estimated, and it surfaced a real bug.* `draft()` requires
**both** scaled dimensions to meet the requested target. Calling `im.draft("RGB", (1600, 1600))` — a
square target, the natural way to write it — against a 4032×3024 (4:3) source silently returns the
full-size image with **zero speedup**: 1/2 scale gives 2016×1512, and 1512 < 1600 fails the check, so
`draft()` declines to scale at all. No error, no warning. The fix is to scale the target by the
image's own long edge:

```python
scale = long_edge / max(im.size)
im.draft("RGB", (int(im.width * scale), int(im.height * scale)))
```

Once fixed, measured speedups (synthetic 4032×3024 JPEG, random-noise pixel content) were **1.17×–1.34×**
across 1/2 to 1/8 scale — well short of the "similar win" this gap originally claimed. Random noise is
close to a worst case for JPEG (near-incompressible, so entropy decode dominates regardless of output
scale); real photographic content, being far more compressible, should show a larger win since the
IDCT/upsample stage `draft()` actually shrinks is proportionally more of total decode time — but that
needs a real photo corpus to confirm (→ G2), and is exactly the kind of "estimated, not measured" trap
§7.1 already warns about. Keep `draft()` as the M0 candidate — it is still simpler than preview
extraction and the win is real — but validate the actual speedup and ship the aspect-ratio-correct
helper above rather than the naive call.

### 6.4 Legal and distribution

**G17 — §3 audits model licenses thoroughly but never audits dependency licenses.**
The stack carries real obligations: **PySide6 is LGPL** (satisfiable with `--onedir` dynamic
linking, but you must ship the LGPL text and preserve the relink right); **pillow-heif** bundles
libheif/libde265, which have their own terms; and HEVC **patent** licensing is a known gray area
that decode-only OSS tools generally live with but should do knowingly. Deliverable: a
`THIRD_PARTY_NOTICES.md` generated at build time and a one-off audit before the repo goes public at
M1. Being scrupulous about model licensing and casual about Qt's would be an odd place to trip.

**G18 — M7's macOS port needs Apple Developer ID + notarization.**
§11.2 budgets Windows signing only. macOS adds $99/yr, an enrollment wait, and notarization
plumbing in CI. Same "start early" logic as SignPath.

### 6.5 Process

**G19 — No test tooling or quality gates are named.**
§12 describes *what* to test but not *with what*. Recommend pytest + ruff + mypy from M1, and make
§12.3's socket-blocked privacy test a **merge-blocking CI job** — including at *import* time, since
HuggingFace-family packages phone home on import (§9.5), not just on use.

**G20 — There is no effort budget.**
See §7. A locked spec with no timeline tends to produce a stalled project rather than a late one.

---

## 7. Effort estimate

Focused-work weeks, solo, for a developer comfortable with Python but new to ONNX packaging:

| Phase | Weeks | Risk |
|---|---|---|
| Phase 0 | 1–2 | Corpus B (G2) deprioritized — still needed by M3, no longer gates Phase 0 |
| M0 | 1–2 | Low — throwaway by design |
| M1 | 4–6 | Medium — carries the structural decisions |
| M2 | 3–4 | Medium — concept curation (Q13) is open-ended |
| M3 | 4–6 | **High** — clustering quality (G12) |
| M4 | 2–3 | Low |
| M5 | 3–4 | Medium — usability testing may force rework |
| M6 | 3–5 | Medium — signing and AV latency are calendar, not effort |
| **v1 total (→ M6)** | **21–32** | |
| M7 | 4–6 | |

At ~12 h/week that is **9–15 months** calendar to a signed Windows v1; full-time, **5–8 months**.
The two items that most reduce calendar risk are both Phase 0: sourcing corpus B, and filing the
signing application at M1 rather than M6.

---

## 8. New open questions

Extending §14's Q13–Q18.

| # | Question | Decide by |
|---|---|---|
| Q19 | INT8 vs fp16 CLIP — now a throughput-only question (N-6 relaxed, see G4): is INT8's speedup worth its accuracy cost, given fp16 already fits the size budget? | M0 |
| Q20 | Where does face corpus B come from, and under what consent? (G2 — deprioritized, decide whenever M3 approaches) | M3 |
| Q21 | Thumbnail store: sharded files on disk, or blobs in SQLite? (G9) | M1 |
| Q22 | Do sidebar counts follow the F-2.3 threshold slider, or use a fixed per-concept calibration? (G14) | M2 |
| Q23 | On re-scan, what is shown for photos whose files are gone — and how is that distinguished from an unplugged drive? (G5) | M1 |
| Q24 | Does the M1 CLI become a supported user-facing surface, or stay an internal test harness? (Q18's follow-on) | M5 |
| ~~Q25~~ | ~~If N-6 cannot be met at INT8, relax the size target or drop to a smaller CLIP variant?~~ **Resolved 2026-09-11** — N-6 relaxed to < 5 GB; fp16 clears it without quantization. | — |

---

## 9. Immediate next actions

1. Write `docs/REFERENCE_HARDWARE.md` (P0.1) — one hour, unblocks every performance number.
2. Decide face corpus provenance (P0.2/Q20) — the long pole; start today.
3. Build `tools/export_models.py`, produce fp32/fp16/int8 artifacts, parity-check (P0.3).
4. Write `tools/bench/scan_bench.py` and run the M0 matrix on all three tiers.
5. Record results in `docs/M0_RESULTS.md`, answer Q14/Q19/Q25, **then delete the spike.**

No GUI code until step 5 is done. The spec is right about that.
