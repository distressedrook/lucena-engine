<p align="center">
  <img src="https://raw.githubusercontent.com/distressedrook/lucena-engine/main/assets/logo.png" width="116" alt="lucena-engine">
</p>

# lucena-engine

[![license](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![core](https://img.shields.io/badge/board%20core-Rust-orange.svg)](rust/)
[![engine](https://img.shields.io/badge/engine-Stockfish%2018-informational.svg)](https://stockfishchess.org/)
[![maia](https://img.shields.io/badge/behaviour-Maia-9cf.svg)](https://github.com/CSSLab/maia3)
[![grpc](https://img.shields.io/badge/API-Python%20%7C%20gRPC-success.svg)](proto/)

**Deterministic chess truth for grounding language models.**

`lucena-engine` turns a chess position into *grounded facts* — a piece list, a material count, a
strategic read, tactical motifs, move evaluations, and a plain-language briefing — computed by
Stockfish and a Rust board core, never guessed. It exists so an LLM can **interpret** chess instead of
**calculating** it: you hand the model the facts, and it explains, teaches, or narrates — without
hallucinating whose move it is, miscounting material, inventing a piece on f6, or imagining a fork that
isn't there.

If you're building anything where an LLM talks about chess positions — a coach, an annotator, a
commentary bot, a puzzle explainer — this is the layer that keeps it honest.

---

## Why

LLMs are fluent about chess and quietly wrong. Ask one "what's going on here?" with a FEN and it will
confidently misread the side to move, hang a piece in analysis, or narrate a tactic from a different
game. The fix isn't a bigger prompt — it's **grounding**: compute everything the model could get wrong,
hand it over as facts, and instruct it to reason only from those.

`lucena-engine` is the grounding source. It emits two shapes of truth:

- **Structured facts** — typed records with squares, scores, and provenance (for your own logic, board
  arrows, or tool-calling).
- **A natural-language fact sheet** — a short list of deterministic English statements you can drop
  straight into a prompt.

Every output is reproducible given the engine limit. No randomness in the facts.

---

## Install

```bash
pip install lucena-engine        # library + `lucena-engine` CLI
brew install stockfish           # required engine (or apt-get install stockfish)

lucena-engine doctor             # check the runtime is wired up
lucena-engine analyze "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

Optional extras: `pip install 'lucena-engine[grpc]'` for the server; human-move prediction (Maia) is
opt-in via `lucena-engine setup-maia` — see [Human-move prediction](#human-move-prediction-maia).

**From source** (development): needs a **Rust toolchain**.
```bash
pip install maturin && maturin develop --release   # builds the Rust board core
export LUCENA_STOCKFISH=$(which stockfish)          # defaults to `stockfish` on PATH
```

---

## Quickstart — a grounded fact sheet in five lines

```python
from lucena_engine import Engine
from lucena_engine.board import Board
from lucena_engine.analysis import build_analysis

engine = Engine()                                          # spawns Stockfish
board  = Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/2N2N2/PPPP1PPP/R1B1K2R b KQkq - 0 4")

for line in build_analysis(board, engine, movetime_ms=500):
    print(line)
```

```
The position is roughly equal.
Material is equal.
White's pieces are more active than Black's.
The center is contested.
Tactics: the knight on f6 can capture on e4.
```

That list *is* the grounding. Put it in a prompt:

```python
facts = build_analysis(board, engine, movetime_ms=500)
prompt = (
    "You are a chess coach. Ground every claim ONLY in these engine facts — "
    "never invent a piece, square, line, or number:\n" + "\n".join(facts) +
    f"\n\nStudent asked: what should I be thinking about here?"
)
# → hand `prompt` to your LLM. It interprets; it does not calculate.
```

---

## What's in the fact sheet

`build_analysis` composes, in order:

1. **Evaluation** — translated to words ("roughly equal", "White is clearly better"); never raw
   centipawns or win%.
2. **Material** — a standing ("Black is up a pawn").
3. **Strategic read** — king safety, piece activity, pawns, and center, leads first.
4. **Tactics** — the salient motifs on the board, joined into one line.

Each piece is also available on its own if you'd rather build your own briefing:

```python
from lucena_engine import build_fact_sheet
from lucena_engine.analysis import analyze_positional
from lucena_engine.reads import material, piece_list, eval_block, pv_san

analysis = engine.analyse(board.fen, multipv=2, movetime_ms=500)
material(board)                    # {"white": 10, "black": 10, "net": 0, "standing": "Material is equal"}
eval_block(analysis.best.score)    # {"cp": 12, "win_pct": 52.1, ...}
pv_san(board.fen, analysis.best.pv)  # ["Nxe4", "d5", "Bd3", ...]  (UCI → SAN)
analyze_positional(board)          # the five strategic terms, each with a "standing"
```

---

## Structured facts (typed, with squares)

For tool-calling, board arrows, or your own routing, take the facts as records instead of prose:

```python
from lucena_engine import build_fact_sheet

for f in build_fact_sheet(board, engine, movetime_ms=500):
    print(f.kind, f.squares, f.salience, "—", f.text)
# hanging   ['e4']   0.9  — the pawn on e4 is undefended
# fork      ['c7']   0.8  — a knight on c7 would fork the king and rook
```

`Fact` = `kind`, `squares`, `text`, `provenance`, `salience` (0..1). The individual detectors are
public too:

```python
from lucena_engine import (
    detect_hanging, detect_fork, detect_pin, detect_defender_removed,
    detect_combination, detect_null_move_threat,
)
```

---

## The board core

A fast Rust board (cozy-chess + a custom SAN encoder and SEE) with a small Python surface — legal
moves, SAN ⇄ UCI, static exchange evaluation, attackers/defenders, apply:

```python
from lucena_engine.board import Board

b = Board(fen)
b.side_to_move        # "white" | "black"
b.legal_moves()       # ["e2e4", "g1f3", ...] (UCI)
b.san("f6e4")         # "Nxe4"
b.uci("Nxe4")         # "f6e4"
b.see("f6e4")         # static exchange eval (centipawns) — is this capture safe?
b.attackers("e4", "white")   # squares of white pieces hitting e4
b.apply("f6e4")       # → a new Board after the move (immutable)
```

Nothing here calls the engine — it's pure, instant board logic.

---

## Move evaluation, hints, and the rest

```python
# Best move + a worded verdict for a candidate
from lucena_engine import classify, Glyph
before = engine.analyse(fen, multipv=2, movetime_ms=500)
after  = engine.analyse(board.apply("f6e4").fen, movetime_ms=500)
glyph, _ = classify(before.best.score, after.best.score.negated())   # OK / dubious / mistake / blunder …

# A ladder of hints, vague → specific, none of which is the answer
from lucena_engine import derive_hints
# Brilliancy / sacrifice detection
from lucena_engine import is_brilliant, is_material_sacrifice
# Whole-game annotation
from lucena_engine import parse_pgn, run_pass
game = parse_pgn(pgn_text)          # Game(plies=[Ply(san, uci, fen), ...])
```

---

## Human-move prediction (Maia)

Optional. Predicts what a human of a given rating would *actually* play (not the engine's best) — useful
for "this is a common trap at your level" style grounding. Runs as its own subprocess.

```python
from lucena_engine.maia import MaiaEngine   # requires LUCENA_MAIA to point at a maia UCI wrapper

maia = MaiaEngine()
maia.top_human_moves(fen, rating=1500, n=5)   # [{"uci": "f6e4", "rank": 1, "policy": 0.42}, ...]
```

---

## Over the wire (gRPC)

Prefer a language-agnostic or networked boundary? The same truth is exposed as a gRPC service — handy
if your app isn't Python, or if you want the engine (and its GPL dependencies) in a separate process.

```bash
python -m lucena_engine.server.serve          # listens on :50051 (LUCENA_ENGINE_PORT)
```

Two services (proto in `proto/lucena/engine/v1/engine.proto`, **SAN on the wire**):

- **Truth** — `Analyze` (FEN → fact sheet), `Evaluate`, `Hints`, `LegalMoves`, `Apply`, `ValidateFen`,
  `DetectFens`, `ParsePgn`, `GetInfo`.
- **Behaviour** — `TopHumanMoves`, `PoisonedLine` (Maia-backed; optional).

---

## Design principles

- **Calculate vs. interpret.** The engine and board *calculate* the truth; your LLM only *interprets*
  it. Anything the model could get wrong is grounded (handed over) or guarded (made structurally
  impossible) — never left to the model.
- **Deterministic.** Same position + same engine limit → same facts. Production uses `movetime_ms`;
  tests use `nodes` with `threads=1` for bit-reproducibility.
- **SAN on the wire.** UCI is engine-internal; everything you see and send is human-readable SAN.
- **No eval leakage into prose.** The natural-language layer says "you're winning", never "‑347 cp".
- **Copyleft hygiene.** GPL engines (Stockfish, Maia) are invoked only as **subprocesses over UCI** —
  arm's-length, never linked. A CI gate keeps GPL libraries out of the shipped code.

---

## Requirements

- Python ≥ 3.9, Rust toolchain (for the board core via maturin).
- Stockfish (18 recommended; the NNUE eval parser is version-aware). Point `LUCENA_STOCKFISH` at it.
- Optional: Maia (`maia3`) in its own environment, via `LUCENA_MAIA`.

---

## Contributing & license

Licensed under **AGPL-3.0-or-later** (see `LICENSE`, `NOTICE`, and `THIRD-PARTY-NOTICES.md`). If you run
a modified version as a network service, AGPL §13 requires you to offer users its source.

Contributions are welcome under the CLA in `CONTRIBUTING.md`. Note the copyleft-hygiene rule: the
shipped library never imports `python-chess` (GPL) — the board core replaces it, and a test enforces it.
