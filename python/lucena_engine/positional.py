"""Deterministic positional analysis — the five classical terms, in centipawns.

Grounds the coach's opening "analysis" beat: material, king safety, piece
activity, pawn structure and centre control, each computed exactly from board
geometry (no engine call, no LLM) with constants adopted from published
standards, never invented:

- piece-square tables + material values: PeSTO (R. Friederich), machine-
  extracted into `_pesto.py` by tools/gen_pesto.py — 768 constants, never
  hand-typed;
- king safety: the CPW attack-units -> safety-table model (Glaurung 1.2 table;
  zone = king ring + three forward squares; units N/B=2 R=3 Q=5);
- mobility / pawn-structure weights: CPW reference values, simplified where
  noted.

These terms are SALIENCE, never a verdict (the LLD §2.3 hierarchy applies
unchanged): they are never summed into an eval and never adjudicate who is
better — Stockfish owns the verdict, and a term that contradicts the engine
loses. The terms decompose *why*, so the coach leads with the dimension that
actually characterises the position and cites exact features instead of
eyeballing the board. All cp values are White-POV; `standing` strings are
plain sentences the coach reads verbatim (no raw numbers — contract rule).

Known honest limitation: static king safety is a *middlegame pawn-shield*
model. It tapers out in endgames, where a concrete mating attack (a king
caught on h5) is tactical, not positional — that danger is the null-move
threat probe's job, not this module's. The two are complementary by design.
"""

from __future__ import annotations

from ._pesto import EG_TABLE, EG_VALUE, MG_TABLE, MG_VALUE
from .detectors._util import PIECE_NAME, king_square, occupancy

# --- CPW king-safety model (Glaurung 1.2 safety table, verbatim) ------------
SAFETY_TABLE = [
    0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 36, 42, 48,
    55, 62, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
    200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330,
    340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470,
    480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610,
    620, 630, 640, 650, 650, 650, 650, 650, 650, 650, 650, 650, 650, 650,
    650, 650, 650, 650, 650, 650, 650, 650, 650,
]
_ATTACK_UNITS = {"N": 2, "B": 2, "R": 3, "Q": 5}   # per attacked zone square

# game-phase increments (PeSTO): minors 1, rooks 2, queens 4; 24 = full MG
_PHASE_INC = {"N": 1, "B": 1, "R": 2, "Q": 4}

# simplified CPW-style mobility: cp per square above/below a typical count
_MOBILITY = {"N": (4, 4), "B": (6, 3), "R": (7, 2), "Q": (13, 1)}  # (baseline, weight)

# pawn-structure weights (mg, eg), simplified CPW reference values
_ISOLATED = (-15, -12)          # per isolated pawn
_DOUBLED = (-10, -20)           # per extra pawn on a file
_ISLAND = (-8, -8)              # per island beyond the first
_PASSED = [(0, 0), (5, 15), (12, 25), (20, 40), (35, 65), (60, 105),
           (100, 160), (0, 0)]  # by relative rank (0 = home, 6 = about to promote)

# centre model: occupation + attack differential on the four core squares
_CORE = ("d4", "e4", "d5", "e5")
_OCCUPY = {"P": (18, 6), "N": (8, 4), "B": (8, 4), "R": (4, 2), "Q": (4, 2)}
_CONTROL = (7, 3)               # cp per net attacker on a core square

_FILES = "abcdefgh"
_LEAD_THRESHOLD = 25            # |cp| for a term to lead the analysis


def _fr(square: str) -> tuple[int, int]:
    return _FILES.index(square[0]), int(square[1]) - 1


def _sq(f: int, r: int) -> str:
    return _FILES[f] + str(r + 1)


def _on(f: int, r: int) -> bool:
    return 0 <= f < 8 and 0 <= r < 8


def _pst(table: dict, piece: str, color: str, square: str) -> int:
    """PeSTO tables are published a8-first (White's visual view); Black mirrors
    the rank."""
    f, r = _fr(square)
    idx = (7 - r) * 8 + f if color == "white" else r * 8 + f
    return table[piece][idx]


def _taper(mg: float, eg: float, phase: float) -> float:
    return mg * phase + eg * (1.0 - phase)


def _side_name(color: str) -> str:
    return "White" if color == "white" else "Black"


# --- shared geometry ---------------------------------------------------------

def _attack_maps(board):
    """One pass over all 64 squares: per-square attacker counts and, inverted,
    per-piece attacked-square sets. `board.attackers` is the only primitive."""
    ctrl: dict[str, tuple[int, int]] = {}
    attacks: dict[str, set[str]] = {}
    for f in range(8):
        for r in range(8):
            s = _sq(f, r)
            w = board.attackers(s, "white")
            b = board.attackers(s, "black")
            ctrl[s] = (len(w), len(b))
            for a in w + b:
                attacks.setdefault(a, set()).add(s)
    return ctrl, attacks


def _phase(occ) -> float:
    gp = sum(_PHASE_INC.get(p.piece, 0) for p in occ.values())
    return min(gp, 24) / 24.0


# --- the five terms ----------------------------------------------------------

def _material_term(board) -> dict:
    """Tapered PeSTO material in cp; the standing sentence is the exact-imbalance
    phrase (single source of truth in mcp.response — lazy import, noted layering
    wart until material() moves engine-side)."""
    from .reads import material as _material

    occ = occupancy(board)
    ph = _phase(occ)
    cp = 0.0
    for p in occ.values():
        v = _taper(MG_VALUE[p.piece], EG_VALUE[p.piece], ph)
        cp += v if p.color == "white" else -v
    return {"cp": round(cp), "standing": _material(board)["standing"]}


def _king_zone(ksq: str, color: str) -> list[str]:
    """CPW zone: the king ring plus three squares two ranks toward the enemy."""
    kf, kr = _fr(ksq)
    zone = [_sq(kf + df, kr + dr) for df in (-1, 0, 1) for dr in (-1, 0, 1)
            if (df or dr) and _on(kf + df, kr + dr)]
    fwd = 2 if color == "white" else -2
    zone += [_sq(kf + df, kr + fwd) for df in (-1, 0, 1) if _on(kf + df, kr + fwd)]
    return zone


def _king_safety_term(board, occ, ph: float) -> dict:
    features = {}
    danger = {}
    pst = {}
    for color in ("white", "black"):
        enemy = "black" if color == "white" else "white"
        ksq = king_square(board, color)
        if ksq is None:                          # analysis boards can be kingless
            danger[color], pst[color] = 0.0, 0.0
            continue
        kf, kr = _fr(ksq)
        pst[color] = _taper(_pst(MG_TABLE, "K", color, ksq),
                            _pst(EG_TABLE, "K", color, ksq), ph)

        # attack units: enemy N/B/R/Q attacks into the zone (CPW weights)
        zone = _king_zone(ksq, color)
        units, attackers = 0, {}
        for zs in zone:
            for a in board.attackers(zs, enemy):
                p = occ.get(a)
                if p is not None and p.piece in _ATTACK_UNITS:
                    units += _ATTACK_UNITS[p.piece]
                    attackers[a] = p.piece
        zone_penalty = SAFETY_TABLE[min(units, 99)] if len(attackers) >= 2 else 0

        # pawn shield: the three files around the king, one/two steps ahead
        fwd = 1 if color == "white" else -1
        shield, shield_penalty = 0, 0
        for df in (-1, 0, 1):
            if not _on(kf + df, kr + fwd):
                continue
            p1 = occ.get(_sq(kf + df, kr + fwd))
            p2 = occ.get(_sq(kf + df, kr + 2 * fwd)) if _on(kf + df, kr + 2 * fwd) else None
            if p1 is not None and p1.color == color and p1.piece == "P":
                shield += 1
            elif p2 is not None and p2.color == color and p2.piece == "P":
                shield += 1
                shield_penalty += 6              # advanced one step: half airy
            else:
                shield_penalty += 14             # no pawn on this file at all

        # open / half-open files beside the king (only if enemy heavy pieces exist)
        own_pawn_files = {_fr(s)[0] for s, p in occ.items()
                          if p.color == color and p.piece == "P"}
        enemy_pawn_files = {_fr(s)[0] for s, p in occ.items()
                            if p.color == enemy and p.piece == "P"}
        heavy = any(p.piece in ("R", "Q") and p.color == enemy for p in occ.values())
        open_files, file_penalty = [], 0
        if heavy:
            for df in (-1, 0, 1):
                f = kf + df
                if not _on(f, 0) or f in own_pawn_files:
                    continue
                open_files.append(_FILES[f])
                file_penalty += 15 if f not in enemy_pawn_files else 10

        # shield/file penalties are middlegame concepts; zone attacks keep a
        # small endgame tail (a cornered king can still be hunted)
        danger[color] = _taper(zone_penalty + shield_penalty + file_penalty,
                               zone_penalty / 4.0, ph)
        features[color] = {
            "king": ksq,
            "zone_attackers": sorted(f"{PIECE_NAME.get(pc, pc)} on {s}"
                                     for s, pc in attackers.items()),
            "attack_units": units,
            "shield_pawns": shield,
            "open_files_nearby": open_files,
        }

    cp = (danger["black"] - danger["white"]) + (pst["white"] - pst["black"])
    worse = max(danger, key=lambda c: danger[c])
    d = danger[worse]
    if d >= 100:
        ft = features[worse]
        bits = [f"{len(ft['zone_attackers'])} enemy pieces attack the squares around it"]
        if ft["open_files_nearby"]:
            bits.append(f"the {'/'.join(ft['open_files_nearby'])}-file beside it is open")
        if ft["shield_pawns"] == 0:
            bits.append("it has no pawn shield")
        standing = f"{_side_name(worse)}'s king is in danger — " + ", and ".join(bits)
    elif d >= 40:
        standing = f"{_side_name(worse)}'s king is a little exposed"
    else:
        standing = "both kings are reasonably safe"
    return {"cp": round(cp), "standing": standing, "features": features}


def _activity_term(board, occ, attacks, ph: float) -> dict:
    sums = {"white": 0.0, "black": 0.0}
    worst = {"white": None, "black": None}
    for s, p in occ.items():
        if p.piece not in _MOBILITY:             # kings/pawns live in other terms
            continue
        place = _taper(_pst(MG_TABLE, p.piece, p.color, s),
                       _pst(EG_TABLE, p.piece, p.color, s), ph)
        base, weight = _MOBILITY[p.piece]
        mob = sum(1 for t in attacks.get(s, ())
                  if occ.get(t) is None or occ[t].color != p.color)
        score = place + (mob - base) * weight
        sums[p.color] += score
        # tie-break on square so identical scores pick the same piece every run
        if worst[p.color] is None or (score, s) < worst[p.color][:2]:
            worst[p.color] = (score, s, p.piece)
    cp = sums["white"] - sums["black"]

    features = {}
    for color in ("white", "black"):
        if worst[color] is not None:
            _, s, pc = worst[color]
            features[f"worst_piece_{color}"] = {"square": s, "piece": pc}
    stm = board.side_to_move
    nudge = ""
    w = worst.get(stm)
    if w is not None:
        nudge = (f"; {_side_name(stm)}'s least active piece is the "
                 f"{PIECE_NAME.get(w[2], w[2])} on {w[1]}")
    if cp >= _LEAD_THRESHOLD:
        standing = "White's pieces are the more active" + nudge
    elif cp <= -_LEAD_THRESHOLD:
        standing = "Black's pieces are the more active" + nudge
    else:
        standing = "piece activity is roughly balanced" + nudge
    return {"cp": round(cp), "standing": standing, "features": features}


def _pawn_features(occ, color: str) -> dict:
    enemy = "black" if color == "white" else "white"
    own = [s for s, p in occ.items() if p.color == color and p.piece == "P"]
    theirs = [s for s, p in occ.items() if p.color == enemy and p.piece == "P"]
    files: dict[int, list[int]] = {}
    for s in own:
        f, r = _fr(s)
        files.setdefault(f, []).append(r)
    efiles: dict[int, list[int]] = {}
    for s in theirs:
        f, r = _fr(s)
        efiles.setdefault(f, []).append(r)

    doubled = sorted(_FILES[f] for f, rs in files.items() if len(rs) > 1)
    isolated = sorted(s for s in own
                      if not any((_fr(s)[0] + d) in files for d in (-1, 1)))
    passed = []
    for s in own:
        f, r = _fr(s)
        ahead = range(r + 1, 8) if color == "white" else range(0, r)
        if not any(df in efiles and any(er in ahead for er in efiles[df])
                   for df in (f - 1, f, f + 1)):
            passed.append(s)
    islands, prev = 0, False
    for f in range(8):
        here = f in files
        islands += here and not prev
        prev = here
    extra_on_file = sum(len(rs) - 1 for rs in files.values())
    return {"doubled_files": doubled, "isolated": isolated,
            "passed": sorted(passed), "islands": islands,
            "_extra": extra_on_file}


def _pawns_term(board, occ, ph: float) -> dict:
    feats, score = {}, {}
    for color in ("white", "black"):
        f = _pawn_features(occ, color)
        cp = 0.0
        cp += len(f["isolated"]) * _taper(*_ISOLATED, ph)
        cp += f["_extra"] * _taper(*_DOUBLED, ph)
        cp += max(0, f["islands"] - 1) * _taper(*_ISLAND, ph)
        for s in f["passed"]:
            _, r = _fr(s)
            rel = r if color == "white" else 7 - r
            cp += _taper(*_PASSED[rel], ph)
        score[color] = cp
        feats[color] = {k: v for k, v in f.items() if not k.startswith("_")}
    cp = score["white"] - score["black"]

    phrases: list[tuple[float, str]] = []
    for color in ("white", "black"):
        name, f = _side_name(color), feats[color]
        if f["passed"]:
            best = max(f["passed"],
                       key=lambda s: _fr(s)[1] if color == "white" else 7 - _fr(s)[1])
            rel = _fr(best)[1] if color == "white" else 7 - _fr(best)[1]
            phrases.append((_taper(*_PASSED[rel], ph),
                            f"{name} has a passed pawn on {best}"))
        for s in f["isolated"]:
            phrases.append((abs(_taper(*_ISOLATED, ph)),
                            f"{name}'s {s} pawn is isolated"))
        for fl in f["doubled_files"]:
            phrases.append((abs(_taper(*_DOUBLED, ph)),
                            f"{name} has doubled pawns on the {fl}-file"))
    phrases.sort(key=lambda t: -t[0])
    standing = ("; ".join(p for _, p in phrases[:2])
                if phrases else "both pawn structures are healthy")
    return {"cp": round(cp), "standing": standing, "features": feats}


def _center_term(occ, ctrl, ph: float) -> dict:
    cp = 0.0
    held = {"white": [], "black": []}
    detail = {}
    for s in _CORE:
        p = occ.get(s)
        if p is not None and p.piece in _OCCUPY:
            v = _taper(*_OCCUPY[p.piece], ph)
            cp += v if p.color == "white" else -v
            held[p.color].append(s)
        w, b = ctrl[s]
        cp += (w - b) * _taper(*_CONTROL, ph)
        detail[s] = {"white_attackers": w, "black_attackers": b}
    if cp >= _LEAD_THRESHOLD:
        standing = "White controls the centre"
    elif cp <= -_LEAD_THRESHOLD:
        standing = "Black controls the centre"
    else:
        standing = "the centre is contested"
    for color in ("white", "black"):
        if held[color]:
            standing += f"; {_side_name(color)} holds {', '.join(held[color])}"
    return {"cp": round(cp), "standing": standing, "features": detail}


# --- assembly ----------------------------------------------------------------

def analyze_positional(board) -> dict:
    """The five-term positional decomposition for `board`.

    Returns `{phase, terms:{material,king_safety,activity,pawns,center},
    leads:[...]}` where each term is `{cp, standing, features?}` (cp White-POV)
    and `leads` names the terms salient enough to open the coaching with,
    strongest first. Purely board-geometric: no engine, deterministic, same
    input -> same output."""
    occ = occupancy(board)
    ph = _phase(occ)
    ctrl, attacks = _attack_maps(board)
    terms = {
        "material": _material_term(board),
        "king_safety": _king_safety_term(board, occ, ph),
        "activity": _activity_term(board, occ, attacks, ph),
        "pawns": _pawns_term(board, occ, ph),
        "center": _center_term(occ, ctrl, ph),
    }
    leads = [k for k in sorted(terms, key=lambda k: -abs(terms[k]["cp"]))
             if abs(terms[k]["cp"]) >= _LEAD_THRESHOLD][:2]
    return {"phase": round(ph, 2), "terms": terms, "leads": leads}
