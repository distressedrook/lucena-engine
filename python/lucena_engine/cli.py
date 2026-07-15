"""`lucena-engine` command-line entry point.

Thin wrapper over the library so a `pip install` puts a usable command on PATH:

    lucena-engine version          # version + where Stockfish/Maia resolve
    lucena-engine doctor           # check the runtime is wired up
    lucena-engine analyze "<FEN>"  # print the grounded fact sheet for a position
    lucena-engine serve            # start the gRPC Truth/Behaviour server (needs the grpc extra)
    lucena-engine setup-maia       # install the optional Maia behaviour model

Everything heavy is imported lazily so `version`/`doctor` stay fast and work
even when an optional piece (grpc, Stockfish, Maia) is absent.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("lucena-engine")
    except Exception:  # pragma: no cover - source checkout without metadata
        return "0+unknown"


def _stockfish_cmd() -> str:
    return os.environ.get("LUCENA_STOCKFISH", "stockfish")


def _maia_cmd() -> str:
    return os.environ.get("LUCENA_MAIA", "maia3-5m")


def _resolve(cmd: str) -> str | None:
    """Absolute path of the first token of `cmd`, or None if not on PATH."""
    first = cmd.split()[0] if cmd else ""
    if os.path.isabs(first):
        return first if os.path.exists(first) else None
    return shutil.which(first)


# ---------------------------------------------------------------- version

def cmd_version(_args) -> int:
    sf = _resolve(_stockfish_cmd())
    maia = _resolve(_maia_cmd())
    print(f"lucena-engine {_version()}  (python {sys.version.split()[0]})")
    print(f"  stockfish : {sf or 'NOT FOUND'}   (LUCENA_STOCKFISH={_stockfish_cmd()})")
    print(f"  maia      : {maia or 'not installed'}   (LUCENA_MAIA={_maia_cmd()}, optional)")
    return 0


# ---------------------------------------------------------------- doctor

def cmd_doctor(_args) -> int:
    ok = True

    # 1. native board core
    try:
        from .board import Board

        Board()
        print("  [ok]   board core (Rust extension) importable")
    except Exception as e:  # pragma: no cover
        ok = False
        print(f"  [FAIL] board core not importable: {e}")

    # 2. Stockfish (required)
    sf = _resolve(_stockfish_cmd())
    if sf:
        try:
            out = subprocess.run([sf], input=b"uci\nquit\n", capture_output=True, timeout=10)
            tag = b"uciok" in out.stdout
            print(f"  [{'ok' if tag else 'warn'}]   stockfish at {sf}"
                  + ("" if tag else "  (did not answer uciok)"))
            ok = ok and tag
        except Exception as e:
            ok = False
            print(f"  [FAIL] stockfish at {sf} did not run: {e}")
    else:
        ok = False
        print(f"  [FAIL] stockfish not found — install it and/or set LUCENA_STOCKFISH")

    # 3. gRPC (optional extra)
    try:
        import grpc  # noqa: F401

        print("  [ok]   grpc extra installed (serve available)")
    except Exception:
        print("  [info] grpc extra not installed — `serve` unavailable "
              "(pip install 'lucena-engine[grpc]')")

    # 4. Maia (optional)
    maia = _resolve(_maia_cmd())
    print(f"  [{'ok' if maia else 'info'}]   maia "
          + (f"at {maia}" if maia else "not installed (optional — run `lucena-engine setup-maia`)"))

    print("\nready" if ok else "\nnot ready — resolve the [FAIL] lines above")
    return 0 if ok else 1


# ---------------------------------------------------------------- analyze

def cmd_analyze(args) -> int:
    from .board import Board
    from .uci import Engine, EngineError
    from .analysis import build_analysis

    try:
        engine = Engine()
    except EngineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    try:
        board = Board(args.fen)
    except Exception as e:
        print(f"error: invalid FEN: {e}", file=sys.stderr)
        return 2
    try:
        for line in build_analysis(board, engine, movetime_ms=args.movetime):
            print(line)
    finally:
        engine.close()
    return 0


# ---------------------------------------------------------------- serve

def cmd_serve(args) -> int:
    try:
        from .server.serve import serve
    except ImportError:
        print("error: the gRPC server needs the 'grpc' extra:\n"
              "  pip install 'lucena-engine[grpc]'", file=sys.stderr)
        return 1
    serve(port=args.port, threads=args.threads)
    return 0


# ---------------------------------------------------------------- setup-maia

_MAIA_STEPS = """\
Maia is optional (it powers human-move prediction and poisoned-line detection).
It is a PyTorch model — kept in its OWN environment so it never links into the
engine (this preserves lucena-engine's license hygiene).

Install it into an isolated venv:

  python3 -m venv ~/.lucena/maia-venv
  ~/.lucena/maia-venv/bin/pip install 'git+https://github.com/CSSLab/maia3'

Then point the engine at the UCI command:

  export LUCENA_MAIA="$HOME/.lucena/maia-venv/bin/maia3-5m"

Verify:

  lucena-engine doctor
"""


def cmd_setup_maia(args) -> int:
    venv = os.path.expanduser("~/.lucena/maia-venv")
    if not args.install:
        print(_MAIA_STEPS)
        print(f"(re-run with --install to create {venv} and install automatically)")
        return 0

    print(f"creating {venv} and installing maia3 (this downloads PyTorch — large)...")
    try:
        subprocess.run([sys.executable, "-m", "venv", venv], check=True)
        pip = os.path.join(venv, "bin", "pip")
        subprocess.run([pip, "install", "git+https://github.com/CSSLab/maia3"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"error: install failed: {e}", file=sys.stderr)
        return 1
    cmd = os.path.join(venv, "bin", "maia3-5m")
    print(f"\ndone. Add this to your shell profile:\n  export LUCENA_MAIA=\"{cmd}\"")
    return 0


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lucena-engine",
                                description="Deterministic chess truth for grounding language models.")
    p.add_argument("-V", "--version", action="version", version=f"lucena-engine {_version()}")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("version", help="print version + component locations").set_defaults(func=cmd_version)
    sub.add_parser("doctor", help="check the runtime (board core, Stockfish, grpc, Maia)").set_defaults(func=cmd_doctor)

    a = sub.add_parser("analyze", help="print the grounded fact sheet for a FEN")
    a.add_argument("fen", help="position in FEN")
    a.add_argument("--movetime", type=int, default=500, help="Stockfish think time in ms (default 500)")
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("serve", help="start the gRPC Truth/Behaviour server")
    s.add_argument("--port", type=int, default=int(os.environ.get("LUCENA_ENGINE_PORT", "50051")))
    s.add_argument("--threads", type=int, default=1)
    s.set_defaults(func=cmd_serve)

    m = sub.add_parser("setup-maia", help="install the optional Maia behaviour model")
    m.add_argument("--install", action="store_true", help="create the venv and install automatically")
    m.set_defaults(func=cmd_setup_maia)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, "func", None):
        build_parser().print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
