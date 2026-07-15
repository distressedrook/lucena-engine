# Releasing lucena-engine

Distributed via **PyPI**. Stockfish is a required runtime engine (installed separately); **Maia stays
opt-in** (PyTorch — installed on demand, never in the default).

```
pip install lucena-engine                 # library + CLI + gRPC-ready
brew install stockfish                    # required engine
lucena-engine setup-maia --install        # opt-in behaviour model
```

---

## PyPI (one-time setup, then tag to ship)

**One-time — configure Trusted Publishing** (no API tokens):
1. Register on PyPI and enable 2FA.
2. https://pypi.org/manage/account/publishing/ → *Add a pending publisher*:
   - PyPI project name: `lucena-engine`
   - Owner: `distressedrook`  ·  Repo: `lucena-engine`  ·  Workflow: `release.yml`  ·  Environment: `pypi`
3. In the GitHub repo: **Settings → Environments → New environment → `pypi`** (no secrets).

**Every release:**
```bash
# bump version in pyproject.toml, commit, then:
git tag v0.1.0
git push origin v0.1.0
```
`.github/workflows/release.yml` builds wheels (macOS arm64, Linux x86_64/aarch64, py3.10–3.13) + an
sdist and publishes them to PyPI via OIDC. No secrets stored.

Local dry run of the build:
```bash
pipx run maturin build --release --manifest-path rust/Cargo.toml
pipx run maturin sdist  --manifest-path rust/Cargo.toml
```

Version = tag: `pyproject.toml`'s `version` must match the `vX.Y.Z` tag. PyPI won't allow re-uploading
the same version — if a run half-fails, bump the version rather than retrying the tag.

---

## What the user gets

| Command | Needs |
|---|---|
| `lucena-engine version` / `doctor` | nothing (reports what's present) |
| `lucena-engine analyze "<FEN>"` | Stockfish |
| `lucena-engine serve` | the `grpc` extra: `pip install 'lucena-engine[grpc]'` |
| `lucena-engine setup-maia` | opt-in; downloads PyTorch into `~/.lucena/maia-venv` |

Intel-mac users (no prebuilt wheel): `pip install lucena-engine` builds from the sdist — needs a Rust
toolchain (`brew install rust`).

## Licensing note

`lucena-engine` is AGPL-3.0. Stockfish (GPL) and Maia (via `maia3`, which pulls GPL `python-chess`) are
invoked **only as subprocesses over UCI** — never linked. Maia lives in its own venv. This arm's-length
boundary is what keeps the shipped package clean (enforced by `tests/test_gpl_hygiene.py`).
