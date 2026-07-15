# Releasing lucena-engine

Two channels: **PyPI** (the real package) and a **Homebrew tap** (the `brew install` UX that wraps it).
Stockfish is a dependency; **Maia stays opt-in** (PyTorch — installed on demand, never in the default).

```
pip install lucena-engine                 # library + CLI + gRPC-ready
brew install distressedrook/tap/lucena-engine   # + Stockfish, on PATH
lucena-engine setup-maia --install        # opt-in behaviour model
```

---

## 1. PyPI (one-time setup, then tag to ship)

**One-time — configure Trusted Publishing** (no API tokens):
1. Create the empty project owner on PyPI, or just register on first publish.
2. https://pypi.org/manage/account/publishing/ → *Add a pending publisher*:
   - PyPI project name: `lucena-engine`
   - Owner: `distressedrook`  ·  Repo: `lucena-engine`  ·  Workflow: `release.yml`  ·  Environment: `pypi`

**Every release:**
```bash
# bump version in pyproject.toml, commit, then:
git tag v0.1.0
git push origin v0.1.0
```
`.github/workflows/release.yml` builds wheels (macOS arm64/x86_64, Linux x86_64/aarch64, py3.10–3.13) +
an sdist and publishes them to PyPI via OIDC. No secrets stored.

Local dry run of the build:
```bash
pipx run maturin build --release --manifest-path rust/Cargo.toml
pipx run maturin sdist  --manifest-path rust/Cargo.toml
```

---

## 2. Homebrew tap

The formula lives in a **separate repo** named `homebrew-tap` (the `homebrew-` prefix is what lets
`brew tap distressedrook/tap` resolve it).

**One-time:**
1. Create `distressedrook/homebrew-tap` (public).
2. Copy `packaging/homebrew/lucena-engine.rb` → `homebrew-tap/Formula/lucena-engine.rb`.

**Each release — point the formula at the new tag and fill the hash:**
```bash
TAG=v0.1.0
URL="https://github.com/distressedrook/lucena-engine/archive/refs/tags/$TAG.tar.gz"
curl -sL "$URL" | shasum -a 256          # paste into the formula's sha256
# set `url` to $URL, commit, push the tap
```

**Verify before announcing:**
```bash
brew install --build-from-source distressedrook/tap/lucena-engine
brew test lucena-engine
brew audit --strict --online distressedrook/tap/lucena-engine
```

The formula builds the Rust extension (`depends_on "rust"/"maturin" => :build`), installs into an
isolated venv, and pulls Stockfish. It has **no runtime Python deps**, so no `resource` blocks are
needed — keep it that way (the `grpc` extra and Maia are deliberately out of the default install).

---

## 3. What the user gets

| Command | Needs |
|---|---|
| `lucena-engine version` / `doctor` | nothing (reports what's present) |
| `lucena-engine analyze "<FEN>"` | Stockfish (dependency) |
| `lucena-engine serve` | the `grpc` extra: `pip install 'lucena-engine[grpc]'` |
| `lucena-engine setup-maia` | opt-in; downloads PyTorch into `~/.lucena/maia-venv` |

## Licensing note

`lucena-engine` is AGPL-3.0. Stockfish (GPL) and Maia (via `maia3`, which pulls GPL `python-chess`) are
invoked **only as subprocesses over UCI** — never linked. Homebrew installs Stockfish as its own
formula; Maia lives in its own venv. This arm's-length boundary is what keeps the shipped package clean
(enforced by `tests/test_gpl_hygiene.py`).
