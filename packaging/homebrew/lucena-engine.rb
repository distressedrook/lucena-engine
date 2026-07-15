# Homebrew formula for lucena-engine.
#
# This lives in a SEPARATE tap repo so users can:
#   brew tap distressedrook/tap
#   brew install lucena-engine        # or: brew install distressedrook/tap/lucena-engine
#
# To publish/update: copy this file into `distressedrook/homebrew-tap/Formula/`,
# then set `url` to the release tag and fill `sha256` (see packaging/README.md).
class LucenaEngine < Formula
  include Language::Python::Virtualenv

  desc "Deterministic chess truth for grounding language models"
  homepage "https://github.com/distressedrook/lucena-engine"
  # Point at the tagged source tarball, then run:
  #   curl -sL <url> | shasum -a 256
  url "https://github.com/distressedrook/lucena-engine/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "REPLACE_WITH_SHA256_OF_THE_TARBALL"
  license "AGPL-3.0-or-later"
  head "https://github.com/distressedrook/lucena-engine.git", branch: "main"

  depends_on "maturin" => :build   # builds the Rust extension
  depends_on "rust" => :build
  depends_on "python@3.12"
  depends_on "stockfish"           # required runtime engine (GPL, arm's-length subprocess)

  def install
    # No runtime Python dependencies — the venv holds only lucena-engine itself.
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      Stockfish was installed as a dependency and is picked up automatically.

      Maia (human-move prediction) is optional and NOT installed by default
      (it pulls in PyTorch). To add it:

        lucena-engine setup-maia --install
        # then follow the printed `export LUCENA_MAIA=...` line

      Check everything is wired up:

        lucena-engine doctor
    EOS
  end

  test do
    assert_match "lucena-engine", shell_output("#{bin}/lucena-engine version")
    assert_match "roughly equal",
      shell_output("#{bin}/lucena-engine analyze " \
        "'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' --movetime 200")
  end
end
