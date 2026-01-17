# Homebrew formula for ABI Framework
# Install with: brew tap donaldfilimon/abi && brew install abi

class Abi < Formula
  desc "Modern Zig 0.16 framework for AI services, vector search, and high-performance tooling"
  homepage "https://github.com/donaldfilimon/abi"
  url "https://github.com/donaldfilimon/abi/archive/refs/tags/v0.3.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256_HASH"
  license "MIT"
  head "https://github.com/donaldfilimon/abi.git", branch: "main"

  depends_on "zig" => ["0.16", :build]

  def install
    # Build with all features enabled
    system "zig", "build",
           "-Doptimize=ReleaseFast",
           "-Denable-ai=true",
           "-Denable-gpu=true",
           "-Denable-database=true",
           "-Denable-network=true",
           "-Denable-web=true",
           "-Denable-profiling=true",
           "--prefix=#{prefix}"

    # Install binary
    bin.install "zig-out/bin/abi"

    # Install documentation
    doc.install "README.md"
    doc.install "CLAUDE.md"
    doc.install "API_REFERENCE.md" if File.exist?("API_REFERENCE.md")
    doc.install Dir["docs/*"]
  end

  def caveats
    <<~EOS
      ABI framework has been installed.

      To get started:
        abi --help
        abi tui              # Interactive launcher
        abi system-info      # Show system capabilities
        abi gpu backends     # List GPU backends

      Configure AI connectors via environment variables:
        export ABI_OPENAI_API_KEY="sk-..."
        export ABI_OLLAMA_HOST="http://localhost:11434"
        export ABI_HF_API_TOKEN="hf_..."

      For more information, see:
        https://github.com/donaldfilimon/abi
        https://donaldfilimon.github.io/abi/
    EOS
  end

  test do
    # Verify binary runs and outputs version
    assert_match "abi version", shell_output("#{bin}/abi version")

    # Test system-info command
    assert_match "System Information", shell_output("#{bin}/abi system-info")
  end
end
