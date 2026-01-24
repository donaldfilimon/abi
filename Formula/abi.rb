# Homebrew formula for ABI Framework
#
# Installation:
#   brew tap donaldthai/abi https://github.com/donaldthai/abi
#   brew install abi
#
# Or install directly:
#   brew install --build-from-source Formula/abi.rb

class Abi < Formula
  desc "High-performance AI & Vector Database Framework in Zig"
  homepage "https://github.com/donaldthai/abi"
  url "https://github.com/donaldthai/abi/archive/refs/tags/v0.4.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"
  head "https://github.com/donaldthai/abi.git", branch: "main"

  depends_on "zig" => "0.16.0"

  def install
    # Build with default features
    system "zig", "build",
           "-Doptimize=ReleaseFast",
           "--prefix=#{prefix}"

    # Generate shell completions
    generate_completions_from_executable(bin/"abi", "completions", "--shell")
  end

  def caveats
    <<~EOS
      ABI Framework has been installed!

      Quick start:
        abi --help              # Show available commands
        abi system-info         # Display system and feature status
        abi db stats            # Database statistics
        abi llm info            # LLM model information

      For GPU acceleration, ensure you have the appropriate drivers installed:
        - CUDA: NVIDIA drivers and CUDA toolkit
        - Vulkan: Vulkan SDK
        - Metal: macOS only (built-in)

      Documentation: https://github.com/donaldthai/abi/docs
    EOS
  end

  test do
    # Test version output
    assert_match version.to_s, shell_output("#{bin}/abi version")

    # Test system info
    assert_match "ABI Framework", shell_output("#{bin}/abi system-info")

    # Test help output
    assert_match "Commands:", shell_output("#{bin}/abi help")
  end
end
