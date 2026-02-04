"""Tests for GPU backend selection in ABI Python bindings.

Run with: python -m pytest test_gpu.py -v
Or: python -m pytest test_gpu.py::test_gpu_backend_selection -v
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from abi import GpuBackend, ABI


class TestGpuBackend:
    """Test the GpuBackend helper class."""

    def test_valid_backends_list(self):
        """Test that VALID_BACKENDS contains all expected backends."""
        expected = ['auto', 'cuda', 'vulkan', 'metal', 'webgpu', 'stdgpu', 'cpu']
        assert GpuBackend.VALID_BACKENDS == expected

    def test_backend_constants(self):
        """Test that backend constants have correct values."""
        assert GpuBackend.AUTO == 'auto'
        assert GpuBackend.CUDA == 'cuda'
        assert GpuBackend.VULKAN == 'vulkan'
        assert GpuBackend.METAL == 'metal'
        assert GpuBackend.WEBGPU == 'webgpu'
        assert GpuBackend.STDGPU == 'stdgpu'
        assert GpuBackend.CPU == 'cpu'

    def test_validate_valid_backend(self):
        """Test validation passes for valid backends."""
        for backend in GpuBackend.VALID_BACKENDS:
            result = GpuBackend.validate(backend)
            assert result == backend

    def test_validate_invalid_backend(self):
        """Test validation raises ValueError for invalid backends."""
        with pytest.raises(ValueError) as exc_info:
            GpuBackend.validate('invalid')
        assert "Invalid backend 'invalid'" in str(exc_info.value)
        assert "Must be one of:" in str(exc_info.value)

    def test_validate_case_sensitive(self):
        """Test that backend validation is case-sensitive."""
        with pytest.raises(ValueError):
            GpuBackend.validate('CUDA')
        with pytest.raises(ValueError):
            GpuBackend.validate('Auto')

    def test_to_int_mapping(self):
        """Test conversion from backend name to C API integer."""
        assert GpuBackend.to_int('auto') == 0
        assert GpuBackend.to_int('cuda') == 1
        assert GpuBackend.to_int('vulkan') == 2
        assert GpuBackend.to_int('metal') == 3
        assert GpuBackend.to_int('webgpu') == 4
        assert GpuBackend.to_int('stdgpu') == 5
        assert GpuBackend.to_int('cpu') == 6

    def test_to_int_unknown_returns_zero(self):
        """Test that unknown backends map to 0 (auto)."""
        assert GpuBackend.to_int('unknown') == 0
        assert GpuBackend.to_int('') == 0


class TestVectorDatabaseBackendSelection:
    """Test GPU backend selection in VectorDatabase.

    Note: These tests validate the Python-side logic. Tests that require
    the native library are marked with pytest.mark.skipif.
    """

    def test_auto_backend_default(self):
        """Test that 'auto' is the default backend."""
        # This test validates argument parsing without native library
        # We can't create a real VectorDatabase without the library,
        # but we can verify the GpuBackend class behavior
        backend = 'auto'
        validated = GpuBackend.validate(backend)
        assert validated == 'auto'

    def test_backend_validation_on_init(self):
        """Test that invalid backend raises ValueError."""
        # Create a mock that tests the validation logic
        with pytest.raises(ValueError) as exc_info:
            GpuBackend.validate('invalid_backend')
        assert "Invalid backend" in str(exc_info.value)

    @pytest.mark.parametrize("backend", GpuBackend.VALID_BACKENDS)
    def test_all_backends_validate(self, backend):
        """Test that all valid backends pass validation."""
        result = GpuBackend.validate(backend)
        assert result == backend


# Integration tests that require the native library
class TestVectorDatabaseIntegration:
    """Integration tests requiring the native ABI library.

    These tests are skipped if the library is not available.
    """

    @pytest.fixture
    def abi(self):
        """Create an ABI instance, skipping if library not found."""
        try:
            return ABI()
        except FileNotFoundError:
            pytest.skip("ABI native library not found")

    def test_gpu_backend_cuda(self, abi):
        """Test creating database with CUDA backend."""
        db = abi.create_db(dimension=128, backend='cuda')
        assert db.backend == 'cuda'
        assert db._backend_int == 1

    def test_gpu_backend_vulkan(self, abi):
        """Test creating database with Vulkan backend."""
        db = abi.create_db(dimension=128, backend='vulkan')
        assert db.backend == 'vulkan'
        assert db._backend_int == 2

    def test_gpu_backend_metal(self, abi):
        """Test creating database with Metal backend."""
        db = abi.create_db(dimension=128, backend='metal')
        assert db.backend == 'metal'
        assert db._backend_int == 3

    def test_cpu_fallback(self, abi):
        """Test creating database with CPU backend (no GPU)."""
        db = abi.create_db(dimension=128, backend='cpu')
        assert db.backend == 'cpu'
        assert db._backend_int == 6

    def test_auto_backend(self, abi):
        """Test creating database with auto backend (default)."""
        db = abi.create_db(dimension=128)
        assert db.backend == 'auto'
        assert db._backend_int == 0

    def test_auto_backend_explicit(self, abi):
        """Test creating database with explicit 'auto' backend."""
        db = abi.create_db(dimension=128, backend='auto')
        assert db.backend == 'auto'
        assert db._backend_int == 0

    def test_invalid_backend_raises(self, abi):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            abi.create_db(dimension=128, backend='invalid')
        assert "Invalid backend 'invalid'" in str(exc_info.value)

    def test_stdgpu_backend(self, abi):
        """Test creating database with stdgpu backend."""
        db = abi.create_db(dimension=128, backend='stdgpu')
        assert db.backend == 'stdgpu'
        assert db._backend_int == 5

    def test_webgpu_backend(self, abi):
        """Test creating database with WebGPU backend."""
        db = abi.create_db(dimension=128, backend='webgpu')
        assert db.backend == 'webgpu'
        assert db._backend_int == 4

    def test_backend_preserved_after_operations(self, abi):
        """Test that backend setting is preserved after insert/search."""
        db = abi.create_db(dimension=4, backend='cuda')

        # Insert a vector
        db.insert(1, [1.0, 0.0, 0.0, 0.0])

        # Backend should still be set
        assert db.backend == 'cuda'

        # Search (just exercise the API, not checking results here)
        _ = db.search([1.0, 0.0, 0.0, 0.0], k=1)

        # Backend should still be set
        assert db.backend == 'cuda'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
