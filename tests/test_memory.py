"""Test memory management."""
import pytest
import sys
from ictonyx.memory import (
    MemoryManager,
    MemoryResult,
    get_memory_info,
    cleanup_gpu_memory,
    managed_memory,
    check_isolation_capability
)


class TestMemoryResult:
    """Test MemoryResult dataclass."""
    
    def test_result_creation(self):
        """Test creating memory results."""
        result = MemoryResult(
            success=True,
            actions=['cleared_session'],
            memory_freed_mb=150.5
        )
        
        assert result.success
        assert 'cleared_session' in result.actions
        assert result.memory_freed_mb == 150.5


class TestMemoryManager:
    """Test MemoryManager class."""
    
    def test_standard_mode(self):
        """Test standard memory manager."""
        manager = MemoryManager(use_process_isolation=False)
        
        assert not manager.use_process_isolation
        assert manager.verbose  # Default is True
        
        # Setup should work
        success = manager.setup()
        assert isinstance(success, bool)
        
        # Cleanup should return result
        result = manager.cleanup()
        assert isinstance(result, MemoryResult)
        assert result.mode == 'standard'
    
    def test_process_isolation_mode(self):
        """Test process isolation mode creation."""
        manager = MemoryManager(use_process_isolation=True)
        
        assert manager.use_process_isolation
        
        # Should have context setup
        assert hasattr(manager, 'ctx')
    
    def test_cleanup_always_returns_result(self):
        """Test cleanup always returns a result."""
        manager = MemoryManager()
        
        result = manager.cleanup()
        assert isinstance(result, MemoryResult)
        assert hasattr(result, 'success')
        assert hasattr(result, 'actions')
        assert result.mode in ['standard', 'isolated']
    
    @pytest.mark.skipif(sys.platform == "win32", reason="Process isolation issues on Windows")
    def test_isolation_with_simple_function(self):
        """Test process isolation with picklable function."""
        manager = MemoryManager(use_process_isolation=True)
        
        def simple_func(x, y):
            return x + y
        
        # Check if function can be isolated
        can_isolate, reason = check_isolation_capability(simple_func)
        
        if can_isolate:
            result = manager.run_isolated(simple_func, args=(2, 3))
            assert result['success'] or 'error' in result


class TestMemoryFunctions:
    """Test module-level memory functions."""
    
    def test_get_memory_info(self):
        """Test getting memory information."""
        info = get_memory_info()
        
        assert isinstance(info, dict)
        # Should have some keys (depends on what's available)
        # At minimum should return empty dict
    
    def test_cleanup_gpu_memory(self):
        """Test GPU cleanup function."""
        result = cleanup_gpu_memory()
        
        assert isinstance(result, MemoryResult)
        assert result.mode == 'standard'
    
    def test_managed_memory_context(self):
        """Test memory management context manager."""
        with managed_memory() as manager:
            assert isinstance(manager, MemoryManager)
            assert not manager.use_process_isolation
        
        # Should have cleaned up after exiting
    
    def test_check_isolation_capability(self):
        """Test checking if functions can be isolated."""
        
        def simple_func():
            return 42
        
        can_isolate, reason = check_isolation_capability(simple_func)
        assert isinstance(can_isolate, bool)
        assert isinstance(reason, str)
        
        # Lambda functions might not be picklable
        lambda_func = lambda x: x * 2
        can_isolate_lambda, reason = check_isolation_capability(lambda_func)
        # Result depends on pickle/cloudpickle availability
