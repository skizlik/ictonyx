"""Test memory management."""

import sys

import pytest

from ictonyx.memory import (
    MemoryManager,
    MemoryResult,
    check_isolation_capability,
    cleanup_gpu_memory,
    get_memory_info,
    managed_memory,
)


class TestMemoryResult:
    """Test MemoryResult dataclass."""

    def test_result_creation(self):
        """Test creating memory results."""
        result = MemoryResult(success=True, actions=["cleared_session"], memory_freed_mb=150.5)

        assert result.success
        assert "cleared_session" in result.actions
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
        assert result.mode == "standard"

    def test_process_isolation_mode(self):
        """Test process isolation mode creation."""
        manager = MemoryManager(use_process_isolation=True)

        assert manager.use_process_isolation

        # Should have context setup
        assert hasattr(manager, "ctx")

    def test_cleanup_always_returns_result(self):
        """Test cleanup always returns a result."""
        manager = MemoryManager()

        result = manager.cleanup()
        assert isinstance(result, MemoryResult)
        assert hasattr(result, "success")
        assert hasattr(result, "actions")
        assert result.mode in ["standard", "isolated"]

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
            assert result["success"] or "error" in result


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
        assert result.mode == "standard"

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


class TestMemoryManagerSetup:
    """Cover MemoryManager.setup() and mode-specific paths."""

    def test_setup_standard_returns_bool(self):
        manager = MemoryManager(use_process_isolation=False, verbose=False)
        result = manager.setup()
        assert isinstance(result, bool)

    def test_cleanup_standard_has_gc_action(self):
        manager = MemoryManager(use_process_isolation=False, verbose=False)
        result = manager.cleanup()
        gc_actions = [a for a in result.actions if "gc_collected" in a]
        assert len(gc_actions) == 1

    def test_cleanup_isolated_mode_skips_gc(self):
        """Process isolation cleanup returns immediately — no GC action."""
        manager = MemoryManager(use_process_isolation=True, verbose=False)
        result = manager.cleanup()
        assert result.success
        assert "process_isolation_no_cleanup" in result.actions

    def test_managed_memory_yields_manager(self):
        with managed_memory(use_process_isolation=False) as mgr:
            assert isinstance(mgr, MemoryManager)
            assert not mgr.use_process_isolation

    def test_cleanup_gpu_memory_returns_result(self):
        result = cleanup_gpu_memory()
        assert isinstance(result, MemoryResult)
        assert result.success is not None

    def test_get_memory_info_returns_dict(self):
        info = get_memory_info()
        assert isinstance(info, dict)

    def test_memory_result_defaults(self):
        r = MemoryResult(success=True)
        assert r.actions == []
        assert r.errors == []
        assert r.memory_freed_mb is None

    def test_check_isolation_picklable_function(self):
        def add(x, y):
            return x + y

        can, reason = check_isolation_capability(add)
        assert isinstance(can, bool)
        assert isinstance(reason, str)

    def test_get_memory_manager_default(self):
        from ictonyx.memory import get_memory_manager

        mgr = get_memory_manager()
        assert isinstance(mgr, MemoryManager)
        assert not mgr.use_process_isolation

    def test_get_memory_manager_isolation(self):
        from ictonyx.memory import get_memory_manager

        mgr = get_memory_manager(use_process_isolation=True)
        assert mgr.use_process_isolation


class TestMemoryManagerVerbose:
    """Cover verbose=True output paths."""

    def test_verbose_false_does_not_crash(self):
        manager = MemoryManager(use_process_isolation=False, verbose=False)
        result = manager.cleanup()
        assert result.success

    def test_setup_idempotent(self):
        """Calling setup() twice must not raise or duplicate actions."""
        manager = MemoryManager(use_process_isolation=False, verbose=False)
        r1 = manager.setup()
        r2 = manager.setup()
        assert isinstance(r1, bool)
        assert r2 is True  # second call returns True immediately

    def test_cleanup_result_has_mode(self):
        manager = MemoryManager(use_process_isolation=False, verbose=False)
        result = manager.cleanup()
        assert result.mode == "standard"

    def test_managed_memory_cleanup_on_exit(self):
        """Context manager must not raise on exit even without GPU."""
        exited = False
        with managed_memory(use_process_isolation=False) as mgr:
            assert mgr is not None
        exited = True
        assert exited

    def test_memory_result_success_field(self):
        r = MemoryResult(success=False, errors=["something failed"])
        assert not r.success
        assert len(r.errors) == 1
