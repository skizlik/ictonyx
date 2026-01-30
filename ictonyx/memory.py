# ictonyx/memory.py - Production-ready memory management with smart serialization
"""
Memory management for ictonyx with two modes:
1. Standard (default): Fast in-process execution with best-effort cleanup
2. Process isolation (opt-in): Guaranteed cleanup via subprocess with smart serialization

Process isolation uses cloudpickle when available to serialize notebook-defined
functions, making it work seamlessly in Jupyter environments.
"""

import gc
import os
import sys
import time
import warnings
import pickle
import traceback
from typing import Optional, Dict, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import multiprocessing as mp

# Core dependencies
import numpy as np

# Optional dependencies
try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    HAS_TENSORFLOW = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

# CloudPickle for advanced serialization
try:
    import cloudpickle

    HAS_CLOUDPICKLE = True
except ImportError:
    cloudpickle = None
    HAS_CLOUDPICKLE = False


@dataclass
class MemoryResult:
    """Result of memory operations."""
    success: bool
    actions: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    memory_before_mb: Optional[float] = None
    memory_after_mb: Optional[float] = None
    memory_freed_mb: Optional[float] = None
    mode: str = "standard"  # "standard" or "isolated"


class MemoryManager:
    """
    Unified memory manager for both standard and process-isolated training.

    Standard mode (default): Fast in-process execution with cleanup
    Process isolation mode: Subprocess execution with guaranteed cleanup
    """

    def __init__(self,
                 use_process_isolation: bool = False,
                 gpu_memory_limit: Optional[int] = None,
                 process_timeout: int = 3600,
                 allow_memory_growth: bool = True,
                 verbose: bool = True):
        """
        Initialize memory manager.

        Args:
            use_process_isolation: If True, run in isolated subprocess
            gpu_memory_limit: GPU memory limit in MB
            process_timeout: Timeout for subprocess execution
            allow_memory_growth: Allow GPU memory to grow as needed
            verbose: Print informative messages
        """
        self.use_process_isolation = use_process_isolation
        self.gpu_memory_limit = gpu_memory_limit
        self.process_timeout = process_timeout
        self.allow_memory_growth = allow_memory_growth
        self.verbose = verbose
        self._setup_complete = False

        # For process isolation, prepare context
        if use_process_isolation:
            self._setup_process_isolation()

    def _setup_process_isolation(self):
        """Setup process isolation with appropriate serialization."""
        # Check serialization capability
        if not HAS_CLOUDPICKLE:
            if self.verbose:
                print("=" * 60)
                print("PROCESS ISOLATION NOTE:")
                print("CloudPickle not found. Installing it will enable")
                print("notebook-defined functions to work with process isolation:")
                print("  pip install cloudpickle")
                print("=" * 60)

        # Setup multiprocessing context (spawn for clean isolation)
        try:
            current_method = mp.get_start_method(allow_none=True)
            if current_method != 'spawn':
                mp.set_start_method('spawn', force=True)
            self.ctx = mp.get_context('spawn')
        except RuntimeError:
            # Already set, just get context
            self.ctx = mp.get_context('spawn')
        except Exception as e:
            warnings.warn(f"Could not setup spawn context: {e}")
            self.ctx = mp.get_context()

    def setup(self) -> bool:
        """
        Configure memory constraints for standard mode.
        Must be called before any GPU operations.
        """
        if self.use_process_isolation:
            # Setup happens in subprocess
            return True

        if self._setup_complete:
            return True

        success = False

        if HAS_TENSORFLOW:
            success = self._setup_tensorflow() or success

        if HAS_TORCH:
            success = self._setup_pytorch() or success

        self._setup_complete = True
        return success

    def _setup_tensorflow(self) -> bool:
        """Configure TensorFlow memory settings."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return True  # CPU mode

            for gpu in gpus:
                try:
                    if self.gpu_memory_limit:
                        # Hard limit
                        tf.config.set_logical_device_configuration(
                            gpu,
                            [tf.config.LogicalDeviceConfiguration(
                                memory_limit=self.gpu_memory_limit
                            )]
                        )
                        if self.verbose:
                            print(f"Set GPU memory limit: {self.gpu_memory_limit}MB")
                    elif self.allow_memory_growth:
                        # Growth mode
                        tf.config.experimental.set_memory_growth(gpu, True)
                        if self.verbose:
                            print("Enabled GPU memory growth")
                except RuntimeError as e:
                    if "visible devices" in str(e).lower():
                        warnings.warn(
                            "GPU already initialized. Memory settings must be "
                            "configured before first TF operation."
                        )
                        return False
                    raise

            return True

        except Exception as e:
            warnings.warn(f"TensorFlow setup failed: {e}")
            return False

    def _setup_pytorch(self) -> bool:
        """Configure PyTorch memory settings."""
        if not HAS_TORCH or not torch.cuda.is_available():
            return True

        try:
            if self.gpu_memory_limit:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    total_mb = props.total_memory / (1024 ** 2)
                    fraction = min(self.gpu_memory_limit / total_mb, 1.0)
                    torch.cuda.set_per_process_memory_fraction(fraction, i)

                if self.verbose:
                    print(f"Set PyTorch memory fraction: {fraction:.2f}")

            return True

        except Exception as e:
            warnings.warn(f"PyTorch setup failed: {e}")
            return False

    def cleanup(self) -> MemoryResult:
        """
        Perform memory cleanup (standard mode only).
        """
        if self.use_process_isolation:
            # No cleanup needed - process will die
            return MemoryResult(
                success=True,
                actions=['process_isolation_no_cleanup'],
                mode='isolated'
            )

        result = MemoryResult(success=True, mode='standard')

        # Measure before
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                result.memory_before_mb = process.memory_info().rss / (1024 ** 2)
            except:
                pass

        # TensorFlow cleanup
        if HAS_TENSORFLOW:
            try:
                tf.keras.backend.clear_session()
                result.actions.append('tf_clear_session')

                # Reset graph
                try:
                    tf.compat.v1.reset_default_graph()
                    result.actions.append('tf_reset_graph')
                except:
                    pass

                # GPU stats reset
                for gpu in tf.config.list_physical_devices('GPU'):
                    try:
                        tf.config.experimental.reset_memory_stats(gpu.name.split(':')[-1])
                        result.actions.append(f'tf_reset_{gpu.name}_stats')
                    except:
                        pass

            except Exception as e:
                result.errors.append(f'TF cleanup: {str(e)[:100]}')

        # PyTorch cleanup
        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                result.actions.append('torch_empty_cache')
            except Exception as e:
                result.errors.append(f'PyTorch cleanup: {str(e)[:100]}')

        # Garbage collection
        collected = sum(gc.collect() for _ in range(3))
        result.actions.append(f'gc_collected_{collected}')

        # Measure after
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                result.memory_after_mb = process.memory_info().rss / (1024 ** 2)
                if result.memory_before_mb:
                    result.memory_freed_mb = result.memory_before_mb - result.memory_after_mb
            except:
                pass

        result.success = len(result.errors) == 0
        return result

    def run_isolated(self, func: Callable, args: tuple = (),
                     kwargs: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run function in isolated subprocess with smart serialization.

        Automatically uses cloudpickle if available for notebook functions,
        falls back to standard pickle for module functions.
        """
        if not self.use_process_isolation:
            raise RuntimeError("run_isolated requires use_process_isolation=True")

        if kwargs is None:
            kwargs = {}

        # Determine serialization strategy
        can_standard_pickle = self._test_pickle(func, args, kwargs)

        if HAS_CLOUDPICKLE:
            # Use cloudpickle for maximum compatibility
            return self._run_with_cloudpickle(func, args, kwargs)

        elif can_standard_pickle:
            # Use standard pickle
            return self._run_with_standard_pickle(func, args, kwargs)

        else:
            # Can't serialize for subprocess
            warning_msg = (
                "Cannot serialize function for process isolation. "
                "Options:\n"
                "1. Install cloudpickle: pip install cloudpickle\n"
                "2. Define function in a module file (not notebook)\n"
                "3. Use standard mode (use_process_isolation=False)\n"
                "\nFalling back to in-process execution with cleanup..."
            )
            warnings.warn(warning_msg)

            # Fallback: aggressive in-process execution
            return self._run_with_aggressive_cleanup(func, args, kwargs)

    def _test_pickle(self, func, args, kwargs):
        """Test if objects can be pickled."""
        try:
            pickle.dumps((func, args, kwargs))
            return True
        except:
            return False

    def _run_with_cloudpickle(self, func, args, kwargs):
        """Run using cloudpickle serialization."""
        result_queue = self.ctx.Queue()

        # Serialize with cloudpickle
        try:
            serialized_payload = cloudpickle.dumps((func, args, kwargs))
        except Exception as e:
            return {
                'success': False,
                'error': f'CloudPickle serialization failed: {e}'
            }

        # Run subprocess
        process = self.ctx.Process(
            target=_cloudpickle_subprocess_worker,
            args=(result_queue, serialized_payload, self.gpu_memory_limit)
        )

        return self._execute_subprocess(process, result_queue)

    def _run_with_standard_pickle(self, func, args, kwargs):
        """Run using standard pickle."""
        result_queue = self.ctx.Queue()

        process = self.ctx.Process(
            target=_standard_subprocess_worker,
            args=(result_queue, func, args, kwargs, self.gpu_memory_limit)
        )

        return self._execute_subprocess(process, result_queue)

    def _run_with_aggressive_cleanup(self, func, args, kwargs):
        """Fallback: Run in-process with aggressive cleanup."""
        # Clean before
        self.cleanup()

        try:
            result = func(*args, **kwargs)
            return {'success': True, 'result': result, 'mode': 'fallback'}

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'mode': 'fallback'
            }

        finally:
            # Aggressive cleanup
            self.cleanup()
            if HAS_TENSORFLOW:
                try:
                    tf.keras.backend.clear_session()
                except:
                    pass
            gc.collect()

    def _execute_subprocess(self, process, result_queue):
        """Common subprocess execution logic."""
        try:
            process.start()
            process.join(timeout=self.process_timeout)

            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
                return {
                    'success': False,
                    'error': f'Process timeout ({self.process_timeout}s)'
                }

            if process.exitcode != 0:
                return {
                    'success': False,
                    'error': f'Process crashed (exit code {process.exitcode})'
                }

            try:
                return result_queue.get(timeout=5)
            except:
                return {
                    'success': False,
                    'error': 'No result from subprocess'
                }

        except Exception as e:
            return {
                'success': False,
                'error': f'Subprocess execution failed: {e}'
            }
        finally:
            if process.is_alive():
                process.terminate()


# Subprocess worker functions (module-level for picklability)

def _cloudpickle_subprocess_worker(queue, serialized_payload, gpu_memory_limit):
    """Worker using cloudpickle deserialization."""
    import cloudpickle

    try:
        # Setup GPU in subprocess
        _setup_subprocess_gpu(gpu_memory_limit)

        # Deserialize and execute
        func, args, kwargs = cloudpickle.loads(serialized_payload)
        result = func(*args, **kwargs)

        # Cleanup before sending result
        _cleanup_subprocess()

        queue.put({'success': True, 'result': result})

    except Exception as e:
        queue.put({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


def _standard_subprocess_worker(queue, func, args, kwargs, gpu_memory_limit):
    """Worker using standard pickle."""
    try:
        # Setup GPU in subprocess
        _setup_subprocess_gpu(gpu_memory_limit)

        # Execute
        result = func(*args, **kwargs)

        # Cleanup
        _cleanup_subprocess()

        queue.put({'success': True, 'result': result})

    except Exception as e:
        queue.put({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


def _setup_subprocess_gpu(gpu_memory_limit):
    """Configure GPU in subprocess."""
    if HAS_TENSORFLOW:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                if gpu_memory_limit:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(
                            memory_limit=gpu_memory_limit
                        )]
                    )
                else:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"GPU setup warning in subprocess: {e}", file=sys.stderr)


def _cleanup_subprocess():
    """Cleanup before subprocess exits."""
    if HAS_TENSORFLOW:
        try:
            tf.keras.backend.clear_session()
        except:
            pass
    gc.collect()


# Public API

def get_memory_manager(use_process_isolation: bool = False, **kwargs) -> MemoryManager:
    """
    Create a memory manager.

    Args:
        use_process_isolation: Enable subprocess isolation (default: False)
        **kwargs: Additional options

    Returns:
        Configured MemoryManager
    """
    return MemoryManager(use_process_isolation=use_process_isolation, **kwargs)


@contextmanager
def managed_memory(use_process_isolation: bool = False, **kwargs):
    """
    Context manager for memory-managed operations.

    Examples:
        # Standard mode (default)
        with managed_memory():
            model.fit(...)

        # Process isolation for extended runs
        with managed_memory(use_process_isolation=True):
            for i in range(100):
                train_model()
    """
    manager = MemoryManager(use_process_isolation=use_process_isolation, **kwargs)

    if not use_process_isolation:
        try:
            manager.setup()
            yield manager
        finally:
            manager.cleanup()
    else:
        yield manager


def cleanup_gpu_memory() -> MemoryResult:
    """Quick GPU memory cleanup."""
    manager = MemoryManager(use_process_isolation=False)
    return manager.cleanup()


def get_memory_info() -> Dict[str, Any]:
    """Get current memory usage."""
    info = {}

    if HAS_PSUTIL:
        try:
            process = psutil.Process()
            info['process_rss_mb'] = process.memory_info().rss / (1024 ** 2)

            vm = psutil.virtual_memory()
            info['system_available_mb'] = vm.available / (1024 ** 2)
            info['system_percent'] = vm.percent
        except:
            pass

    if HAS_TENSORFLOW:
        try:
            for i, gpu in enumerate(tf.config.list_physical_devices('GPU')):
                stats = tf.config.experimental.get_memory_info(f'GPU:{i}')
                info[f'gpu{i}_current_mb'] = stats.get('current', 0) / (1024 ** 2)
        except:
            pass

    return info


def check_isolation_capability(func: Callable) -> Tuple[bool, str]:
    """
    Check if a function can be used with process isolation.

    Returns:
        (can_isolate, reason)
    """
    if HAS_CLOUDPICKLE:
        try:
            cloudpickle.dumps(func)
            return True, "cloudpickle serialization available"
        except Exception as e:
            return False, f"cloudpickle failed: {e}"
    else:
        try:
            pickle.dumps(func)
            return True, "standard pickle works"
        except:
            return False, "install cloudpickle for notebook functions"