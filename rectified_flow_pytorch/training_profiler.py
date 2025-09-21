"""
Training Pipeline Profiler for Rectified Flow

This module provides profiling utilities to measure the performance of different
components in the rectified flow training pipeline.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ProfileResult:
    """Container for profiling results."""
    name: str
    count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)

    def add_measurement(self, duration: float):
        """Add a timing measurement."""
        self.count += 1
        self.total_time += duration
        self.avg_time = self.total_time / self.count
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.times.append(duration)

    def summary(self) -> str:
        """Return a summary string."""
        return (".4f"
                ".4f")


class TrainingProfiler:
    """Profiler for training pipeline components."""

    def __init__(self, log_every: int = 100, detailed: bool = False):
        self.log_every = log_every
        self.detailed = detailed
        self.results: Dict[str, ProfileResult] = {}
        self.current_sections: List[str] = []
        self.start_times: Dict[str, float] = {}
        self.step_count = 0

    @contextmanager
    def profile_section(self, name: str):
        """Context manager to profile a section of code."""
        start_time = time.time()
        self.start_times[name] = start_time

        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time

            if name not in self.results:
                self.results[name] = ProfileResult(name=name)
            self.results[name].add_measurement(duration)

    def start_step(self):
        """Mark the start of a training step."""
        self.step_count += 1

    def end_step(self):
        """Mark the end of a training step and log if needed."""
        if self.step_count % self.log_every == 0:
            self.log_progress()

    def log_progress(self):
        """Log current profiling progress."""
        print(f"\n=== Profiling Report (Step {self.step_count}) ===")

        # Sort by total time (descending)
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].total_time,
            reverse=True
        )

        for name, result in sorted_results:
            if result.count > 0:
                print(result.summary())

        # Calculate percentages
        total_time = sum(r.total_time for r in self.results.values())
        if total_time > 0:
            print("\nTime Distribution:")
            for name, result in sorted_results:
                if result.count > 0:
                    pct = (result.total_time / total_time) * 100
                    print(f"{name}: {pct:.1f}%")

    def get_summary(self) -> str:
        """Get a complete profiling summary."""
        lines = [f"\n=== Final Profiling Summary ({self.step_count} steps) ==="]

        # Sort by total time
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1].total_time,
            reverse=True
        )

        lines.append("\nDetailed Breakdown:")
        for name, result in sorted_results:
            if result.count > 0:
                lines.append(result.summary())

        # Time distribution
        total_time = sum(r.total_time for r in self.results.values())
        if total_time > 0:
            lines.append("\nTime Distribution:")
            for name, result in sorted_results:
                if result.count > 0:
                    pct = (result.total_time / total_time) * 100
                    lines.append(f"{name}: {pct:.1f}%")

        # Performance insights
        lines.append("\nPerformance Insights:")
        if self.results:
            slowest = max(self.results.items(), key=lambda x: x[1].avg_time)
            lines.append(f"- Slowest component: {slowest[0]} ({slowest[1].avg_time:.4f}s avg)")

            # Check for potential bottlenecks
            data_loading = self.results.get('data_loading', ProfileResult('data_loading'))
            forward_pass = self.results.get('forward_pass', ProfileResult('forward_pass'))
            backward_pass = self.results.get('backward_pass', ProfileResult('backward_pass'))

            if data_loading.count > 0 and forward_pass.count > 0:
                data_ratio = data_loading.avg_time / (data_loading.avg_time + forward_pass.avg_time + backward_pass.avg_time)
                if data_ratio > 0.3:
                    lines.append(".1f")
        return "\n".join(lines)

    def reset(self):
        """Reset all profiling data."""
        self.results.clear()
        self.step_count = 0


# Global profiler instance
_profiler = None


def get_profiler(log_every: int = 100, detailed: bool = False) -> TrainingProfiler:
    """Get or create the global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = TrainingProfiler(log_every=log_every, detailed=detailed)
    return _profiler


def profile_section(name: str):
    """Decorator to profile a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            with profiler.profile_section(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Convenience functions for common profiling sections
def start_training_profiling(log_every: int = 100, detailed: bool = False):
    """Initialize training profiling."""
    global _profiler
    _profiler = TrainingProfiler(log_every=log_every, detailed=detailed)
    print(f"Training profiling enabled (log every {log_every} steps)")


def end_training_profiling():
    """End training profiling and print summary."""
    global _profiler
    if _profiler:
        summary = _profiler.get_summary()
        print(summary)
        _profiler = None


def profile_data_loading():
    """Context manager for profiling data loading."""
    return get_profiler().profile_section("data_loading")


def profile_forward_pass():
    """Context manager for profiling forward pass."""
    return get_profiler().profile_section("forward_pass")


def profile_backward_pass():
    """Context manager for profiling backward pass."""
    return get_profiler().profile_section("backward_pass")


def profile_optimizer_step():
    """Context manager for profiling optimizer step."""
    return get_profiler().profile_section("optimizer_step")


def profile_ema_update():
    """Context manager for profiling EMA updates."""
    return get_profiler().profile_section("ema_update")


def profile_metrics_computation():
    """Context manager for profiling metrics computation."""
    return get_profiler().profile_section("metrics_computation")


def profile_sampling():
    """Context manager for profiling sampling."""
    return get_profiler().profile_section("sampling")


def profile_checkpointing():
    """Context manager for profiling checkpointing."""
    return get_profiler().profile_section("checkpointing")


def profile_synchronization():
    """Context manager for profiling synchronization."""
    return get_profiler().profile_section("synchronization")
