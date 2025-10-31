"""
Performance Profiler

Performance monitoring and optimization tools for the adaptive language system.
Tracks processing times, memory usage, and identifies bottlenecks.
"""

import logging
import time
try:
    import psutil
except ImportError:
    psutil = None
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
import tracemalloc

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """
    Comprehensive performance profiling for adaptive language system
    
    Monitors processing times, memory usage, async operations,
    and provides optimization recommendations.
    """
    
    def __init__(self):
        self.operation_timings = defaultdict(list)
        self.memory_snapshots = deque(maxlen=100)
        self.async_operation_stats = defaultdict(dict)
        self.bottleneck_alerts = deque(maxlen=50)
        
        # Performance thresholds (ms)
        self.thresholds = {
            'semantic_analysis': 2000,
            'mood_detection': 500,
            'prompt_building': 1000,
            'pattern_learning': 300,
            'full_prompt_generation': 3000
        }
        
        # Memory monitoring
        self.memory_monitoring_enabled = False
        self.baseline_memory = None
        
        logger.info("⚡ Performance Profiler initialized")
    
    def enable_memory_monitoring(self):
        """Enable detailed memory monitoring"""
        try:
            tracemalloc.start()
            self.memory_monitoring_enabled = True
            self.baseline_memory = self._get_memory_snapshot()
            logger.info("📊 Memory monitoring enabled")
        except Exception as e:
            logger.warning(f"Could not enable memory monitoring: {e}")
    
    def disable_memory_monitoring(self):
        """Disable memory monitoring"""
        if self.memory_monitoring_enabled:
            tracemalloc.stop()
            self.memory_monitoring_enabled = False
            logger.info("📊 Memory monitoring disabled")
    
    @contextmanager
    def time_operation(self, operation_name: str, details: Optional[Dict] = None):
        """Context manager for timing operations"""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage() if self.memory_monitoring_enabled else None
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            end_memory = self._get_memory_usage() if self.memory_monitoring_enabled else None
            memory_delta = (end_memory - start_memory) if (start_memory and end_memory) else None
            
            # Record timing
            timing_data = {
                'duration_ms': duration_ms,
                'timestamp': datetime.now().isoformat(),
                'details': details or {},
                'memory_delta_mb': memory_delta / (1024 * 1024) if memory_delta else None
            }
            
            self.operation_timings[operation_name].append(timing_data)
            
            # Check for bottlenecks
            self._check_bottleneck(operation_name, duration_ms, details)
            
            # Limit stored timings
            if len(self.operation_timings[operation_name]) > 200:
                self.operation_timings[operation_name] = self.operation_timings[operation_name][-150:]
    
    @asynccontextmanager
    async def time_async_operation(self, operation_name: str, details: Optional[Dict] = None):
        """Async context manager for timing async operations"""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage() if self.memory_monitoring_enabled else None
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            end_memory = self._get_memory_usage() if self.memory_monitoring_enabled else None
            memory_delta = (end_memory - start_memory) if (start_memory and end_memory) else None
            
            # Record async timing
            timing_data = {
                'duration_ms': duration_ms,
                'timestamp': datetime.now().isoformat(),
                'details': details or {},
                'memory_delta_mb': memory_delta / (1024 * 1024) if memory_delta else None,
                'async': True
            }
            
            self.operation_timings[operation_name].append(timing_data)
            
            # Update async stats
            if operation_name not in self.async_operation_stats:
                self.async_operation_stats[operation_name] = {
                    'total_calls': 0,
                    'total_time_ms': 0,
                    'avg_time_ms': 0,
                    'concurrent_calls': 0
                }
            
            stats = self.async_operation_stats[operation_name]
            stats['total_calls'] += 1
            stats['total_time_ms'] += duration_ms
            stats['avg_time_ms'] = stats['total_time_ms'] / stats['total_calls']
            
            # Check for bottlenecks
            self._check_bottleneck(operation_name, duration_ms, details)
    
    def profile_function(self, operation_name: Optional[str] = None):
        """Decorator for profiling functions"""
        def decorator(func: Callable):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with self.time_async_operation(name):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.time_operation(name):
                        return func(*args, **kwargs)
                return sync_wrapper
        
        return decorator
    
    def take_memory_snapshot(self, label: str = ""):
        """Take a memory snapshot for analysis"""
        if not self.memory_monitoring_enabled:
            return None
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'label': label,
            'memory_usage_mb': self._get_memory_usage() / (1024 * 1024),
            'tracemalloc_snapshot': tracemalloc.take_snapshot()
        }
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        summary = {
            'analysis_period_hours': hours,
            'timestamp': datetime.now().isoformat(),
            'operation_performance': {},
            'memory_analysis': {},
            'bottleneck_summary': {},
            'optimization_recommendations': [],
            'system_resources': self._get_system_resources()
        }
        
        # Analyze each operation type
        for operation, timings in self.operation_timings.items():
            recent_timings = [
                t for t in timings 
                if datetime.fromisoformat(t['timestamp']) > cutoff_time
            ]
            
            if recent_timings:
                durations = [t['duration_ms'] for t in recent_timings]
                memory_deltas = [t['memory_delta_mb'] for t in recent_timings if t['memory_delta_mb']]
                
                performance_data = {
                    'call_count': len(recent_timings),
                    'avg_duration_ms': sum(durations) / len(durations),
                    'min_duration_ms': min(durations),
                    'max_duration_ms': max(durations),
                    'p95_duration_ms': self._percentile(durations, 95),
                    'threshold_ms': self.thresholds.get(operation, 1000),
                    'threshold_violations': sum(1 for d in durations if d > self.thresholds.get(operation, 1000)),
                    'calls_per_hour': len(recent_timings) / hours
                }
                
                if memory_deltas:
                    performance_data['avg_memory_delta_mb'] = sum(memory_deltas) / len(memory_deltas)
                    performance_data['max_memory_delta_mb'] = max(memory_deltas)
                
                summary['operation_performance'][operation] = performance_data
        
        # Memory analysis
        if self.memory_snapshots:
            recent_snapshots = [
                s for s in self.memory_snapshots
                if datetime.fromisoformat(s['timestamp']) > cutoff_time
            ]
            
            if recent_snapshots:
                memory_usages = [s['memory_usage_mb'] for s in recent_snapshots]
                summary['memory_analysis'] = {
                    'current_usage_mb': memory_usages[-1] if memory_usages else None,
                    'peak_usage_mb': max(memory_usages),
                    'avg_usage_mb': sum(memory_usages) / len(memory_usages),
                    'memory_growth_mb': memory_usages[-1] - memory_usages[0] if len(memory_usages) > 1 else 0,
                    'snapshots_taken': len(recent_snapshots)
                }
        
        # Bottleneck analysis
        recent_bottlenecks = [
            b for b in self.bottleneck_alerts
            if datetime.fromisoformat(b['timestamp']) > cutoff_time
        ]
        
        if recent_bottlenecks:
            bottleneck_counts = defaultdict(int)
            for bottleneck in recent_bottlenecks:
                bottleneck_counts[bottleneck['operation']] += 1
            
            summary['bottleneck_summary'] = {
                'total_bottlenecks': len(recent_bottlenecks),
                'bottleneck_operations': dict(bottleneck_counts),
                'most_problematic': max(bottleneck_counts.items(), key=lambda x: x[1])[0] if bottleneck_counts else None
            }
        
        # Generate optimization recommendations
        summary['optimization_recommendations'] = self._generate_optimization_recommendations(summary)
        
        return summary
    
    def get_operation_analysis(self, operation_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get detailed analysis for a specific operation"""
        
        if operation_name not in self.operation_timings:
            return {"status": "no_data", "operation": operation_name}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_timings = [
            t for t in self.operation_timings[operation_name]
            if datetime.fromisoformat(t['timestamp']) > cutoff_time
        ]
        
        if not recent_timings:
            return {"status": "no_recent_data", "operation": operation_name, "period_hours": hours}
        
        durations = [t['duration_ms'] for t in recent_timings]
        
        analysis = {
            'operation': operation_name,
            'period_hours': hours,
            'statistics': {
                'call_count': len(recent_timings),
                'avg_duration_ms': sum(durations) / len(durations),
                'median_duration_ms': self._percentile(durations, 50),
                'p95_duration_ms': self._percentile(durations, 95),
                'p99_duration_ms': self._percentile(durations, 99),
                'min_duration_ms': min(durations),
                'max_duration_ms': max(durations),
                'std_deviation_ms': self._std_deviation(durations)
            },
            'performance_trend': self._calculate_performance_trend(recent_timings),
            'threshold_analysis': {
                'threshold_ms': self.thresholds.get(operation_name, 1000),
                'violations': sum(1 for d in durations if d > self.thresholds.get(operation_name, 1000)),
                'violation_rate': sum(1 for d in durations if d > self.thresholds.get(operation_name, 1000)) / len(durations)
            },
            'outlier_analysis': self._analyze_outliers(recent_timings),
            'recommendations': self._get_operation_recommendations(operation_name, recent_timings)
        }
        
        # Add memory analysis if available
        memory_deltas = [t['memory_delta_mb'] for t in recent_timings if t['memory_delta_mb']]
        if memory_deltas:
            analysis['memory_analysis'] = {
                'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
                'max_memory_delta_mb': max(memory_deltas),
                'memory_efficiency_score': self._calculate_memory_efficiency(durations, memory_deltas)
            }
        
        return analysis
    
    def identify_bottlenecks(self, threshold_multiplier: float = 2.0) -> List[Dict[str, Any]]:
        """Identify current system bottlenecks"""
        
        bottlenecks = []
        
        for operation, timings in self.operation_timings.items():
            if not timings:
                continue
            
            recent_timings = timings[-20:]  # Last 20 calls
            durations = [t['duration_ms'] for t in recent_timings]
            
            if not durations:
                continue
            
            avg_duration = sum(durations) / len(durations)
            threshold = self.thresholds.get(operation, 1000)
            
            # Check if operation is consistently slow
            if avg_duration > threshold * threshold_multiplier:
                bottleneck = {
                    'operation': operation,
                    'avg_duration_ms': avg_duration,
                    'threshold_ms': threshold,
                    'severity': min(5.0, avg_duration / threshold),
                    'recent_calls': len(recent_timings),
                    'recommendation': self._get_bottleneck_recommendation(operation, avg_duration, threshold)
                }
                bottlenecks.append(bottleneck)
        
        # Sort by severity
        bottlenecks.sort(key=lambda x: x['severity'], reverse=True)
        
        return bottlenecks
    
    def optimize_thresholds(self):
        """Automatically optimize performance thresholds based on historical data"""
        
        for operation, timings in self.operation_timings.items():
            if len(timings) < 50:  # Need enough data
                continue
            
            durations = [t['duration_ms'] for t in timings[-200:]]  # Recent data
            
            # Set threshold at 95th percentile
            p95_duration = self._percentile(durations, 95)
            
            # Don't make thresholds too strict or too loose
            current_threshold = self.thresholds.get(operation, 1000)
            new_threshold = max(min(p95_duration * 1.2, current_threshold * 2), current_threshold * 0.5)
            
            if abs(new_threshold - current_threshold) > current_threshold * 0.1:  # 10% change threshold
                self.thresholds[operation] = new_threshold
                logger.info(f"⚡ Updated threshold for {operation}: {current_threshold:.0f}ms → {new_threshold:.0f}ms")
    
    def _check_bottleneck(self, operation_name: str, duration_ms: float, details: Optional[Dict]):
        """Check if operation represents a bottleneck"""
        
        threshold = self.thresholds.get(operation_name, 1000)
        
        if duration_ms > threshold * 1.5:  # 1.5x threshold triggers alert
            alert = {
                'timestamp': datetime.now().isoformat(),
                'operation': operation_name,
                'duration_ms': duration_ms,
                'threshold_ms': threshold,
                'severity': duration_ms / threshold,
                'details': details or {}
            }
            
            self.bottleneck_alerts.append(alert)
            
            logger.warning(f"⚡ BOTTLENECK: {operation_name} took {duration_ms:.0f}ms (threshold: {threshold:.0f}ms)")
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except:
            return 0
    
    def _get_memory_snapshot(self) -> Dict[str, Any]:
        """Get detailed memory snapshot"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            snapshot = {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent()
            }
            
            if self.memory_monitoring_enabled:
                tracemalloc_snapshot = tracemalloc.take_snapshot()
                top_stats = tracemalloc_snapshot.statistics('lineno')
                
                snapshot['top_memory_lines'] = [
                    {
                        'filename': stat.traceback.format()[0] if stat.traceback.format() else 'unknown',
                        'size_mb': stat.size / (1024 * 1024),
                        'count': stat.count
                    }
                    for stat in top_stats[:5]
                ]
            
            return snapshot
            
        except Exception as e:
            logger.warning(f"Could not get memory snapshot: {e}")
            return {}
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'disk_usage_percent': psutil.disk_usage('/').percent if psutil.disk_usage('/') else None,
                'process_count': len(psutil.pids())
            }
        except Exception as e:
            logger.warning(f"Could not get system resources: {e}")
            return {}
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (percentile / 100)
        f = int(k)
        c = k - f
        
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        else:
            return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    def _std_deviation(self, data: List[float]) -> float:
        """Calculate standard deviation"""
        if len(data) < 2:
            return 0.0
        
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return variance ** 0.5
    
    def _calculate_performance_trend(self, timings: List[Dict]) -> str:
        """Calculate performance trend over time"""
        if len(timings) < 10:
            return "insufficient_data"
        
        # Split into first and second half
        mid_point = len(timings) // 2
        first_half = [t['duration_ms'] for t in timings[:mid_point]]
        second_half = [t['duration_ms'] for t in timings[mid_point:]]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        change_percent = ((second_avg - first_avg) / first_avg) * 100
        
        if change_percent > 10:
            return "degrading"
        elif change_percent < -10:
            return "improving"
        else:
            return "stable"
    
    def _analyze_outliers(self, timings: List[Dict]) -> Dict[str, Any]:
        """Analyze outlier operations"""
        durations = [t['duration_ms'] for t in timings]
        
        if len(durations) < 10:
            return {"status": "insufficient_data"}
        
        q1 = self._percentile(durations, 25)
        q3 = self._percentile(durations, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [t for t in timings if t['duration_ms'] < lower_bound or t['duration_ms'] > upper_bound]
        
        return {
            'outlier_count': len(outliers),
            'outlier_rate': len(outliers) / len(timings),
            'upper_bound_ms': upper_bound,
            'lower_bound_ms': lower_bound,
            'worst_outlier_ms': max(o['duration_ms'] for o in outliers) if outliers else None
        }
    
    def _calculate_memory_efficiency(self, durations: List[float], memory_deltas: List[float]) -> float:
        """Calculate memory efficiency score"""
        if not durations or not memory_deltas:
            return 0.0
        
        # Lower memory usage per unit time is better
        avg_duration = sum(durations) / len(durations)
        avg_memory = sum(abs(delta) for delta in memory_deltas) / len(memory_deltas)
        
        if avg_duration == 0:
            return 1.0
        
        # Normalize: lower memory per ms is better (higher score)
        efficiency = max(0.0, 1.0 - (avg_memory / (avg_duration / 1000)))
        return min(1.0, efficiency)
    
    def _generate_optimization_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on performance data"""
        recommendations = []
        
        # Check operation performance
        for operation, perf_data in summary.get('operation_performance', {}).items():
            violation_rate = perf_data.get('threshold_violations', 0) / max(perf_data.get('call_count', 1), 1)
            
            if violation_rate > 0.2:  # More than 20% threshold violations
                recommendations.append(f"Optimize {operation} - {violation_rate*100:.0f}% of calls exceed threshold")
            
            if perf_data.get('avg_duration_ms', 0) > perf_data.get('threshold_ms', 1000) * 1.5:
                recommendations.append(f"Critical: {operation} is consistently slow (avg: {perf_data['avg_duration_ms']:.0f}ms)")
        
        # Check memory usage
        memory_analysis = summary.get('memory_analysis', {})
        if memory_analysis.get('memory_growth_mb', 0) > 50:  # More than 50MB growth
            recommendations.append("Monitor memory usage - significant growth detected")
        
        # Check system resources
        system_resources = summary.get('system_resources', {})
        if system_resources.get('memory_percent', 0) > 80:
            recommendations.append("System memory usage is high - consider resource optimization")
        
        if system_resources.get('cpu_percent', 0) > 80:
            recommendations.append("High CPU usage detected - check for blocking operations")
        
        # Check bottlenecks
        bottleneck_summary = summary.get('bottleneck_summary', {})
        if bottleneck_summary.get('total_bottlenecks', 0) > 10:
            recommendations.append("Multiple bottlenecks detected - comprehensive optimization needed")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable parameters")
        
        return recommendations
    
    def _get_operation_recommendations(self, operation_name: str, timings: List[Dict]) -> List[str]:
        """Get specific recommendations for an operation"""
        recommendations = []
        
        durations = [t['duration_ms'] for t in timings]
        avg_duration = sum(durations) / len(durations)
        threshold = self.thresholds.get(operation_name, 1000)
        
        if avg_duration > threshold * 2:
            recommendations.append("Critical performance issue - immediate optimization required")
        elif avg_duration > threshold * 1.5:
            recommendations.append("Performance optimization recommended")
        
        # Check for high variance
        std_dev = self._std_deviation(durations)
        if std_dev > avg_duration * 0.5:
            recommendations.append("High variability in performance - investigate inconsistent behavior")
        
        # Check memory if available
        memory_deltas = [t['memory_delta_mb'] for t in timings if t['memory_delta_mb']]
        if memory_deltas:
            avg_memory = sum(abs(delta) for delta in memory_deltas) / len(memory_deltas)
            if avg_memory > 10:  # More than 10MB per operation
                recommendations.append("High memory usage per operation - check for memory leaks")
        
        return recommendations if recommendations else ["Performance is within acceptable range"]
    
    def _get_bottleneck_recommendation(self, operation: str, avg_duration: float, threshold: float) -> str:
        """Get recommendation for a specific bottleneck"""
        severity = avg_duration / threshold
        
        if severity > 3:
            return f"Critical bottleneck - {operation} needs immediate optimization"
        elif severity > 2:
            return f"Significant performance issue in {operation} - optimization recommended"
        else:
            return f"Monitor {operation} performance - may need tuning"


# Global profiler instance
_global_profiler = None

def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler