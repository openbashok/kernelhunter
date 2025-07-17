#!/usr/bin/env python3
"""
Advanced Performance Optimizer for KernelHunter
Implements async processing, memory management, and I/O optimization
"""

import asyncio
import aiohttp
import aiofiles
import psutil
import gc
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Optional, Callable
import multiprocessing as mp
from dataclasses import dataclass
import queue
import weakref
from pathlib import Path
import json
import logging
from collections import deque
import numpy as np

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    max_memory_mb: int = 2048
    max_workers: int = mp.cpu_count()
    io_buffer_size: int = 8192
    async_batch_size: int = 100
    gc_threshold: int = 1000
    checkpoint_interval: int = 50
    enable_profiling: bool = True

class MemoryManager:
    """Advanced memory management with garbage collection and pooling"""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_usage = 0
        self.object_pool = weakref.WeakValueDictionary()
        self.large_objects = deque(maxlen=100)
        self.gc_counter = 0
        
    def check_memory_usage(self) -> float:
        """Check current memory usage as percentage"""
        process = psutil.Process()
        self.current_usage = process.memory_info().rss
        return self.current_usage / self.max_memory_bytes
    
    def should_gc(self) -> bool:
        """Determine if garbage collection is needed"""
        usage = self.check_memory_usage()
        self.gc_counter += 1
        
        return (usage > 0.8 or 
                self.gc_counter > 1000 or 
                len(self.large_objects) > 50)
    
    def optimize_memory(self):
        """Perform memory optimization"""
        if self.should_gc():
            # Clear object pool
            self.object_pool.clear()
            
            # Clear large objects cache
            self.large_objects.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Reset counter
            self.gc_counter = 0
            
            logging.info(f"Memory optimization completed. Usage: {self.check_memory_usage():.2%}")
    
    def cache_object(self, key: str, obj: Any):
        """Cache object in memory pool"""
        if self.check_memory_usage() < 0.9:
            self.object_pool[key] = obj
        else:
            self.optimize_memory()
    
    def get_cached_object(self, key: str) -> Optional[Any]:
        """Retrieve cached object"""
        return self.object_pool.get(key)

class AsyncIOManager:
    """Asynchronous I/O manager with buffering and batching"""
    
    def __init__(self, buffer_size: int = 8192, batch_size: int = 100):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.write_queue = asyncio.Queue(maxsize=1000)
        self.read_cache = {}
        self.write_buffers = {}
        
    async def write_file_async(self, filepath: str, data: str, mode: str = 'a'):
        """Write to file asynchronously with buffering"""
        await self.write_queue.put((filepath, data, mode))
    
    async def write_batch_async(self, filepath: str, data_list: List[str], mode: str = 'a'):
        """Write batch of data asynchronously"""
        async with aiofiles.open(filepath, mode, encoding='utf-8') as f:
            for data in data_list:
                await f.write(data + '\n')
            await f.flush()
    
    async def read_file_async(self, filepath: str) -> str:
        """Read file asynchronously with caching"""
        if filepath in self.read_cache:
            return self.read_cache[filepath]
        
        async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
            content = await f.read()
            self.read_cache[filepath] = content
            return content
    
    async def process_write_queue(self):
        """Process write queue in background"""
        while True:
            try:
                batch = []
                for _ in range(self.batch_size):
                    try:
                        item = await asyncio.wait_for(self.write_queue.get(), timeout=1.0)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    # Group by filepath
                    file_groups = {}
                    for filepath, data, mode in batch:
                        if filepath not in file_groups:
                            file_groups[filepath] = []
                        file_groups[filepath].append(data)
                    
                    # Write each file
                    for filepath, data_list in file_groups.items():
                        await self.write_batch_async(filepath, data_list, 'a')
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Error in write queue processing: {e}")
                await asyncio.sleep(1)

class AsyncProcessor:
    """Advanced async processor with load balancing and error handling"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.semaphore = asyncio.Semaphore(self.max_workers)
        self.task_queue = asyncio.Queue()
        self.results = {}
        self.errors = {}
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'avg_processing_time': 0
        }
    
    async def submit_task(self, task_id: str, coro_func: Callable, *args, **kwargs):
        """Submit task for async processing"""
        await self.task_queue.put((task_id, coro_func, args, kwargs))
    
    async def process_task(self, task_id: str, coro_func: Callable, args: tuple, kwargs: dict):
        """Process individual task with error handling"""
        start_time = time.time()
        
        try:
            async with self.semaphore:
                result = await coro_func(*args, **kwargs)
                self.results[task_id] = result
                self.stats['tasks_completed'] += 1
                
        except Exception as e:
            self.errors[task_id] = str(e)
            self.stats['tasks_failed'] += 1
            logging.error(f"Task {task_id} failed: {e}")
        
        finally:
            processing_time = time.time() - start_time
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['tasks_completed'] - 1) + processing_time) /
                self.stats['tasks_completed']
            )
    
    async def process_queue(self):
        """Process task queue continuously"""
        while True:
            try:
                task_id, coro_func, args, kwargs = await self.task_queue.get()
                await self.process_task(task_id, coro_func, args, kwargs)
                self.task_queue.task_done()
                
            except Exception as e:
                logging.error(f"Error in queue processing: {e}")
                await asyncio.sleep(1)
    
    async def wait_for_completion(self, timeout: float = None):
        """Wait for all tasks to complete"""
        await self.task_queue.join()
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return self.stats.copy()

class ProcessPoolManager:
    """Advanced process pool manager with load balancing"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers * 2)
        self.task_futures = {}
        self.stats = {
            'processes_created': 0,
            'tasks_submitted': 0,
            'tasks_completed': 0
        }
    
    async def submit_process_task(self, func: Callable, *args, **kwargs):
        """Submit task to process pool"""
        loop = asyncio.get_event_loop()
        future = await loop.run_in_executor(self.executor, func, *args, **kwargs)
        self.stats['tasks_submitted'] += 1
        return future
    
    async def submit_thread_task(self, func: Callable, *args, **kwargs):
        """Submit task to thread pool"""
        loop = asyncio.get_event_loop()
        future = await loop.run_in_executor(self.thread_executor, func, *args, **kwargs)
        return future
    
    def shutdown(self):
        """Shutdown executors"""
        self.executor.shutdown(wait=True)
        self.thread_executor.shutdown(wait=True)

class PerformanceProfiler:
    """Performance profiler with detailed metrics"""
    
    def __init__(self, enable_profiling: bool = True):
        self.enable_profiling = enable_profiling
        self.metrics = {
            'cpu_usage': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'io_operations': deque(maxlen=1000),
            'processing_times': deque(maxlen=1000)
        }
        self.start_time = time.time()
        
    def record_metric(self, metric_type: str, value: float):
        """Record performance metric"""
        if self.enable_profiling:
            self.metrics[metric_type].append((time.time(), value))
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        return psutil.cpu_percent(interval=0.1)
    
    def get_memory_usage(self) -> Dict:
        """Get detailed memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss,
            'vms': memory_info.vms,
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available
        }
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate averages
        cpu_avg = np.mean([v for _, v in self.metrics['cpu_usage']]) if self.metrics['cpu_usage'] else 0
        memory_avg = np.mean([v for _, v in self.metrics['memory_usage']]) if self.metrics['memory_usage'] else 0
        io_avg = np.mean([v for _, v in self.metrics['io_operations']]) if self.metrics['io_operations'] else 0
        processing_avg = np.mean([v for _, v in self.metrics['processing_times']]) if self.metrics['processing_times'] else 0
        
        return {
            'uptime_seconds': uptime,
            'current_cpu_usage': self.get_cpu_usage(),
            'current_memory_usage': self.get_memory_usage(),
            'average_cpu_usage': cpu_avg,
            'average_memory_usage': memory_avg,
            'average_io_operations': io_avg,
            'average_processing_time': processing_avg,
            'total_metrics_recorded': sum(len(metric) for metric in self.metrics.values())
        }

class AdvancedPerformanceOptimizer:
    """Main performance optimizer class"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.memory_manager = MemoryManager(config.max_memory_mb)
        self.io_manager = AsyncIOManager(config.io_buffer_size, config.async_batch_size)
        self.async_processor = AsyncProcessor(config.max_workers)
        self.process_pool = ProcessPoolManager(config.max_workers)
        self.profiler = PerformanceProfiler(config.enable_profiling)
        
        # Background tasks
        self.background_tasks = []
        self.running = False
        
    async def start(self):
        """Start the performance optimizer"""
        self.running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self.io_manager.process_write_queue()),
            asyncio.create_task(self.async_processor.process_queue()),
            asyncio.create_task(self._monitor_performance())
        ]
        
        logging.info("Performance optimizer started")
    
    async def stop(self):
        """Stop the performance optimizer"""
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for completion
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown process pool
        self.process_pool.shutdown()
        
        logging.info("Performance optimizer stopped")
    
    async def _monitor_performance(self):
        """Monitor system performance and optimize"""
        while self.running:
            try:
                # Record metrics
                cpu_usage = self.profiler.get_cpu_usage()
                memory_usage = self.profiler.get_memory_usage()
                
                self.profiler.record_metric('cpu_usage', cpu_usage)
                self.profiler.record_metric('memory_usage', memory_usage['percent'])
                
                # Optimize memory if needed
                if self.memory_manager.should_gc():
                    self.memory_manager.optimize_memory()
                
                # Log performance stats periodically
                if len(self.profiler.metrics['cpu_usage']) % 100 == 0:
                    report = self.profiler.get_performance_report()
                    logging.info(f"Performance Report: {report}")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logging.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(5)
    
    async def optimize_shellcode_processing(self, shellcodes: List[bytes]) -> List[bytes]:
        """Optimize shellcode processing with parallel execution"""
        if not shellcodes:
            return []
        
        # Split into batches
        batch_size = max(1, len(shellcodes) // self.config.max_workers)
        batches = [shellcodes[i:i + batch_size] for i in range(0, len(shellcodes), batch_size)]
        
        # Process batches in parallel
        tasks = []
        for i, batch in enumerate(batches):
            task = self.async_processor.submit_task(
                f"batch_{i}",
                self._process_shellcode_batch,
                batch
            )
            tasks.append(task)
        
        # Wait for completion
        await self.async_processor.wait_for_completion()
        
        # Collect results
        results = []
        for task_id in [f"batch_{i}" for i in range(len(batches))]:
            if task_id in self.async_processor.results:
                results.extend(self.async_processor.results[task_id])
        
        return results
    
    async def _process_shellcode_batch(self, shellcodes: List[bytes]) -> List[bytes]:
        """Process a batch of shellcodes"""
        start_time = time.time()
        
        # Process shellcodes (placeholder for actual processing)
        processed = []
        for shellcode in shellcodes:
            # Add processing logic here
            processed.append(shellcode)
        
        # Record processing time
        processing_time = time.time() - start_time
        self.profiler.record_metric('processing_times', processing_time)
        
        return processed
    
    async def write_metrics_async(self, metrics: Dict, filepath: str):
        """Write metrics asynchronously"""
        await self.io_manager.write_file_async(
            filepath,
            json.dumps(metrics) + '\n'
        )
    
    def get_optimization_stats(self) -> Dict:
        """Get optimization statistics"""
        return {
            'memory_usage_percent': self.memory_manager.check_memory_usage(),
            'async_processor_stats': self.async_processor.get_stats(),
            'performance_report': self.profiler.get_performance_report(),
            'io_queue_size': self.io_manager.write_queue.qsize()
        }

# Global performance optimizer instance
performance_optimizer = None

def get_performance_optimizer() -> AdvancedPerformanceOptimizer:
    """Get or create global performance optimizer instance"""
    global performance_optimizer
    if performance_optimizer is None:
        config = PerformanceConfig()
        performance_optimizer = AdvancedPerformanceOptimizer(config)
    return performance_optimizer

async def optimize_kernelhunter_performance():
    """Main function to optimize KernelHunter performance"""
    optimizer = get_performance_optimizer()
    
    try:
        await optimizer.start()
        
        # Example usage
        test_shellcodes = [b"\x90\x90\x48\x31\xc0\x0f\x05"] * 1000
        processed = await optimizer.optimize_shellcode_processing(test_shellcodes)
        
        # Write metrics
        stats = optimizer.get_optimization_stats()
        await optimizer.write_metrics_async(stats, "performance_metrics.json")
        
        print(f"Processed {len(processed)} shellcodes")
        print(f"Performance stats: {stats}")
        
    finally:
        await optimizer.stop()

if __name__ == "__main__":
    # Test the performance optimizer
    asyncio.run(optimize_kernelhunter_performance()) 