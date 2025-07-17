#!/usr/bin/env python3
"""
Real-time Analytics System for KernelHunter
Implements stream processing, anomaly detection, and advanced visualization
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque, defaultdict
import threading
import queue
import websockets
import redis
from datetime import datetime, timedelta
import hashlib

@dataclass
class AnalyticsConfig:
    """Configuration for analytics system"""
    stream_buffer_size: int = 1000
    anomaly_detection_window: int = 100
    update_interval: float = 1.0
    redis_host: str = "localhost"
    redis_port: int = 6379
    websocket_port: int = 8765
    dashboard_port: int = 8050

class StreamProcessor:
    """Real-time stream processor for KernelHunter data"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.data_stream = deque(maxlen=buffer_size)
        self.processing_queue = asyncio.Queue()
        self.subscribers = []
        self.running = False
        
    async def add_data_point(self, data: Dict):
        """Add data point to stream"""
        timestamp = time.time()
        data_point = {
            "timestamp": timestamp,
            "data": data
        }
        
        self.data_stream.append(data_point)
        await self.processing_queue.put(data_point)
        
        # Notify subscribers
        await self._notify_subscribers(data_point)
    
    async def _notify_subscribers(self, data_point: Dict):
        """Notify all subscribers of new data"""
        for subscriber in self.subscribers:
            try:
                await subscriber(data_point)
            except Exception as e:
                logging.error(f"Error notifying subscriber: {e}")
    
    def subscribe(self, callback):
        """Subscribe to data stream"""
        self.subscribers.append(callback)
    
    def get_recent_data(self, window_size: int = None) -> List[Dict]:
        """Get recent data from stream"""
        if window_size is None:
            window_size = len(self.data_stream)
        
        return list(self.data_stream)[-window_size:]
    
    async def process_stream(self):
        """Process data stream in background"""
        self.running = True
        
        while self.running:
            try:
                # Process data points
                while not self.processing_queue.empty():
                    data_point = await self.processing_queue.get()
                    await self._process_data_point(data_point)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Error in stream processing: {e}")
                await asyncio.sleep(1)
    
    async def _process_data_point(self, data_point: Dict):
        """Process individual data point"""
        # Extract features for analysis
        features = self._extract_features(data_point["data"])
        
        # Store processed data
        processed_data = {
            "timestamp": data_point["timestamp"],
            "features": features,
            "raw_data": data_point["data"]
        }
        
        # Add to analytics engine
        await analytics_engine.add_processed_data(processed_data)
    
    def _extract_features(self, data: Dict) -> Dict:
        """Extract features from raw data"""
        features = {}
        
        # Crash-related features
        features["crash_rate"] = data.get("crash_rate", 0)
        features["system_impacts"] = data.get("system_impacts", 0)
        features["total_crashes"] = data.get("total_crashes", 0)
        
        # Performance features
        features["avg_shellcode_length"] = data.get("avg_shellcode_length", 0)
        features["generation_time"] = data.get("generation_time", 0)
        features["population_size"] = data.get("population_size", 0)
        
        # Diversity features
        features["diversity_score"] = data.get("diversity_score", 0)
        features["unique_crash_types"] = data.get("unique_crash_types", 0)
        
        # ML features
        features["rl_epsilon"] = data.get("rl_epsilon", 0)
        features["attack_success_rate"] = data.get("attack_success_rate", 0)
        features["mutation_success_rate"] = data.get("mutation_success_rate", 0)
        
        return features

class AnomalyDetector:
    """Advanced anomaly detection using multiple algorithms"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.feature_history = deque(maxlen=window_size)
        self.anomaly_scores = deque(maxlen=window_size)
        self.threshold = 0.8
        
    def add_data_point(self, features: Dict) -> Tuple[bool, float]:
        """Add data point and detect anomalies"""
        # Convert features to array
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        # Add to history
        self.feature_history.append(feature_array.flatten())
        
        # Detect anomaly if we have enough data
        if len(self.feature_history) >= self.window_size:
            return self._detect_anomaly(feature_array)
        else:
            return False, 0.0
    
    def _detect_anomaly(self, feature_array: np.ndarray) -> Tuple[bool, float]:
        """Detect anomaly using isolation forest"""
        try:
            # Fit model if needed
            if len(self.feature_history) == self.window_size:
                history_array = np.array(list(self.feature_history))
                self.scaler.fit(history_array)
                self.isolation_forest.fit(self.scaler.transform(history_array))
            
            # Transform and predict
            transformed = self.scaler.transform(feature_array)
            score = self.isolation_forest.decision_function(transformed)[0]
            anomaly_score = 1 - score  # Convert to anomaly score
            
            # Store score
            self.anomaly_scores.append(anomaly_score)
            
            # Determine if anomaly
            is_anomaly = anomaly_score > self.threshold
            
            return is_anomaly, anomaly_score
            
        except Exception as e:
            logging.error(f"Error in anomaly detection: {e}")
            return False, 0.0
    
    def get_anomaly_stats(self) -> Dict:
        """Get anomaly detection statistics"""
        if not self.anomaly_scores:
            return {"total_anomalies": 0, "avg_score": 0, "recent_anomalies": 0}
        
        scores = list(self.anomaly_scores)
        recent_scores = scores[-10:] if len(scores) >= 10 else scores
        
        return {
            "total_anomalies": sum(1 for s in scores if s > self.threshold),
            "avg_score": np.mean(scores),
            "recent_anomalies": sum(1 for s in recent_scores if s > self.threshold),
            "threshold": self.threshold
        }

class RealTimeAnalytics:
    """Main real-time analytics engine"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.stream_processor = StreamProcessor(config.stream_buffer_size)
        self.anomaly_detector = AnomalyDetector(config.anomaly_detection_window)
        
        # Data storage
        self.processed_data = deque(maxlen=10000)
        self.anomalies = deque(maxlen=1000)
        self.metrics = defaultdict(list)
        
        # Redis connection for distributed analytics
        self.redis_client = None
        self._init_redis()
        
        # Background tasks
        self.background_tasks = []
        self.running = False
        
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True
            )
            self.redis_client.ping()
            logging.info("Connected to Redis")
        except Exception as e:
            logging.warning(f"Could not connect to Redis: {e}")
            self.redis_client = None
    
    async def start(self):
        """Start the analytics engine"""
        self.running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self.stream_processor.process_stream()),
            asyncio.create_task(self._metrics_aggregation_loop()),
            asyncio.create_task(self._anomaly_monitoring_loop())
        ]
        
        logging.info("Real-time analytics engine started")
    
    async def stop(self):
        """Stop the analytics engine"""
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for completion
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logging.info("Real-time analytics engine stopped")
    
    async def add_processed_data(self, data: Dict):
        """Add processed data to analytics engine"""
        self.processed_data.append(data)
        
        # Detect anomalies
        is_anomaly, score = self.anomaly_detector.add_data_point(data["features"])
        
        if is_anomaly:
            anomaly_data = {
                "timestamp": data["timestamp"],
                "score": score,
                "features": data["features"],
                "raw_data": data["raw_data"]
            }
            self.anomalies.append(anomaly_data)
            
            # Publish to Redis if available
            if self.redis_client:
                self.redis_client.publish("kernelhunter_anomalies", json.dumps(anomaly_data))
        
        # Update metrics
        for key, value in data["features"].items():
            self.metrics[key].append({
                "timestamp": data["timestamp"],
                "value": value
            })
    
    async def _metrics_aggregation_loop(self):
        """Aggregate metrics in background"""
        while self.running:
            try:
                # Aggregate metrics every minute
                await asyncio.sleep(60)
                
                # Calculate aggregated metrics
                aggregated = self._calculate_aggregated_metrics()
                
                # Store in Redis if available
                if self.redis_client:
                    self.redis_client.setex(
                        "kernelhunter_metrics",
                        300,  # 5 minutes TTL
                        json.dumps(aggregated)
                    )
                
            except Exception as e:
                logging.error(f"Error in metrics aggregation: {e}")
                await asyncio.sleep(60)
    
    async def _anomaly_monitoring_loop(self):
        """Monitor anomalies in background"""
        while self.running:
            try:
                # Check for new anomalies every 10 seconds
                await asyncio.sleep(10)
                
                # Get anomaly stats
                stats = self.anomaly_detector.get_anomaly_stats()
                
                # Alert if too many anomalies
                if stats["recent_anomalies"] > 5:
                    logging.warning(f"High anomaly rate detected: {stats['recent_anomalies']} in last 10 data points")
                
            except Exception as e:
                logging.error(f"Error in anomaly monitoring: {e}")
                await asyncio.sleep(30)
    
    def _calculate_aggregated_metrics(self) -> Dict:
        """Calculate aggregated metrics from recent data"""
        if not self.processed_data:
            return {}
        
        recent_data = list(self.processed_data)[-100:]  # Last 100 data points
        
        aggregated = {}
        for metric_name in self.metrics.keys():
            values = [d["features"].get(metric_name, 0) for d in recent_data]
            if values:
                aggregated[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }
        
        return aggregated
    
    def get_analytics_summary(self) -> Dict:
        """Get comprehensive analytics summary"""
        return {
            "total_data_points": len(self.processed_data),
            "total_anomalies": len(self.anomalies),
            "anomaly_stats": self.anomaly_detector.get_anomaly_stats(),
            "recent_metrics": self._calculate_aggregated_metrics(),
            "stream_stats": {
                "buffer_size": len(self.stream_processor.data_stream),
                "queue_size": self.stream_processor.processing_queue.qsize()
            }
        }

class WebSocketServer:
    """WebSocket server for real-time data streaming"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.clients = set()
        self.server = None
        
    async def start(self):
        """Start WebSocket server"""
        self.server = await websockets.serve(
            self._handle_client,
            "localhost",
            self.port
        )
        logging.info(f"WebSocket server started on port {self.port}")
    
    async def stop(self):
        """Stop WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
    
    async def _handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                # Handle client messages if needed
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
    
    async def broadcast(self, data: Dict):
        """Broadcast data to all connected clients"""
        if not self.clients:
            return
        
        message = json.dumps(data)
        await asyncio.gather(
            *[client.send(message) for client in self.clients],
            return_exceptions=True
        )

class DashboardApp:
    """Interactive dashboard using Dash"""
    
    def __init__(self, port: int = 8050):
        self.port = port
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("KernelHunter Real-Time Analytics", 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            # Summary cards
            html.Div([
                html.Div([
                    html.H3(id='total-crashes'),
                    html.P("Total Crashes")
                ], className='card'),
                html.Div([
                    html.H3(id='anomaly-count'),
                    html.P("Anomalies Detected")
                ], className='card'),
                html.Div([
                    html.H3(id='crash-rate'),
                    html.P("Current Crash Rate")
                ], className='card'),
                html.Div([
                    html.H3(id='system-impacts'),
                    html.P("System Impacts")
                ], className='card')
            ], className='summary-cards'),
            
            # Charts
            html.Div([
                # Crash rate over time
                dcc.Graph(id='crash-rate-chart'),
                
                # Anomaly detection
                dcc.Graph(id='anomaly-chart'),
                
                # Performance metrics
                dcc.Graph(id='performance-chart'),
                
                # Diversity metrics
                dcc.Graph(id='diversity-chart')
            ], className='charts-container'),
            
            # Update interval
            dcc.Interval(
                id='interval-component',
                interval=1000,  # Update every second
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            [Output('total-crashes', 'children'),
             Output('anomaly-count', 'children'),
             Output('crash-rate', 'children'),
             Output('system-impacts', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_summary_cards(n):
            # Get data from analytics engine
            summary = analytics_engine.get_analytics_summary()
            
            total_crashes = summary.get("recent_metrics", {}).get("total_crashes", {}).get("mean", 0)
            anomaly_count = summary.get("total_anomalies", 0)
            crash_rate = summary.get("recent_metrics", {}).get("crash_rate", {}).get("mean", 0)
            system_impacts = summary.get("recent_metrics", {}).get("system_impacts", {}).get("mean", 0)
            
            return [
                f"{total_crashes:.0f}",
                f"{anomaly_count}",
                f"{crash_rate:.2%}",
                f"{system_impacts:.0f}"
            ]
        
        @self.app.callback(
            Output('crash-rate-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_crash_rate_chart(n):
            # Get recent data
            recent_data = analytics_engine.stream_processor.get_recent_data(100)
            
            if not recent_data:
                return go.Figure()
            
            timestamps = [d["timestamp"] for d in recent_data]
            crash_rates = [d["data"].get("crash_rate", 0) for d in recent_data]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=crash_rates,
                mode='lines+markers',
                name='Crash Rate',
                line=dict(color='#e74c3c')
            ))
            
            fig.update_layout(
                title="Crash Rate Over Time",
                xaxis_title="Time",
                yaxis_title="Crash Rate",
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output('anomaly-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_anomaly_chart(n):
            # Get anomaly data
            anomalies = list(analytics_engine.anomalies)
            
            if not anomalies:
                return go.Figure()
            
            timestamps = [a["timestamp"] for a in anomalies]
            scores = [a["score"] for a in anomalies]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=scores,
                mode='markers',
                name='Anomaly Score',
                marker=dict(
                    color='#f39c12',
                    size=10,
                    symbol='diamond'
                )
            ))
            
            fig.update_layout(
                title="Anomaly Detection",
                xaxis_title="Time",
                yaxis_title="Anomaly Score",
                height=400
            )
            
            return fig
    
    def run(self, debug: bool = False):
        """Run the dashboard"""
        self.app.run_server(debug=debug, port=self.port)

# Global instances
analytics_engine = None
websocket_server = None
dashboard_app = None

def get_analytics_engine() -> RealTimeAnalytics:
    """Get or create global analytics engine instance"""
    global analytics_engine
    if analytics_engine is None:
        config = AnalyticsConfig()
        analytics_engine = RealTimeAnalytics(config)
    return analytics_engine

async def start_real_time_analytics():
    """Start the complete real-time analytics system"""
    global websocket_server, dashboard_app
    
    # Initialize components
    analytics_engine = get_analytics_engine()
    websocket_server = WebSocketServer()
    dashboard_app = DashboardApp()
    
    try:
        # Start analytics engine
        await analytics_engine.start()
        
        # Start WebSocket server
        await websocket_server.start()
        
        # Start dashboard in separate thread
        dashboard_thread = threading.Thread(
            target=dashboard_app.run,
            kwargs={'debug': False}
        )
        dashboard_thread.daemon = True
        dashboard_thread.start()
        
        logging.info("Real-time analytics system started")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Shutting down analytics system...")
    finally:
        await analytics_engine.stop()
        await websocket_server.stop()

if __name__ == "__main__":
    # Test the real-time analytics system
    asyncio.run(start_real_time_analytics()) 