#!/usr/bin/env python3
"""
Advanced Distributed Orchestrator for KernelHunter
Implements Kubernetes orchestration, Service Mesh, and intelligent load balancing
"""

import asyncio
import aiohttp
import kubernetes
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import docker
import yaml
import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import subprocess
import os
from pathlib import Path
import hashlib
import random
from collections import defaultdict

@dataclass
class NodeConfig:
    """Configuration for a Kubernetes node"""
    name: str
    image: str = "kernelhunter:latest"
    replicas: int = 1
    cpu_limit: str = "1000m"
    memory_limit: str = "2Gi"
    cpu_request: str = "500m"
    memory_request: str = "1Gi"
    node_selector: Dict[str, str] = None
    env_vars: Dict[str, str] = None

@dataclass
class ServiceMeshConfig:
    """Configuration for service mesh"""
    enable_istio: bool = True
    enable_traffic_splitting: bool = True
    circuit_breaker_enabled: bool = True
    retry_policy: Dict[str, Any] = None

class KubernetesOrchestrator:
    """Advanced Kubernetes orchestrator for KernelHunter"""
    
    def __init__(self, namespace: str = "kernelhunter"):
        self.namespace = namespace
        self.api_client = None
        self.apps_v1 = None
        self.core_v1 = None
        self.networking_v1 = None
        
        # Initialize Kubernetes client
        self._init_kubernetes()
        
        # Node registry
        self.nodes = {}
        self.node_health = {}
        self.load_balancer = LoadBalancer()
        
    def _init_kubernetes(self):
        """Initialize Kubernetes client"""
        try:
            # Try to load in-cluster config first
            config.load_incluster_config()
        except config.ConfigException:
            try:
                # Fall back to kubeconfig
                config.load_kube_config()
            except config.ConfigException:
                raise Exception("Could not load Kubernetes configuration")
        
        self.api_client = client.ApiClient()
        self.apps_v1 = client.AppsV1Api(self.api_client)
        self.core_v1 = client.CoreV1Api(self.api_client)
        self.networking_v1 = client.NetworkingV1Api(self.api_client)
        
        # Create namespace if it doesn't exist
        self._create_namespace()
    
    def _create_namespace(self):
        """Create namespace if it doesn't exist"""
        try:
            self.core_v1.read_namespace(self.namespace)
        except ApiException as e:
            if e.status == 404:
                namespace = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=self.namespace)
                )
                self.core_v1.create_namespace(namespace)
                logging.info(f"Created namespace: {self.namespace}")
    
    def create_kernelhunter_deployment(self, node_config: NodeConfig) -> str:
        """Create Kubernetes deployment for KernelHunter node"""
        
        # Create container
        container = client.V1Container(
            name="kernelhunter",
            image=node_config.image,
            ports=[client.V1ContainerPort(container_port=8000)],
            resources=client.V1ResourceRequirements(
                limits={
                    "cpu": node_config.cpu_limit,
                    "memory": node_config.memory_limit
                },
                requests={
                    "cpu": node_config.cpu_request,
                    "memory": node_config.memory_request
                }
            ),
            env=[
                client.V1EnvVar(name=k, value=v)
                for k, v in (node_config.env_vars or {}).items()
            ]
        )
        
        # Create pod template
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels={"app": f"kernelhunter-{node_config.name}"}
            ),
            spec=client.V1PodSpec(
                containers=[container],
                node_selector=node_config.node_selector or {}
            )
        )
        
        # Create deployment
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(name=f"kernelhunter-{node_config.name}"),
            spec=client.V1DeploymentSpec(
                replicas=node_config.replicas,
                selector=client.V1LabelSelector(
                    match_labels={"app": f"kernelhunter-{node_config.name}"}
                ),
                template=template
            )
        )
        
        try:
            result = self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )
            logging.info(f"Created deployment: {result.metadata.name}")
            return result.metadata.name
        except ApiException as e:
            logging.error(f"Error creating deployment: {e}")
            raise
    
    def create_service(self, node_name: str, port: int = 8000) -> str:
        """Create Kubernetes service for node"""
        
        service = client.V1Service(
            metadata=client.V1ObjectMeta(name=f"kernelhunter-{node_name}-svc"),
            spec=client.V1ServiceSpec(
                selector={"app": f"kernelhunter-{node_name}"},
                ports=[client.V1ServicePort(port=port, target_port=8000)],
                type="ClusterIP"
            )
        )
        
        try:
            result = self.core_v1.create_namespaced_service(
                namespace=self.namespace,
                body=service
            )
            logging.info(f"Created service: {result.metadata.name}")
            return result.metadata.name
        except ApiException as e:
            logging.error(f"Error creating service: {e}")
            raise
    
    def scale_deployment(self, deployment_name: str, replicas: int):
        """Scale deployment to specified number of replicas"""
        try:
            self.apps_v1.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=self.namespace,
                body={"spec": {"replicas": replicas}}
            )
            logging.info(f"Scaled deployment {deployment_name} to {replicas} replicas")
        except ApiException as e:
            logging.error(f"Error scaling deployment: {e}")
            raise
    
    def get_deployment_status(self, deployment_name: str) -> Dict:
        """Get deployment status"""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            return {
                "name": deployment.metadata.name,
                "replicas": deployment.spec.replicas,
                "available_replicas": deployment.status.available_replicas,
                "ready_replicas": deployment.status.ready_replicas,
                "updated_replicas": deployment.status.updated_replicas
            }
        except ApiException as e:
            logging.error(f"Error getting deployment status: {e}")
            return {}
    
    def delete_deployment(self, deployment_name: str):
        """Delete deployment"""
        try:
            self.apps_v1.delete_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            logging.info(f"Deleted deployment: {deployment_name}")
        except ApiException as e:
            logging.error(f"Error deleting deployment: {e}")
            raise

class ServiceMesh:
    """Service mesh implementation with Istio"""
    
    def __init__(self, namespace: str = "kernelhunter"):
        self.namespace = namespace
        self.istio_enabled = False
        
    def enable_istio(self):
        """Enable Istio service mesh"""
        try:
            # Check if Istio is installed
            result = subprocess.run(
                ["istioctl", "version"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.istio_enabled = True
                logging.info("Istio service mesh enabled")
            else:
                logging.warning("Istio not found, service mesh features disabled")
                
        except FileNotFoundError:
            logging.warning("Istio CLI not found, service mesh features disabled")
    
    def create_virtual_service(self, service_name: str, routes: List[Dict]) -> str:
        """Create Istio VirtualService"""
        if not self.istio_enabled:
            return None
        
        virtual_service = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {
                "name": f"{service_name}-virtual",
                "namespace": self.namespace
            },
            "spec": {
                "hosts": [f"{service_name}.{self.namespace}.svc.cluster.local"],
                "http": routes
            }
        }
        
        # Apply using kubectl
        yaml_content = yaml.dump(virtual_service)
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=yaml_content,
            text=True,
            capture_output=True
        )
        
        if result.returncode == 0:
            logging.info(f"Created VirtualService: {service_name}-virtual")
            return f"{service_name}-virtual"
        else:
            logging.error(f"Error creating VirtualService: {result.stderr}")
            return None
    
    def create_destination_rule(self, service_name: str, subsets: List[Dict]) -> str:
        """Create Istio DestinationRule"""
        if not self.istio_enabled:
            return None
        
        destination_rule = {
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "DestinationRule",
            "metadata": {
                "name": f"{service_name}-destination",
                "namespace": self.namespace
            },
            "spec": {
                "host": f"{service_name}.{self.namespace}.svc.cluster.local",
                "subsets": subsets
            }
        }
        
        # Apply using kubectl
        yaml_content = yaml.dump(destination_rule)
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=yaml_content,
            text=True,
            capture_output=True
        )
        
        if result.returncode == 0:
            logging.info(f"Created DestinationRule: {service_name}-destination")
            return f"{service_name}-destination"
        else:
            logging.error(f"Error creating DestinationRule: {result.stderr}")
            return None

class LoadBalancer:
    """Intelligent load balancer with health checking and failover"""
    
    def __init__(self):
        self.nodes = {}
        self.health_checks = {}
        self.load_distribution = defaultdict(int)
        self.circuit_breakers = {}
        
    def add_node(self, node_id: str, endpoint: str, weight: int = 1):
        """Add node to load balancer"""
        self.nodes[node_id] = {
            "endpoint": endpoint,
            "weight": weight,
            "healthy": True,
            "last_health_check": time.time(),
            "response_time": 0
        }
        
        logging.info(f"Added node {node_id} to load balancer")
    
    def remove_node(self, node_id: str):
        """Remove node from load balancer"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            del self.load_distribution[node_id]
            logging.info(f"Removed node {node_id} from load balancer")
    
    async def health_check_node(self, node_id: str):
        """Perform health check on node"""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{node['endpoint']}/health", timeout=5) as response:
                    if response.status == 200:
                        node["healthy"] = True
                        node["response_time"] = time.time() - start_time
                        node["last_health_check"] = time.time()
                        return True
                    else:
                        node["healthy"] = False
                        return False
        except Exception as e:
            node["healthy"] = False
            logging.error(f"Health check failed for node {node_id}: {e}")
            return False
    
    async def health_check_all_nodes(self):
        """Perform health check on all nodes"""
        tasks = [
            self.health_check_node(node_id)
            for node_id in self.nodes.keys()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def select_node(self, strategy: str = "weighted_round_robin") -> Optional[str]:
        """Select node using specified strategy"""
        healthy_nodes = {
            node_id: node
            for node_id, node in self.nodes.items()
            if node["healthy"]
        }
        
        if not healthy_nodes:
            return None
        
        if strategy == "weighted_round_robin":
            return self._weighted_round_robin(healthy_nodes)
        elif strategy == "least_connections":
            return self._least_connections(healthy_nodes)
        elif strategy == "fastest_response":
            return self._fastest_response(healthy_nodes)
        else:
            return random.choice(list(healthy_nodes.keys()))
    
    def _weighted_round_robin(self, healthy_nodes: Dict) -> str:
        """Weighted round-robin selection"""
        total_weight = sum(node["weight"] for node in healthy_nodes.values())
        if total_weight == 0:
            return random.choice(list(healthy_nodes.keys()))
        
        # Calculate weighted distribution
        weights = []
        node_ids = []
        for node_id, node in healthy_nodes.items():
            weights.append(node["weight"])
            node_ids.append(node_id)
        
        # Select based on weights
        selected = random.choices(node_ids, weights=weights, k=1)[0]
        self.load_distribution[selected] += 1
        return selected
    
    def _least_connections(self, healthy_nodes: Dict) -> str:
        """Least connections selection"""
        return min(
            healthy_nodes.keys(),
            key=lambda node_id: self.load_distribution.get(node_id, 0)
        )
    
    def _fastest_response(self, healthy_nodes: Dict) -> str:
        """Fastest response time selection"""
        return min(
            healthy_nodes.keys(),
            key=lambda node_id: healthy_nodes[node_id]["response_time"]
        )
    
    def get_load_stats(self) -> Dict:
        """Get load balancing statistics"""
        return {
            "total_nodes": len(self.nodes),
            "healthy_nodes": sum(1 for node in self.nodes.values() if node["healthy"]),
            "load_distribution": dict(self.load_distribution),
            "node_health": {
                node_id: {
                    "healthy": node["healthy"],
                    "response_time": node["response_time"],
                    "last_check": node["last_health_check"]
                }
                for node_id, node in self.nodes.items()
            }
        }

class DistributedOrchestrator:
    """Main distributed orchestrator class"""
    
    def __init__(self, namespace: str = "kernelhunter"):
        self.namespace = namespace
        self.kubernetes = KubernetesOrchestrator(namespace)
        self.service_mesh = ServiceMesh(namespace)
        self.load_balancer = LoadBalancer()
        
        # Node registry
        self.nodes = {}
        self.deployments = {}
        self.services = {}
        
        # Background tasks
        self.background_tasks = []
        self.running = False
        
    async def start(self):
        """Start the distributed orchestrator"""
        self.running = True
        
        # Enable service mesh
        self.service_mesh.enable_istio()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._load_balancing_loop())
        ]
        
        logging.info("Distributed orchestrator started")
    
    async def stop(self):
        """Stop the distributed orchestrator"""
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for completion
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logging.info("Distributed orchestrator stopped")
    
    async def deploy_node(self, node_config: NodeConfig) -> str:
        """Deploy a new KernelHunter node"""
        
        # Create deployment
        deployment_name = self.kubernetes.create_kernelhunter_deployment(node_config)
        self.deployments[node_config.name] = deployment_name
        
        # Create service
        service_name = self.kubernetes.create_service(node_config.name)
        self.services[node_config.name] = service_name
        
        # Add to load balancer
        service_endpoint = f"http://{service_name}.{self.namespace}.svc.cluster.local:8000"
        self.load_balancer.add_node(node_config.name, service_endpoint, node_config.replicas)
        
        # Store node config
        self.nodes[node_config.name] = node_config
        
        logging.info(f"Deployed node: {node_config.name}")
        return deployment_name
    
    async def scale_node(self, node_name: str, replicas: int):
        """Scale a node to specified number of replicas"""
        if node_name not in self.deployments:
            raise ValueError(f"Node {node_name} not found")
        
        deployment_name = self.deployments[node_name]
        self.kubernetes.scale_deployment(deployment_name, replicas)
        
        # Update load balancer
        if node_name in self.load_balancer.nodes:
            self.load_balancer.nodes[node_name]["weight"] = replicas
        
        logging.info(f"Scaled node {node_name} to {replicas} replicas")
    
    async def remove_node(self, node_name: str):
        """Remove a node"""
        if node_name not in self.deployments:
            return
        
        # Remove from load balancer
        self.load_balancer.remove_node(node_name)
        
        # Delete deployment
        deployment_name = self.deployments[node_name]
        self.kubernetes.delete_deployment(deployment_name)
        
        # Clean up
        del self.deployments[node_name]
        del self.services[node_name]
        del self.nodes[node_name]
        
        logging.info(f"Removed node: {node_name}")
    
    async def get_node_status(self, node_name: str) -> Dict:
        """Get status of a specific node"""
        if node_name not in self.deployments:
            return {}
        
        deployment_name = self.deployments[node_name]
        status = self.kubernetes.get_deployment_status(deployment_name)
        
        # Add load balancer info
        if node_name in self.load_balancer.nodes:
            status["load_balancer"] = self.load_balancer.nodes[node_name]
        
        return status
    
    async def get_cluster_status(self) -> Dict:
        """Get status of entire cluster"""
        cluster_status = {
            "nodes": {},
            "load_balancer": self.load_balancer.get_load_stats(),
            "total_deployments": len(self.deployments),
            "total_services": len(self.services)
        }
        
        # Get status for each node
        for node_name in self.nodes.keys():
            cluster_status["nodes"][node_name] = await self.get_node_status(node_name)
        
        return cluster_status
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while self.running:
            try:
                await self.load_balancer.health_check_all_nodes()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logging.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)
    
    async def _load_balancing_loop(self):
        """Background load balancing loop"""
        while self.running:
            try:
                # Update load distribution based on health
                stats = self.load_balancer.get_load_stats()
                logging.debug(f"Load balancer stats: {stats}")
                await asyncio.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logging.error(f"Error in load balancing loop: {e}")
                await asyncio.sleep(30)

# Global orchestrator instance
distributed_orchestrator = None

def get_distributed_orchestrator() -> DistributedOrchestrator:
    """Get or create global distributed orchestrator instance"""
    global distributed_orchestrator
    if distributed_orchestrator is None:
        distributed_orchestrator = DistributedOrchestrator()
    return distributed_orchestrator

async def deploy_kernelhunter_cluster():
    """Deploy a complete KernelHunter cluster"""
    orchestrator = get_distributed_orchestrator()
    
    try:
        await orchestrator.start()
        
        # Deploy multiple nodes
        node_configs = [
            NodeConfig(
                name="node-1",
                replicas=3,
                env_vars={"NODE_ID": "node-1", "MAX_GENERATIONS": "1000"}
            ),
            NodeConfig(
                name="node-2",
                replicas=2,
                env_vars={"NODE_ID": "node-2", "MAX_GENERATIONS": "1000"}
            ),
            NodeConfig(
                name="node-3",
                replicas=2,
                env_vars={"NODE_ID": "node-3", "MAX_GENERATIONS": "1000"}
            )
        ]
        
        for config in node_configs:
            await orchestrator.deploy_node(config)
        
        # Wait for deployments to be ready
        await asyncio.sleep(30)
        
        # Get cluster status
        status = await orchestrator.get_cluster_status()
        print(f"Cluster status: {json.dumps(status, indent=2)}")
        
    finally:
        await orchestrator.stop()

if __name__ == "__main__":
    # Test the distributed orchestrator
    asyncio.run(deploy_kernelhunter_cluster()) 