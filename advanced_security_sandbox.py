#!/usr/bin/env python3
"""
Advanced Security Sandbox for KernelHunter
Implements container isolation, virtual machines, and hardware virtualization
"""

import docker
import subprocess
import asyncio
import aiohttp
import json
import time
import logging
import tempfile
import os
import shutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import psutil
import signal
import threading
from contextlib import asynccontextmanager
import hashlib
import numpy as np

@dataclass
class SandboxConfig:
    """Configuration for security sandbox"""
    isolation_level: str = "container"  # container, vm, hardware
    max_execution_time: int = 30
    max_memory_mb: int = 512
    max_cpu_percent: int = 50
    enable_network_isolation: bool = True
    enable_filesystem_isolation: bool = True
    enable_process_isolation: bool = True
    docker_image: str = "ubuntu:20.04"
    vm_image_path: str = "/var/lib/libvirt/images/kernelhunter.qcow2"
    hardware_vm_path: str = "/var/lib/vmware/kernelhunter.vmx"

class ContainerSandbox:
    """Docker-based container sandbox"""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.docker_client = docker.from_env()
        self.active_containers = {}
        
    async def create_sandbox(self, shellcode: bytes, sandbox_id: str) -> str:
        """Create container sandbox for shellcode execution"""
        
        # Create temporary directory for shellcode
        temp_dir = tempfile.mkdtemp(prefix=f"kernelhunter_{sandbox_id}_")
        
        try:
            # Write shellcode to file
            shellcode_file = os.path.join(temp_dir, "shellcode.bin")
            with open(shellcode_file, "wb") as f:
                f.write(shellcode)
            
            # Create C wrapper
            c_wrapper = self._create_c_wrapper(shellcode_file)
            wrapper_file = os.path.join(temp_dir, "wrapper.c")
            with open(wrapper_file, "w") as f:
                f.write(c_wrapper)
            
            # Create Dockerfile
            dockerfile = self._create_dockerfile()
            dockerfile_path = os.path.join(temp_dir, "Dockerfile")
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile)
            
            # Build container
            container_name = f"kernelhunter_sandbox_{sandbox_id}"
            
            # Build image
            image, logs = self.docker_client.images.build(
                path=temp_dir,
                tag=f"kernelhunter:{sandbox_id}",
                rm=True
            )
            
            # Create container with resource limits
            container = self.docker_client.containers.run(
                image.id,
                name=container_name,
                detach=True,
                mem_limit=f"{self.config.max_memory_mb}m",
                cpu_period=100000,
                cpu_quota=int(100000 * self.config.max_cpu_percent / 100),
                network_mode="none" if self.config.enable_network_isolation else "bridge",
                volumes={
                    temp_dir: {"bind": "/workspace", "mode": "ro"}
                },
                security_opt=["no-new-privileges"],
                cap_drop=["ALL"],
                read_only=True
            )
            
            self.active_containers[sandbox_id] = {
                "container": container,
                "temp_dir": temp_dir,
                "start_time": time.time()
            }
            
            logging.info(f"Created container sandbox: {sandbox_id}")
            return container_name
            
        except Exception as e:
            # Cleanup on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e
    
    def _create_c_wrapper(self, shellcode_path: str) -> str:
        """Create C wrapper for shellcode execution"""
        return f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int main() {{
    FILE *f = fopen("{shellcode_path}", "rb");
    if (!f) {{
        fprintf(stderr, "Failed to open shellcode file\\n");
        return 1;
    }}
    
    // Get file size
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    // Allocate executable memory
    void *exec_mem = mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (exec_mem == MAP_FAILED) {{
        fprintf(stderr, "Failed to allocate executable memory\\n");
        fclose(f);
        return 1;
    }}
    
    // Read shellcode
    fread(exec_mem, 1, size, f);
    fclose(f);
    
    // Execute shellcode
    ((void (*)())exec_mem)();
    
    return 0;
}}
"""
    
    def _create_dockerfile(self) -> str:
        """Create Dockerfile for sandbox"""
        return f"""
FROM {self.config.docker_image}

# Install required packages
RUN apt-get update && apt-get install -y \\
    gcc \\
    make \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Compile wrapper
RUN gcc -o wrapper wrapper.c -static

# Run wrapper
CMD ["./wrapper"]
"""
    
    async def execute_sandbox(self, sandbox_id: str) -> Dict:
        """Execute shellcode in sandbox"""
        if sandbox_id not in self.active_containers:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        sandbox_info = self.active_containers[sandbox_id]
        container = sandbox_info["container"]
        
        try:
            # Start container
            container.start()
            
            # Wait for completion with timeout
            try:
                result = container.wait(timeout=self.config.max_execution_time)
                logs = container.logs().decode('utf-8', errors='ignore')
                
                return {
                    "sandbox_id": sandbox_id,
                    "exit_code": result["StatusCode"],
                    "logs": logs,
                    "execution_time": time.time() - sandbox_info["start_time"],
                    "status": "completed"
                }
                
            except Exception as e:
                # Container timed out or failed
                container.kill()
                return {
                    "sandbox_id": sandbox_id,
                    "exit_code": -1,
                    "logs": str(e),
                    "execution_time": time.time() - sandbox_info["start_time"],
                    "status": "timeout"
                }
                
        except Exception as e:
            return {
                "sandbox_id": sandbox_id,
                "exit_code": -1,
                "logs": str(e),
                "execution_time": 0,
                "status": "error"
            }
    
    def cleanup_sandbox(self, sandbox_id: str):
        """Clean up sandbox resources"""
        if sandbox_id not in self.active_containers:
            return
        
        sandbox_info = self.active_containers[sandbox_id]
        
        try:
            # Stop and remove container
            container = sandbox_info["container"]
            container.stop(timeout=5)
            container.remove()
            
            # Remove temporary directory
            shutil.rmtree(sandbox_info["temp_dir"], ignore_errors=True)
            
            # Remove image
            try:
                self.docker_client.images.remove(f"kernelhunter:{sandbox_id}", force=True)
            except:
                pass
            
            del self.active_containers[sandbox_id]
            
            logging.info(f"Cleaned up sandbox: {sandbox_id}")
            
        except Exception as e:
            logging.error(f"Error cleaning up sandbox {sandbox_id}: {e}")

class VirtualMachineSandbox:
    """QEMU-based virtual machine sandbox"""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.active_vms = {}
        
    async def create_sandbox(self, shellcode: bytes, sandbox_id: str) -> str:
        """Create VM sandbox for shellcode execution"""
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"kernelhunter_vm_{sandbox_id}_")
        
        try:
            # Create VM disk image
            disk_image = os.path.join(temp_dir, "disk.qcow2")
            subprocess.run([
                "qemu-img", "create", "-f", "qcow2", disk_image, "10G"
            ], check=True)
            
            # Create shellcode file
            shellcode_file = os.path.join(temp_dir, "shellcode.bin")
            with open(shellcode_file, "wb") as f:
                f.write(shellcode)
            
            # Create VM configuration
            vm_config = self._create_vm_config(disk_image, shellcode_file)
            config_file = os.path.join(temp_dir, "vm.xml")
            with open(config_file, "w") as f:
                f.write(vm_config)
            
            # Start VM
            vm_name = f"kernelhunter_vm_{sandbox_id}"
            
            # Define and start VM using libvirt
            subprocess.run([
                "virsh", "define", config_file
            ], check=True)
            
            subprocess.run([
                "virsh", "start", vm_name
            ], check=True)
            
            self.active_vms[sandbox_id] = {
                "vm_name": vm_name,
                "temp_dir": temp_dir,
                "start_time": time.time()
            }
            
            logging.info(f"Created VM sandbox: {sandbox_id}")
            return vm_name
            
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e
    
    def _create_vm_config(self, disk_image: str, shellcode_file: str) -> str:
        """Create VM configuration XML"""
        return f"""
<domain type='kvm'>
  <name>kernelhunter_vm</name>
  <memory unit='MB'>{self.config.max_memory_mb}</memory>
  <vcpu>{max(1, self.config.max_cpu_percent // 25)}</vcpu>
  <os>
    <type arch='x86_64'>hvm</type>
    <boot dev='hd'/>
  </os>
  <devices>
    <disk type='file' device='disk'>
      <source file='{disk_image}'/>
      <target dev='hda' bus='ide'/>
    </disk>
    <disk type='file' device='cdrom'>
      <source file='{shellcode_file}'/>
      <target dev='hdc' bus='ide'/>
    </disk>
    <interface type='network'>
      <source network='default'/>
    </interface>
    <console type='pty'/>
  </devices>
</domain>
"""
    
    async def execute_sandbox(self, sandbox_id: str) -> Dict:
        """Execute shellcode in VM sandbox"""
        if sandbox_id not in self.active_vms:
            raise ValueError(f"VM sandbox {sandbox_id} not found")
        
        vm_info = self.active_vms[sandbox_id]
        vm_name = vm_info["vm_name"]
        
        try:
            # Wait for VM to start
            await asyncio.sleep(10)
            
            # Check VM status
            result = subprocess.run([
                "virsh", "domstate", vm_name
            ], capture_output=True, text=True)
            
            if "running" not in result.stdout:
                return {
                    "sandbox_id": sandbox_id,
                    "exit_code": -1,
                    "logs": "VM failed to start",
                    "execution_time": time.time() - vm_info["start_time"],
                    "status": "error"
                }
            
            # Wait for completion or timeout
            start_time = time.time()
            while time.time() - start_time < self.config.max_execution_time:
                result = subprocess.run([
                    "virsh", "domstate", vm_name
                ], capture_output=True, text=True)
                
                if "shut off" in result.stdout:
                    break
                
                await asyncio.sleep(1)
            
            # Get VM logs
            logs_result = subprocess.run([
                "virsh", "domdisplay", vm_name
            ], capture_output=True, text=True)
            
            return {
                "sandbox_id": sandbox_id,
                "exit_code": 0,
                "logs": logs_result.stdout,
                "execution_time": time.time() - vm_info["start_time"],
                "status": "completed"
            }
            
        except Exception as e:
            return {
                "sandbox_id": sandbox_id,
                "exit_code": -1,
                "logs": str(e),
                "execution_time": time.time() - vm_info["start_time"],
                "status": "error"
            }
    
    def cleanup_sandbox(self, sandbox_id: str):
        """Clean up VM sandbox resources"""
        if sandbox_id not in self.active_vms:
            return
        
        vm_info = self.active_vms[sandbox_id]
        
        try:
            # Stop and undefine VM
            vm_name = vm_info["vm_name"]
            subprocess.run(["virsh", "destroy", vm_name], check=False)
            subprocess.run(["virsh", "undefine", vm_name], check=False)
            
            # Remove temporary directory
            shutil.rmtree(vm_info["temp_dir"], ignore_errors=True)
            
            del self.active_vms[sandbox_id]
            
            logging.info(f"Cleaned up VM sandbox: {sandbox_id}")
            
        except Exception as e:
            logging.error(f"Error cleaning up VM sandbox {sandbox_id}: {e}")

class HardwareVirtualizationSandbox:
    """Hardware-level virtualization sandbox using VMware/VirtualBox"""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.active_vms = {}
        
    async def create_sandbox(self, shellcode: bytes, sandbox_id: str) -> str:
        """Create hardware virtualization sandbox"""
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"kernelhunter_hw_{sandbox_id}_")
        
        try:
            # Create shellcode file
            shellcode_file = os.path.join(temp_dir, "shellcode.bin")
            with open(shellcode_file, "wb") as f:
                f.write(shellcode)
            
            # Create VM configuration
            vm_config = self._create_vm_config(shellcode_file)
            config_file = os.path.join(temp_dir, "vm.vmx")
            with open(config_file, "w") as f:
                f.write(vm_config)
            
            # Start VM using VMware Player
            vm_name = f"kernelhunter_hw_{sandbox_id}"
            
            # Start VM
            process = subprocess.Popen([
                "vmplayer", config_file
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.active_vms[sandbox_id] = {
                "process": process,
                "temp_dir": temp_dir,
                "start_time": time.time()
            }
            
            logging.info(f"Created hardware virtualization sandbox: {sandbox_id}")
            return vm_name
            
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise e
    
    def _create_vm_config(self, shellcode_file: str) -> str:
        """Create VMware VM configuration"""
        return f"""
.encoding = "windows-1252"
config.version = "8"
virtualHW.version = "19"
numvcpus = "1"
memsize = "{self.config.max_memory_mb}"
displayName = "KernelHunter Sandbox"
guestOS = "ubuntu-64"
powerType.powerOff = "soft"
powerType.powerOn = "soft"
powerType.suspend = "soft"
powerType.reset = "soft"
scsi0.present = "TRUE"
scsi0.virtualDev = "lsisas1068"
scsi0:0.present = "TRUE"
scsi0:0.fileName = "disk.vmdk"
scsi0:0.size = "10737418240"
scsi0:0.writeThrough = "FALSE"
scsi0:1.present = "TRUE"
scsi0:1.fileName = "{shellcode_file}"
scsi0:1.deviceType = "cdrom-raw"
ethernet0.present = "TRUE"
ethernet0.virtualDev = "e1000"
ethernet0.networkName = "VM Network"
ethernet0.addressType = "generated"
"""
    
    async def execute_sandbox(self, sandbox_id: str) -> Dict:
        """Execute shellcode in hardware virtualization sandbox"""
        if sandbox_id not in self.active_vms:
            raise ValueError(f"Hardware sandbox {sandbox_id} not found")
        
        vm_info = self.active_vms[sandbox_id]
        process = vm_info["process"]
        
        try:
            # Wait for completion or timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, process.communicate
                    ),
                    timeout=self.config.max_execution_time
                )
                
                return {
                    "sandbox_id": sandbox_id,
                    "exit_code": process.returncode,
                    "logs": stdout.decode('utf-8', errors='ignore'),
                    "execution_time": time.time() - vm_info["start_time"],
                    "status": "completed"
                }
                
            except asyncio.TimeoutError:
                # Kill process on timeout
                process.kill()
                return {
                    "sandbox_id": sandbox_id,
                    "exit_code": -1,
                    "logs": "Execution timeout",
                    "execution_time": time.time() - vm_info["start_time"],
                    "status": "timeout"
                }
                
        except Exception as e:
            return {
                "sandbox_id": sandbox_id,
                "exit_code": -1,
                "logs": str(e),
                "execution_time": time.time() - vm_info["start_time"],
                "status": "error"
            }
    
    def cleanup_sandbox(self, sandbox_id: str):
        """Clean up hardware virtualization sandbox"""
        if sandbox_id not in self.active_vms:
            return
        
        vm_info = self.active_vms[sandbox_id]
        
        try:
            # Kill process
            process = vm_info["process"]
            process.kill()
            
            # Remove temporary directory
            shutil.rmtree(vm_info["temp_dir"], ignore_errors=True)
            
            del self.active_vms[sandbox_id]
            
            logging.info(f"Cleaned up hardware sandbox: {sandbox_id}")
            
        except Exception as e:
            logging.error(f"Error cleaning up hardware sandbox {sandbox_id}: {e}")

class AdvancedSecuritySandbox:
    """Main security sandbox class with multiple isolation levels"""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        
        # Initialize sandbox based on isolation level
        if config.isolation_level == "container":
            self.sandbox = ContainerSandbox(config)
        elif config.isolation_level == "vm":
            self.sandbox = VirtualMachineSandbox(config)
        elif config.isolation_level == "hardware":
            self.sandbox = HardwareVirtualizationSandbox(config)
        else:
            raise ValueError(f"Unsupported isolation level: {config.isolation_level}")
        
        # Sandbox registry
        self.active_sandboxes = {}
        self.execution_history = []
        
    async def execute_shellcode(self, shellcode: bytes, sandbox_id: str = None) -> Dict:
        """Execute shellcode in secure sandbox"""
        
        if sandbox_id is None:
            sandbox_id = hashlib.md5(shellcode).hexdigest()[:8]
        
        try:
            # Create sandbox
            sandbox_name = await self.sandbox.create_sandbox(shellcode, sandbox_id)
            
            # Record sandbox creation
            self.active_sandboxes[sandbox_id] = {
                "name": sandbox_name,
                "created_at": time.time(),
                "shellcode_hash": hashlib.md5(shellcode).hexdigest()
            }
            
            # Execute in sandbox
            result = await self.sandbox.execute_sandbox(sandbox_id)
            
            # Add metadata
            result["sandbox_type"] = self.config.isolation_level
            result["shellcode_length"] = len(shellcode)
            result["shellcode_hash"] = hashlib.md5(shellcode).hexdigest()
            
            # Record execution
            self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            return {
                "sandbox_id": sandbox_id,
                "exit_code": -1,
                "logs": str(e),
                "execution_time": 0,
                "status": "error",
                "sandbox_type": self.config.isolation_level
            }
        
        finally:
            # Cleanup sandbox
            self.sandbox.cleanup_sandbox(sandbox_id)
            if sandbox_id in self.active_sandboxes:
                del self.active_sandboxes[sandbox_id]
    
    async def execute_batch(self, shellcodes: List[bytes], max_concurrent: int = 5) -> List[Dict]:
        """Execute multiple shellcodes in parallel"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(shellcode: bytes, index: int):
            async with semaphore:
                sandbox_id = f"batch_{index}_{hashlib.md5(shellcode).hexdigest()[:8]}"
                return await self.execute_shellcode(shellcode, sandbox_id)
        
        tasks = [
            execute_with_semaphore(shellcode, i)
            for i, shellcode in enumerate(shellcodes)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [
            result for result in results
            if isinstance(result, dict)
        ]
        
        return valid_results
    
    def get_sandbox_stats(self) -> Dict:
        """Get sandbox statistics"""
        return {
            "isolation_level": self.config.isolation_level,
            "active_sandboxes": len(self.active_sandboxes),
            "total_executions": len(self.execution_history),
            "successful_executions": sum(
                1 for r in self.execution_history
                if r.get("status") == "completed"
            ),
            "failed_executions": sum(
                1 for r in self.execution_history
                if r.get("status") in ["error", "timeout"]
            ),
            "avg_execution_time": np.mean([
                r.get("execution_time", 0)
                for r in self.execution_history
            ]) if self.execution_history else 0
        }
    
    def cleanup_all(self):
        """Clean up all active sandboxes"""
        for sandbox_id in list(self.active_sandboxes.keys()):
            self.sandbox.cleanup_sandbox(sandbox_id)
        
        self.active_sandboxes.clear()

# Global sandbox instance
security_sandbox = None

def get_security_sandbox() -> AdvancedSecuritySandbox:
    """Get or create global security sandbox instance"""
    global security_sandbox
    if security_sandbox is None:
        config = SandboxConfig()
        security_sandbox = AdvancedSecuritySandbox(config)
    return security_sandbox

async def test_security_sandbox():
    """Test the security sandbox with sample shellcodes"""
    sandbox = get_security_sandbox()
    
    # Test shellcodes
    test_shellcodes = [
        b"\x90\x90\x48\x31\xc0\x0f\x05",  # Simple syscall
        b"\x48\xc7\xc0\x3c\x00\x00\x00\x0f\x05",  # Exit syscall
        b"\x90" * 100,  # NOP sled
    ]
    
    print("Testing security sandbox...")
    
    for i, shellcode in enumerate(test_shellcodes):
        print(f"Executing shellcode {i+1}...")
        result = await sandbox.execute_shellcode(shellcode, f"test_{i}")
        print(f"Result: {result}")
    
    # Test batch execution
    print("Testing batch execution...")
    batch_results = await sandbox.execute_batch(test_shellcodes)
    print(f"Batch results: {len(batch_results)} executions")
    
    # Get stats
    stats = sandbox.get_sandbox_stats()
    print(f"Sandbox stats: {stats}")

if __name__ == "__main__":
    # Test the security sandbox
    asyncio.run(test_security_sandbox()) 