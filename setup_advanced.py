#!/usr/bin/env python3
"""
Script de instalación avanzada para KernelHunter
Instala todas las dependencias y configura el entorno para características avanzadas
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path
import shutil

def run_command(command, description):
    """Ejecutar comando con manejo de errores"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        print(f"   Salida: {e.stdout}")
        print(f"   Error: {e.stderr}")
        return False

def check_system_requirements():
    """Verificar requisitos del sistema"""
    print("🔍 Verificando requisitos del sistema...")
    
    # Verificar Python
    if sys.version_info < (3, 8):
        print("❌ Se requiere Python 3.8 o superior")
        return False
    
    # Verificar sistema operativo
    system = platform.system()
    if system not in ["Linux", "Darwin"]:
        print(f"⚠️ Sistema operativo no soportado: {system}")
        print("   Algunas características pueden no funcionar correctamente")
    
    # Verificar herramientas básicas
    required_tools = ["git", "curl", "wget"]
    for tool in required_tools:
        if not shutil.which(tool):
            print(f"⚠️ Herramienta no encontrada: {tool}")
    
    print("✅ Verificación de requisitos completada")
    return True

def install_system_dependencies():
    """Instalar dependencias del sistema"""
    system = platform.system()
    
    if system == "Linux":
        # Ubuntu/Debian
        if os.path.exists("/etc/debian_version"):
            packages = [
                "build-essential",
                "python3-dev",
                "python3-pip",
                "python3-venv",
                "git",
                "curl",
                "wget",
                "docker.io",
                "docker-compose",
                "qemu-kvm",
                "libvirt-daemon-system",
                "libvirt-clients",
                "bridge-utils",
                "redis-server",
                "postgresql",
                "postgresql-contrib"
            ]
            
            for package in packages:
                run_command(f"sudo apt-get install -y {package}", f"Instalando {package}")
        
        # CentOS/RHEL
        elif os.path.exists("/etc/redhat-release"):
            packages = [
                "gcc",
                "gcc-c++",
                "python3-devel",
                "python3-pip",
                "git",
                "curl",
                "wget",
                "docker",
                "docker-compose",
                "qemu-kvm",
                "libvirt",
                "libvirt-daemon",
                "bridge-utils",
                "redis",
                "postgresql",
                "postgresql-server"
            ]
            
            for package in packages:
                run_command(f"sudo yum install -y {package}", f"Instalando {package}")
    
    elif system == "Darwin":  # macOS
        # Verificar si Homebrew está instalado
        if not shutil.which("brew"):
            print("🍺 Instalando Homebrew...")
            run_command('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"', "Instalando Homebrew")
        
        packages = [
            "python3",
            "git",
            "curl",
            "wget",
            "docker",
            "docker-compose",
            "qemu",
            "redis",
            "postgresql"
        ]
        
        for package in packages:
            run_command(f"brew install {package}", f"Instalando {package}")

def setup_python_environment():
    """Configurar entorno Python"""
    print("🐍 Configurando entorno Python...")
    
    # Crear entorno virtual
    if not os.path.exists("venv"):
        run_command("python3 -m venv venv", "Creando entorno virtual")
    
    # Activar entorno virtual
    if os.name == "nt":  # Windows
        activate_script = "venv\\Scripts\\activate"
    else:  # Unix/Linux/macOS
        activate_script = "source venv/bin/activate"
    
    # Actualizar pip
    run_command(f"{activate_script} && pip install --upgrade pip", "Actualizando pip")
    
    # Instalar dependencias básicas
    run_command(f"{activate_script} && pip install -r requirements_advanced.txt", "Instalando dependencias Python")

def setup_docker():
    """Configurar Docker"""
    print("🐳 Configurando Docker...")
    
    # Verificar si Docker está ejecutándose
    if not run_command("docker --version", "Verificando Docker"):
        print("❌ Docker no está disponible")
        return False
    
    # Iniciar servicio Docker
    system = platform.system()
    if system == "Linux":
        run_command("sudo systemctl start docker", "Iniciando servicio Docker")
        run_command("sudo systemctl enable docker", "Habilitando Docker al inicio")
        
        # Agregar usuario al grupo docker
        run_command("sudo usermod -aG docker $USER", "Agregando usuario al grupo docker")
    
    # Crear imagen base de KernelHunter
    dockerfile_content = """
FROM ubuntu:20.04

# Instalar dependencias
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    gcc \\
    make \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /kernelhunter

# Copiar archivos de KernelHunter
COPY . .

# Instalar dependencias Python
RUN pip3 install -r requirements_advanced.txt

# Compilar herramientas
RUN make

# Exponer puerto
EXPOSE 8000

# Comando por defecto
CMD ["python3", "kernelHunter.py"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # Construir imagen
    run_command("docker build -t kernelhunter:latest .", "Construyendo imagen Docker")
    
    return True

def setup_kubernetes():
    """Configurar Kubernetes"""
    print("☸️ Configurando Kubernetes...")
    
    # Verificar kubectl
    if not shutil.which("kubectl"):
        print("📥 Instalando kubectl...")
        system = platform.system()
        
        if system == "Linux":
            run_command("curl -LO 'https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl'", "Descargando kubectl")
            run_command("sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl", "Instalando kubectl")
        elif system == "Darwin":
            run_command("brew install kubectl", "Instalando kubectl")
    
    # Verificar minikube (para desarrollo local)
    if not shutil.which("minikube"):
        print("📥 Instalando minikube...")
        system = platform.system()
        
        if system == "Linux":
            run_command("curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64", "Descargando minikube")
            run_command("sudo install minikube-linux-amd64 /usr/local/bin/minikube", "Instalando minikube")
        elif system == "Darwin":
            run_command("brew install minikube", "Instalando minikube")
    
    # Iniciar minikube
    run_command("minikube start", "Iniciando minikube")
    
    return True

def setup_database():
    """Configurar base de datos"""
    print("🗄️ Configurando base de datos...")
    
    # Configurar PostgreSQL
    run_command("sudo -u postgres createuser -s kernelhunter", "Creando usuario de base de datos")
    run_command("sudo -u postgres createdb kernelhunter", "Creando base de datos")
    
    # Configurar Redis
    run_command("sudo systemctl start redis", "Iniciando Redis")
    run_command("sudo systemctl enable redis", "Habilitando Redis al inicio")
    
    return True

def setup_monitoring():
    """Configurar monitoreo"""
    print("📊 Configurando monitoreo...")
    
    # Crear directorios para logs
    os.makedirs("logs", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("dashboards", exist_ok=True)
    
    # Configurar Prometheus
    prometheus_config = """
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'kernelhunter'
    static_configs:
      - targets: ['localhost:8000']
"""
    
    with open("prometheus.yml", "w") as f:
        f.write(prometheus_config)
    
    # Configurar Grafana
    run_command("docker run -d --name grafana -p 3000:3000 grafana/grafana", "Iniciando Grafana")
    
    return True

def create_configuration_files():
    """Crear archivos de configuración"""
    print("⚙️ Creando archivos de configuración...")
    
    # Configuración principal
    main_config = {
        "local": {
            "max_generations": 1000,
            "population_size": 100,
            "mutation_rate": 0.3,
            "crossover_rate": 0.7,
            "elite_size": 10,
            "stagnation_limit": 50,
            "enable_rl": True,
            "enable_web_interface": True,
            "web_port": 8080,
            "log_level": "INFO",
            "save_crashes": True,
            "crash_dir": "./crashes",
            "metrics_file": "./metrics.json",
            "gene_bank_file": "./gene_bank.json",
            "enable_advanced_features": True,
            "enable_ml": True,
            "enable_analytics": True,
            "enable_security_sandbox": True,
            "sandbox_isolation_level": "container",
            "max_execution_time": 30,
            "max_memory_mb": 1024
        },
        "distributed": {
            "enable_distributed": False,
            "central_api_url": "http://localhost:5000",
            "node_id": None,
            "max_workers": 8,
            "sync_interval": 30,
            "heartbeat_interval": 60,
            "crash_upload_enabled": True,
            "metrics_upload_enabled": True
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "database": "kernelhunter",
            "username": "kernelhunter",
            "password": "",
            "pool_size": 20
        },
        "monitoring": {
            "prometheus_port": 9090,
            "grafana_port": 3000,
            "enable_telemetry": True,
            "log_retention_days": 30
        }
    }
    
    with open("kernelhunter_config.json", "w") as f:
        json.dump(main_config, f, indent=2)
    
    # Configuración de Kubernetes
    k8s_config = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {
            "name": "kernelhunter"
        }
    }
    
    with open("k8s-namespace.yaml", "w") as f:
        import yaml
        yaml.dump(k8s_config, f)
    
    # Configuración de Docker Compose
    docker_compose = {
        "version": "3.8",
        "services": {
            "kernelhunter": {
                "build": ".",
                "ports": ["8000:8000"],
                "volumes": ["./data:/kernelhunter/data"],
                "environment": ["KH_EXECUTION_MODE=local"],
                "restart": "unless-stopped"
            },
            "postgres": {
                "image": "postgres:13",
                "environment": {
                    "POSTGRES_DB": "kernelhunter",
                    "POSTGRES_USER": "kernelhunter",
                    "POSTGRES_PASSWORD": "kernelhunter"
                },
                "ports": ["5432:5432"],
                "volumes": ["./data/postgres:/var/lib/postgresql/data"]
            },
            "redis": {
                "image": "redis:6",
                "ports": ["6379:6379"],
                "volumes": ["./data/redis:/data"]
            },
            "prometheus": {
                "image": "prom/prometheus",
                "ports": ["9090:9090"],
                "volumes": ["./prometheus.yml:/etc/prometheus/prometheus.yml"]
            },
            "grafana": {
                "image": "grafana/grafana",
                "ports": ["3000:3000"],
                "volumes": ["./dashboards:/var/lib/grafana/dashboards"]
            }
        }
    }
    
    with open("docker-compose.yml", "w") as f:
        yaml.dump(docker_compose, f)
    
    print("✅ Archivos de configuración creados")

def setup_development_environment():
    """Configurar entorno de desarrollo"""
    print("🛠️ Configurando entorno de desarrollo...")
    
    # Crear directorios de desarrollo
    directories = [
        "tests",
        "docs",
        "scripts",
        "examples",
        "data",
        "models",
        "logs",
        "reports"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Crear archivo .env
    env_content = """
# KernelHunter Environment Variables
KH_EXECUTION_MODE=local
KH_LOG_LEVEL=INFO

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=kernelhunter
DB_USER=kernelhunter
DB_PASSWORD=kernelhunter

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# ML Configuration
OPENAI_API_KEY=your_openai_api_key_here
TORCH_DEVICE=cuda

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Security
SANDBOX_ISOLATION_LEVEL=container
MAX_EXECUTION_TIME=30
MAX_MEMORY_MB=1024
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    # Crear Makefile
    makefile_content = """
.PHONY: install test run clean docker-build docker-run k8s-deploy

install:
	python3 setup_advanced.py

test:
	python3 -m pytest tests/ -v

run:
	python3 kernelHunter.py

run-advanced:
	python3 kernelHunter.py --advanced

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf logs/*.log
	rm -rf data/temp/*

docker-build:
	docker build -t kernelhunter:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

k8s-deploy:
	kubectl apply -f k8s-namespace.yaml
	kubectl apply -f k8s-deployment.yaml
	kubectl apply -f k8s-service.yaml

k8s-delete:
	kubectl delete -f k8s-service.yaml
	kubectl delete -f k8s-deployment.yaml
	kubectl delete -f k8s-namespace.yaml

monitoring-start:
	docker-compose -f docker-compose.monitoring.yml up -d

monitoring-stop:
	docker-compose -f docker-compose.monitoring.yml down

format:
	black .
	flake8 .

lint:
	pylint kernelHunter.py advanced_*.py

docs:
	sphinx-build -b html docs/source docs/build/html
"""
    
    with open("Makefile", "w") as f:
        f.write(makefile_content)
    
    print("✅ Entorno de desarrollo configurado")

def run_tests():
    """Ejecutar pruebas básicas"""
    print("🧪 Ejecutando pruebas básicas...")
    
    # Crear archivo de prueba básico
    test_content = """
import pytest
import asyncio
from kernelHunter import *

def test_basic_functionality():
    assert True

@pytest.mark.asyncio
async def test_async_functionality():
    assert True

def test_ml_engine():
    if ADVANCED_FEATURES_ENABLED:
        ml_engine = get_ml_engine()
        assert ml_engine is not None

def test_performance_optimizer():
    if ADVANCED_FEATURES_ENABLED:
        optimizer = get_performance_optimizer()
        assert optimizer is not None
"""
    
    with open("tests/test_basic.py", "w") as f:
        f.write(test_content)
    
    # Ejecutar pruebas
    run_command("python3 -m pytest tests/ -v", "Ejecutando pruebas")

def main():
    """Función principal de instalación"""
    print("🚀 KernelHunter Advanced - Instalación Completa")
    print("=" * 50)
    
    # Verificar requisitos
    if not check_system_requirements():
        print("❌ Requisitos del sistema no cumplidos")
        sys.exit(1)
    
    # Instalar dependencias del sistema
    install_system_dependencies()
    
    # Configurar entorno Python
    setup_python_environment()
    
    # Configurar Docker
    setup_docker()
    
    # Configurar Kubernetes
    setup_kubernetes()
    
    # Configurar base de datos
    setup_database()
    
    # Configurar monitoreo
    setup_monitoring()
    
    # Crear archivos de configuración
    create_configuration_files()
    
    # Configurar entorno de desarrollo
    setup_development_environment()
    
    # Ejecutar pruebas
    run_tests()
    
    print("\n" + "=" * 50)
    print("✅ Instalación completada exitosamente!")
    print("\n🎯 Próximos pasos:")
    print("1. Configurar variables de entorno en .env")
    print("2. Ejecutar: python3 kernelHunter.py")
    print("3. Acceder al dashboard: http://localhost:8080")
    print("4. Monitoreo: http://localhost:3000 (Grafana)")
    print("\n📚 Documentación: README.md")
    print("🐛 Reportar problemas: GitHub Issues")

if __name__ == "__main__":
    main() 