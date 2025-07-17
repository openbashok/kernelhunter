#!/bin/bash

# KernelHunter Distributed Cluster Deployment Script
# Despliega 10 nodos en Linode y configura el cluster

set -e

# Configuración
LINODE_TOKEN="your-linode-api-token"
CLUSTER_NAME="kernelhunter-cluster"
REGION="us-east"
TYPE="g6-standard-1"  # 1 CPU, 2GB RAM
IMAGE="linode/ubuntu22.04"
ROOT_PASS="$(openssl rand -base64 32)"
SSH_KEY_ID="your-ssh-key-id"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Verificar dependencias
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v curl &> /dev/null; then
        error "curl is required but not installed"
    fi
    
    if ! command -v jq &> /dev/null; then
        error "jq is required but not installed"
    fi
    
    if ! command -v openssl &> /dev/null; then
        error "openssl is required but not installed"
    fi
    
    log "All dependencies found"
}

# Verificar token de Linode
check_linode_token() {
    log "Verifying Linode API token..."
    
    response=$(curl -s -H "Authorization: Bearer $LINODE_TOKEN" \
        "https://api.linode.com/v4/account")
    
    if echo "$response" | jq -e '.errors' > /dev/null; then
        error "Invalid Linode API token"
    fi
    
    log "Linode API token verified"
}

# Crear nodo individual
create_node() {
    local node_id=$1
    local node_name="$CLUSTER_NAME-$node_id"
    
    log "Creating node $node_name..."
    
    # Crear instancia
    response=$(curl -s -X POST \
        -H "Authorization: Bearer $LINODE_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"type\": \"$TYPE\",
            \"region\": \"$REGION\",
            \"image\": \"$IMAGE\",
            \"root_pass\": \"$ROOT_PASS\",
            \"label\": \"$node_name\",
            \"authorized_keys\": [\"$SSH_KEY_ID\"],
            \"tags\": [\"kernelhunter\", \"$CLUSTER_NAME\"]
        }" \
        "https://api.linode.com/v4/linode/instances")
    
    if echo "$response" | jq -e '.errors' > /dev/null; then
        error "Failed to create node $node_name: $(echo "$response" | jq -r '.errors[0].reason')"
    fi
    
    local linode_id=$(echo "$response" | jq -r '.id')
    local ip_address=$(echo "$response" | jq -r '.ipv4[0]')
    
    log "Node $node_name created with ID: $linode_id, IP: $ip_address"
    
    # Esperar a que esté listo
    wait_for_node_ready "$linode_id" "$node_name"
    
    echo "$linode_id:$ip_address"
}

# Esperar a que el nodo esté listo
wait_for_node_ready() {
    local linode_id=$1
    local node_name=$2
    
    log "Waiting for $node_name to be ready..."
    
    while true; do
        response=$(curl -s -H "Authorization: Bearer $LINODE_TOKEN" \
            "https://api.linode.com/v4/linode/instances/$linode_id")
        
        status=$(echo "$response" | jq -r '.status')
        
        if [ "$status" = "running" ]; then
            log "$node_name is ready"
            break
        fi
        
        log "$node_name status: $status, waiting..."
        sleep 10
    done
}

# Configurar nodo individual
setup_node() {
    local ip_address=$1
    local node_id=$2
    
    log "Setting up node $node_id at $ip_address..."
    
    # Esperar a que SSH esté disponible
    wait_for_ssh "$ip_address"
    
    # Instalar dependencias
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 root@"$ip_address" << 'EOF'
        # Actualizar sistema
        apt-get update
        apt-get upgrade -y
        
        # Instalar dependencias básicas
        apt-get install -y python3 python3-pip git curl wget htop
        
        # Instalar compilador
        apt-get install -y clang gcc make
        
        # Instalar dependencias de desarrollo
        apt-get install -y build-essential linux-headers-$(uname -r)
        
        # Instalar dependencias Python
        pip3 install asyncio aiohttp psycopg2-binary numpy matplotlib
        
        # Crear directorio de trabajo
        mkdir -p /root/kernelhunter
        cd /root/kernelhunter
        
        # Clonar KernelHunter (asumiendo que está en un repo público)
        git clone https://github.com/your-repo/kernelhunter.git .
        
        # Configurar firewall básico
        ufw allow ssh
        ufw allow 8000  # Puerto para API
        ufw --force enable
        
        # Crear usuario para KernelHunter
        useradd -m -s /bin/bash kernelhunter
        usermod -aG sudo kernelhunter
        
        # Configurar systemd service
        cat > /etc/systemd/system/kernelhunter.service << 'SERVICE_EOF'
[Unit]
Description=KernelHunter Fuzzer
After=network.target

[Service]
Type=simple
User=kernelhunter
WorkingDirectory=/home/kernelhunter/kernelhunter
ExecStart=/usr/bin/python3 kernelHunter.py --node-id NODE_ID --api-mode
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE_EOF
        
        # Habilitar servicio
        systemctl enable kernelhunter.service
        
        # Configurar logrotate
        cat > /etc/logrotate.d/kernelhunter << 'LOGROTATE_EOF'
/home/kernelhunter/kernelhunter/*.log {
    daily
    missingok
    rotate 7
    compress
    notifempty
    create 644 kernelhunter kernelhunter
}
LOGROTATE_EOF
EOF
    
    log "Node $node_id setup completed"
}

# Esperar a que SSH esté disponible
wait_for_ssh() {
    local ip_address=$1
    
    log "Waiting for SSH to be available on $ip_address..."
    
    while ! nc -z "$ip_address" 22; do
        sleep 5
    done
    
    # Esperar un poco más para que SSH esté completamente listo
    sleep 30
}

# Crear archivo de configuración de nodos
create_nodes_config() {
    local nodes_file="nodes_config.json"
    
    log "Creating nodes configuration file..."
    
    # Crear array de nodos
    local nodes_array="["
    for i in {01..10}; do
        if [ $i -gt 1 ]; then
            nodes_array="$nodes_array,"
        fi
        
        local ip="192.168.1.$((10 + i))"
        nodes_array="$nodes_array{
            \"id\": \"node_$i\",
            \"ip\": \"$ip\",
            \"ssh_port\": 22,
            \"ssh_user\": \"root\",
            \"ssh_key\": \"~/.ssh/id_rsa\",
            \"kernel_target\": \"linux-5.15.0\",
            \"max_generations\": 1000,
            \"population_size\": 100,
            \"use_rl_weights\": true,
            \"timeout\": 3,
            \"checkpoint_interval\": 10
        }"
    done
    nodes_array="$nodes_array]"
    
    # Crear archivo de configuración completo
    cat > "$nodes_file" << EOF
{
  "cluster_name": "KernelHunter-Distributed",
  "api_endpoint": "http://your-central-server:8000",
  "database_url": "postgresql://user:password@localhost/kernelhunter",
  "nodes": $nodes_array,
  "global_settings": {
    "heartbeat_interval": 30,
    "crash_sync_interval": 60,
    "metrics_aggregation_interval": 300,
    "max_retries": 3,
    "auto_restart_failed_nodes": true,
    "load_balancing": true
  }
}
EOF
    
    log "Nodes configuration file created: $nodes_file"
}

# Función principal
main() {
    log "Starting KernelHunter distributed cluster deployment..."
    
    # Verificar dependencias
    check_dependencies
    
    # Verificar token
    check_linode_token
    
    # Crear nodos
    local node_info=()
    for i in {01..10}; do
        log "Creating node $i of 10..."
        info=$(create_node "$i")
        node_info+=("$info")
    done
    
    # Configurar nodos
    for info in "${node_info[@]}"; do
        IFS=':' read -r linode_id ip_address <<< "$info"
        node_id=$(echo "$info" | cut -d':' -f1 | sed 's/.*-//')
        setup_node "$ip_address" "$node_id"
    done
    
    # Crear configuración
    create_nodes_config
    
    # Mostrar resumen
    log "Deployment completed successfully!"
    log "Cluster information:"
    echo "  - Cluster name: $CLUSTER_NAME"
    echo "  - Total nodes: 10"
    echo "  - Region: $REGION"
    echo "  - Instance type: $TYPE"
    echo ""
    log "Node details:"
    for info in "${node_info[@]}"; do
        IFS=':' read -r linode_id ip_address <<< "$info"
        node_id=$(echo "$info" | cut -d':' -f1 | sed 's/.*-//')
        echo "  - $node_id: $ip_address (ID: $linode_id)"
    done
    echo ""
    log "Next steps:"
    echo "  1. Update the API endpoint in nodes_config.json"
    echo "  2. Start the central dashboard: python3 central_dashboard.py"
    echo "  3. Monitor the cluster at: http://your-server:8000"
    echo ""
    log "Root password for all nodes: $ROOT_PASS"
    log "Save this password securely!"
}

# Ejecutar función principal
main "$@" 