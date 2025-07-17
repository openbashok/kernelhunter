#!/bin/bash

# KernelHunter Web Interface Launcher
# Este script inicia tanto el backend PHP como el frontend React

echo "🚀 Starting KernelHunter Web Interface..."

# Verificar que estamos en el directorio correcto
if [ ! -f "kernelHunter.py" ]; then
    echo "❌ Error: Please run this script from the KernelHunter root directory"
    exit 1
fi

# Verificar dependencias
echo "📋 Checking dependencies..."

# Verificar PHP
if ! command -v php &> /dev/null; then
    echo "❌ Error: PHP is not installed"
    echo "Please install PHP: sudo apt-get install php php-cli"
    exit 1
fi

# Verificar Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed"
    echo "Please install Node.js: https://nodejs.org/"
    exit 1
fi

# Verificar npm
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed"
    exit 1
fi

echo "✅ Dependencies OK"

# Crear directorios si no existen
echo "📁 Setting up directories..."
mkdir -p web_api
mkdir -p web_frontend

# Verificar si el frontend está configurado
if [ ! -f "web_frontend/package.json" ]; then
    echo "❌ Error: Frontend not configured. Please run setup_web_interface.sh first"
    exit 1
fi

# Instalar dependencias del frontend si es necesario
echo "📦 Installing frontend dependencies..."
cd web_frontend
if [ ! -d "node_modules" ]; then
    npm install
fi
cd ..

# Función para limpiar procesos al salir
cleanup() {
    echo "🛑 Shutting down web interface..."
    kill $PHP_PID $REACT_PID 2>/dev/null
    exit 0
}

# Capturar Ctrl+C
trap cleanup SIGINT

# Iniciar servidor PHP
echo "🌐 Starting PHP backend server..."
cd web_api
php -S localhost:8000 ../index.php &
PHP_PID=$!
cd ..

# Esperar un momento para que PHP se inicie
sleep 2

# Verificar que PHP está funcionando
if ! curl -s http://localhost:8000/api/stats > /dev/null; then
    echo "❌ Error: PHP backend failed to start"
    kill $PHP_PID 2>/dev/null
    exit 1
fi

echo "✅ PHP backend running on http://localhost:8000"

# Iniciar servidor React
echo "⚛️  Starting React frontend..."
cd web_frontend
npm start &
REACT_PID=$!
cd ..

# Esperar un momento para que React se inicie
sleep 5

echo ""
echo "🎉 KernelHunter Web Interface is running!"
echo ""
echo "📊 Dashboard: http://localhost:3000"
echo "🔧 API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Mantener el script corriendo
wait 