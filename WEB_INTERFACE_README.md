# 🌐 KernelHunter Web Interface

Una interfaz web moderna y en tiempo real para monitorear y analizar la evolución de KernelHunter.

## 🚀 Características

### **Dashboard en Tiempo Real**
- **Gráficos interactivos** de evolución de generaciones
- **Métricas en vivo** de crash rates y system impacts
- **Visualización de población** evolutiva
- **Estadísticas de ataques** y mutaciones

### **Explorador de Crashes**
- **Filtros avanzados** por tipo y severidad
- **Análisis detallado** de shellcodes
- **Búsqueda inteligente** en crashes
- **Exportación de datos** en múltiples formatos

### **Análisis de Shellcodes**
- **Disassembler visual** de instrucciones
- **Análisis de entropía** y patrones
- **Detección automática** de instrucciones comunes
- **Historial de mutaciones** y evolución

## 🏗️ Arquitectura

```
KernelHunter Web Interface/
├── web_api/                    # Backend PHP
│   └── index.php              # API REST principal
├── web_frontend/              # Frontend React
│   ├── src/
│   │   ├── components/        # Componentes React
│   │   ├── App.js            # Aplicación principal
│   │   └── App.css           # Estilos modernos
│   └── package.json          # Dependencias React
├── start_web_interface.sh     # Script de inicio
└── WEB_INTERFACE_README.md    # Esta documentación
```

## 📋 Requisitos

### **Sistema**
- **PHP 7.4+** con extensiones JSON y cURL
- **Node.js 16+** y npm
- **Navegador moderno** (Chrome, Firefox, Safari, Edge)

### **Instalación de Dependencias**

#### **Ubuntu/Debian**
```bash
# PHP
sudo apt-get update
sudo apt-get install php php-cli php-json php-curl

# Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

#### **CentOS/RHEL**
```bash
# PHP
sudo yum install php php-cli php-json php-curl

# Node.js
curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
sudo yum install -y nodejs
```

#### **macOS**
```bash
# PHP
brew install php

# Node.js
brew install node
```

## 🚀 Instalación y Configuración

### **1. Configuración Automática**
```bash
# Desde el directorio raíz de KernelHunter
chmod +x start_web_interface.sh
./start_web_interface.sh
```

### **2. Configuración Manual**

#### **Backend PHP**
```bash
# Crear directorio para la API
mkdir -p web_api

# El archivo index.php ya está creado con toda la funcionalidad
# Iniciar servidor PHP
cd web_api
php -S localhost:8000 ../index.php
```

#### **Frontend React**
```bash
# Crear aplicación React
npx create-react-app web_frontend
cd web_frontend

# Instalar dependencias adicionales
npm install react-router-dom react-chartjs-2 chart.js

# Copiar archivos de componentes
# (Los archivos ya están creados en el proyecto)

# Iniciar servidor de desarrollo
npm start
```

## 🌐 Uso de la Interfaz

### **Acceso**
- **Dashboard Principal**: http://localhost:3000
- **API Backend**: http://localhost:8000
- **Documentación API**: http://localhost:8000/api/stats

### **Navegación**

#### **Dashboard Principal**
- **Gráfico de Crash Rate**: Evolución de la tasa de crashes por generación
- **System Impacts**: Impactos del sistema en tiempo real
- **Shellcode Length**: Evolución de la longitud promedio de shellcodes
- **Crash Distribution**: Distribución de tipos de crashes encontrados
- **Attack Types**: Tipos de ataques más utilizados

#### **Explorador de Crashes**
- **Filtros**: Por tipo (crítico, alto, medio, bajo) y búsqueda
- **Vista de Tarjetas**: Información resumida de cada crash
- **Panel de Detalles**: Análisis completo al hacer clic en un crash
- **Acciones**: Análisis completo y exportación de datos

#### **Análisis de Shellcodes**
- **Vista Hex**: Representación hexadecimal del shellcode
- **Análisis de Patrones**: Detección de instrucciones comunes
- **Cálculo de Entropía**: Medida de aleatoriedad del shellcode
- **Historial**: Evolución del shellcode a través de generaciones

## 🔧 API Endpoints

### **Métricas Generales**
```http
GET /api/metrics
```
**Respuesta:**
```json
{
  "metrics": {
    "generations": [0, 1, 2, 3],
    "crash_rates": [0.75, 0.80, 0.85, 0.90],
    "system_impacts": [2, 5, 8, 12],
    "shellcode_lengths": [12.2, 17.8, 22.1, 25.5]
  },
  "stats": {
    "total_generations": 4,
    "avg_crash_rate": 0.825,
    "total_system_impacts": 27
  }
}
```

### **Estadísticas**
```http
GET /api/stats
```
**Respuesta:**
```json
{
  "total_generations": 4,
  "total_crashes": 150,
  "critical_crashes": 25,
  "avg_crash_rate": 0.825,
  "most_common_crash": "SIGNAL_SIGSEGV",
  "total_shellcodes": 600,
  "avg_shellcode_length": 19.4
}
```

### **Crashes**
```http
GET /api/crashes
GET /api/critical
```
**Respuesta:**
```json
{
  "crashes": [
    {
      "generation": 3,
      "program_id": 15,
      "shellcode_hex": "4831c00f05...",
      "crash_type": "SIGNAL_SIGSEGV",
      "system_impact": true,
      "timestamp": 1640995200
    }
  ],
  "count": 150
}
```

### **Población**
```http
GET /api/population
```
**Respuesta:**
```json
{
  "population": [
    {
      "id": "0001",
      "shellcode": "4831c00f05",
      "length": 5
    }
  ],
  "count": 50
}
```

### **Análisis de Shellcode**
```http
GET /api/shellcode?hash=4831c00f05
```
**Respuesta:**
```json
{
  "shellcode": "4831c00f05",
  "analysis": {
    "length": 5,
    "patterns": {
      "xor": 1,
      "syscall": 1
    },
    "possible_instructions": [
      "xor rax, rax",
      "syscall"
    ],
    "entropy": 3.2
  },
  "crashes": [...]
}
```

## 🎨 Personalización

### **Temas de Colores**
Los colores se pueden personalizar editando las variables CSS en `web_frontend/src/App.css`:

```css
:root {
  --primary-color: #2563eb;
  --secondary-color: #7c3aed;
  --success-color: #059669;
  --warning-color: #d97706;
  --danger-color: #dc2626;
  --critical-color: #991b1b;
  --background-color: #0f172a;
  --surface-color: #1e293b;
}
```

### **Configuración del Backend**
Editar `web_api/index.php` para modificar:
- Rutas de archivos de KernelHunter
- Endpoints de la API
- Lógica de análisis de shellcodes

## 🔍 Troubleshooting

### **Problemas Comunes**

#### **Backend PHP no inicia**
```bash
# Verificar que PHP está instalado
php --version

# Verificar puerto disponible
netstat -tulpn | grep :8000

# Iniciar con puerto diferente
php -S localhost:8080 ../index.php
```

#### **Frontend React no inicia**
```bash
# Verificar Node.js
node --version
npm --version

# Limpiar cache
npm cache clean --force

# Reinstalar dependencias
rm -rf node_modules package-lock.json
npm install
```

#### **API no responde**
```bash
# Verificar que KernelHunter está generando datos
ls -la kernelhunter_metrics.json
ls -la kernelhunter_crashes/

# Verificar permisos de archivos
chmod 644 kernelhunter_metrics.json
chmod -R 755 kernelhunter_crashes/
```

#### **CORS Errors**
Si hay problemas de CORS, verificar que el proxy está configurado en `package.json`:
```json
{
  "proxy": "http://localhost:8000"
}
```

## 🚀 Despliegue en Producción

### **Backend PHP**
```bash
# Usar Apache o Nginx
sudo apt-get install apache2 php libapache2-mod-php

# Configurar virtual host
sudo nano /etc/apache2/sites-available/kernelhunter.conf
```

### **Frontend React**
```bash
# Build para producción
cd web_frontend
npm run build

# Servir con nginx
sudo apt-get install nginx
sudo cp -r build/* /var/www/html/
```

## 🤝 Contribuir

### **Nuevos Componentes**
1. Crear componente en `web_frontend/src/components/`
2. Añadir ruta en `App.js`
3. Actualizar navegación en `Navigation.js`

### **Nuevos Endpoints API**
1. Añadir función en `web_api/index.php`
2. Documentar en esta README
3. Actualizar frontend para usar el nuevo endpoint

### **Mejoras de UI/UX**
1. Modificar estilos en `App.css`
2. Añadir animaciones y transiciones
3. Mejorar responsividad móvil

## 📞 Soporte

- **Issues**: Crear issue en el repositorio de KernelHunter
- **Documentación**: Revisar esta README y comentarios en el código
- **Comunidad**: Unirse al canal de Discord/Slack del proyecto

---

**¡Happy Fuzzing con la Web Interface! 🎯** 