#!/bin/bash

set -e

echo "[*] Instalando KernelHunter para todos los usuarios..."

# Ruta de instalación
INSTALL_DIR="/opt/kernelhunter"
SCRIPT_NAME="kernelhunter.py"
EXECUTABLE_NAME="kernelhunter"
SYMLINK_PATH="/usr/local/bin/$EXECUTABLE_NAME"

# Crear carpeta si no existe
mkdir -p "$INSTALL_DIR"

# Copiar el script
cp "$SCRIPT_NAME" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/$SCRIPT_NAME"

# Crear symlink global
ln -sf "$INSTALL_DIR/$SCRIPT_NAME" "$SYMLINK_PATH"

# Asegurar permisos
chmod -R o+rx "$INSTALL_DIR"

# Crear carpetas de trabajo globales si querés
mkdir -p /var/log/kernelhunter
chmod 777 /var/log/kernelhunter

echo "[+] Instalación completa."
echo ">> Ejecutá con: $EXECUTABLE_NAME"

