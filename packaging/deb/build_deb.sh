#!/usr/bin/env bash
set -euo pipefail

VERSION="1.0.0"
ARCH="amd64"
PACKAGE_NAME="espectroapp"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

EXECUTABLE_DIR="$PROJECT_ROOT/dist/EspectroApp"
PACKAGE_ROOT="$PROJECT_ROOT/build/deb/${PACKAGE_NAME}_${VERSION}_${ARCH}"
OUTPUT_DIR="$PROJECT_ROOT/dist/packages"

if [[ ! -x "$EXECUTABLE_DIR/EspectroApp" ]]; then
    echo "ERROR: No se encontró el ejecutable:"
    echo "  $EXECUTABLE_DIR/EspectroApp"
    echo
    echo "Primero ejecuta:"
    echo "  ./packaging/linux/build_linux.sh"
    exit 1
fi

if [[ ! -f "$PROJECT_ROOT/packaging/deb/espectroapp.png" ]]; then
    echo "ERROR: Falta el icono:"
    echo "  packaging/deb/espectroapp.png"
    exit 1
fi

rm -rf "$PACKAGE_ROOT"

mkdir -p "$PACKAGE_ROOT/DEBIAN"
mkdir -p "$PACKAGE_ROOT/opt/espectroapp"
mkdir -p "$PACKAGE_ROOT/usr/share/applications"
mkdir -p "$PACKAGE_ROOT/usr/share/icons/hicolor/512x512/apps"
mkdir -p "$OUTPUT_DIR"

# Copiar toda la aplicación generada por PyInstaller.
cp -a "$EXECUTABLE_DIR/." "$PACKAGE_ROOT/opt/espectroapp/"

# Copiar acceso del menú e icono.
cp "$PROJECT_ROOT/packaging/deb/espectroapp.desktop" \
   "$PACKAGE_ROOT/usr/share/applications/espectroapp.desktop"

cp "$PROJECT_ROOT/packaging/deb/espectroapp.png" \
   "$PACKAGE_ROOT/usr/share/icons/hicolor/512x512/apps/espectroapp.png"

# Archivo de control del paquete.
cat > "$PACKAGE_ROOT/DEBIAN/control" <<EOF
Package: $PACKAGE_NAME
Version: $VERSION
Section: science
Priority: optional
Architecture: $ARCH
Maintainer: EspectroApp Development Team
Description: Open platform for spectral data analysis
 EspectroApp provides spectral preprocessing, visualization,
 multivariate analysis, hierarchical clustering and data fusion
 through an interactive graphical interface.
EOF

# Permisos.
chmod 755 "$PACKAGE_ROOT/DEBIAN"
chmod 755 "$PACKAGE_ROOT/opt/espectroapp/EspectroApp"
chmod 644 "$PACKAGE_ROOT/usr/share/applications/espectroapp.desktop"
chmod 644 "$PACKAGE_ROOT/usr/share/icons/hicolor/512x512/apps/espectroapp.png"

# Construir paquete.
dpkg-deb --build \
    --root-owner-group \
    "$PACKAGE_ROOT" \
    "$OUTPUT_DIR/${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"

echo
echo "Paquete creado correctamente:"
echo "  $OUTPUT_DIR/${PACKAGE_NAME}_${VERSION}_${ARCH}.deb"