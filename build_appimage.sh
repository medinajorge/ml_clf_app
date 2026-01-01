#!/bin/bash
set -e  # Exit on error

echo "Building DMSC AppImage..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf DMSC.AppDir DMSC.AppImage

# Build with PyInstaller
echo "Building with PyInstaller..."
pyinstaller app.spec --clean --noconfirm

# Create AppDir structure
echo "Creating AppDir structure..."
mkdir -p DMSC.AppDir/usr/bin

# Copy executable and dependencies - FIX: Copy contents, not the directory itself
echo "Copying application files..."
cp -r dist/DMSC/* DMSC.AppDir/usr/bin/

# Create desktop file
echo "Creating desktop file..."
cat > DMSC.AppDir/dmsc.desktop << EOF
[Desktop Entry]
Type=Application
Name=DMSC
Comment=DMSC Application
Exec=DMSC
Icon=dmsc
Categories=Science
Terminal=false
EOF

# Copy icons
echo "Copying icons..."
if [ -f "assets/dmsc_v1.png" ]; then
    cp assets/dmsc_v1.png DMSC.AppDir/dmsc.png
    cp assets/dmsc_v1.png DMSC.AppDir/.DirIcon
else
    echo "Warning: Icon file not found!"
fi

# Create AppRun - FIXED paths
echo "Creating AppRun script..."
cat > DMSC.AppDir/AppRun << 'EOF'
#!/bin/bash
SELF=$(readlink -f "$0")
HERE=${SELF%/*}

# Set library path to include _internal directory
export LD_LIBRARY_PATH="${HERE}/usr/bin/_internal:${LD_LIBRARY_PATH}"

# Change to the bin directory where DMSC executable is
cd "${HERE}/usr/bin" || exit 1

# Execute the application
exec "./DMSC" "$@"
EOF

chmod +x DMSC.AppDir/AppRun

# Verify structure
echo "Verifying AppDir structure..."
echo "Checking for required files..."

if [ ! -f "DMSC.AppDir/AppRun" ]; then
    echo "ERROR: AppRun not found!"
    exit 1
fi

if [ ! -f "DMSC.AppDir/dmsc.desktop" ]; then
    echo "ERROR: Desktop file not found!"
    exit 1
fi

if [ ! -f "DMSC.AppDir/usr/bin/DMSC" ]; then
    echo "ERROR: DMSC executable not found!"
    echo "Directory contents:"
    ls -la DMSC.AppDir/usr/bin/
    exit 1
fi

echo "Structure verified. Contents of DMSC.AppDir/usr/bin:"
ls -la DMSC.AppDir/usr/bin/ | head -20

# Build AppImage
echo "Building AppImage..."
if [ -f "./appimagetool-x86_64.AppImage" ]; then
    ./appimagetool-x86_64.AppImage DMSC.AppDir DMSC.AppImage
else
    echo "ERROR: appimagetool-x86_64.AppImage not found!"
    exit 1
fi

# Make executable
chmod +x DMSC.AppImage

echo "✓ AppImage created successfully: DMSC.AppImage"
echo "Run with: ./DMSC.AppImage"
