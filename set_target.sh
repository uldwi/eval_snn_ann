#!/bin/bash
# set_target.sh - Select an ESP32 target and rebuild the project
#
# Usage:
#   ./set_target.sh esp32c6    # Select ESP32-C6
#   ./set_target.sh esp32s3    # Select ESP32-S3
#   ./set_target.sh            # Show the current target and help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Supported targets
VALID_TARGETS=("esp32c6" "esp32s3")

# Read the current target from config.yaml
get_config_target() {
    python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('microcontroller', {}).get('target', 'esp32c6'))
" 2>/dev/null || echo "esp32c6"
}

# Update the target in config.yaml
set_config_target() {
    local target="$1"
    python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['microcontroller']['target'] = '$target'
with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
"
}

# Read board details from config.yaml
get_board_info() {
    local target="$1"
    python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
boards = config.get('microcontroller', {}).get('boards', {})
board = boards.get('$target', {})
print(board.get('name', '$target'))
print(board.get('board', 'Unknown'))
print(board.get('architecture', 'Unknown'))
print(board.get('cpu_freq_mhz', '?'))
print(board.get('trigger_gpio', '?'))
" 2>/dev/null
}

# Display help
show_help() {
    echo ""
    echo "Usage: ./set_target.sh <target>"
    echo ""
    echo "Supported targets:"
    for t in "${VALID_TARGETS[@]}"; do
        info=$(get_board_info "$t")
        name=$(echo "$info" | sed -n '1p')
        board=$(echo "$info" | sed -n '2p')
        arch=$(echo "$info" | sed -n '3p')
        freq=$(echo "$info" | sed -n '4p')
        echo "  $t  -  $name ($board, $arch, ${freq} MHz)"
    done
    echo ""
    echo "Current target in config.yaml: $(get_config_target)"
    echo ""
    echo "Example:"
    echo "  ./set_target.sh esp32s3    # Select ESP32-S3"
    echo ""
}

# Check arguments
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

TARGET="$1"

# Check that the target is valid
if ! printf "%s\n" "${VALID_TARGETS[@]}" | grep -qw "$TARGET"; then
    echo "Error: Unknown target '$TARGET'"
    echo "Supported targets: ${VALID_TARGETS[*]}"
    exit 1
fi

# Read board details
info=$(get_board_info "$TARGET")
name=$(echo "$info" | sed -n '1p')
board=$(echo "$info" | sed -n '2p')
arch=$(echo "$info" | sed -n '3p')
freq=$(echo "$info" | sed -n '4p')
gpio=$(echo "$info" | sed -n '5p')

echo ""
echo "================================================================"
echo "  Select microcontroller target"
echo "================================================================"
echo ""
echo "  Target:      $TARGET"
echo "  Board:       $name ($board)"
echo "  Architecture: $arch (${freq} MHz)"
echo "  Trigger-GPIO: $gpio"
echo ""

# 1. Update config.yaml
echo "[1/4] Updating config.yaml..."
set_config_target "$TARGET"
echo "       config.yaml: microcontroller.target = $TARGET"

# 2. Clear the CMake cache
echo "[2/4] Removing CMake cache (build/)..."
if [ -d "build" ]; then
    rm -rf build
    echo "       build/ removed"
else
    echo "       build/ does not exist (clean start)"
fi

# 3. Set the ESP-IDF target
echo "[3/4] Setting ESP-IDF target to $TARGET..."
. "$HOME/esp/esp-idf/export.sh" 2>/dev/null
idf.py set-target "$TARGET"

# 4. Build the project
echo "[4/4] Building project..."
idf.py build

echo ""
echo "================================================================"
echo "  Done! Current target: $TARGET ($name)"
echo "================================================================"
echo ""
echo "To flash the board:"
echo "  idf.py flash monitor"
echo ""
