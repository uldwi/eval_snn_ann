#!/bin/bash
# set_target.sh - Wechselt den ESP32-Mikrocontroller-Target und baut das Projekt neu
#
# Verwendung:
#   ./set_target.sh esp32c6    # Wechselt zu ESP32-C6
#   ./set_target.sh esp32s3    # Wechselt zu ESP32-S3
#   ./set_target.sh            # Zeigt aktuellen Target und Hilfe an

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Verfügbare Targets
VALID_TARGETS=("esp32c6" "esp32s3")

# Lese den aktuellen Target aus der config.yaml
get_config_target() {
    python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('microcontroller', {}).get('target', 'esp32c6'))
" 2>/dev/null || echo "esp32c6"
}

# Setze den Target in der config.yaml
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

# Hole Board-Info aus config.yaml
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

# Zeige Hilfe
show_help() {
    echo ""
    echo "Verwendung: ./set_target.sh <target>"
    echo ""
    echo "Verfügbare Targets:"
    for t in "${VALID_TARGETS[@]}"; do
        info=$(get_board_info "$t")
        name=$(echo "$info" | sed -n '1p')
        board=$(echo "$info" | sed -n '2p')
        arch=$(echo "$info" | sed -n '3p')
        freq=$(echo "$info" | sed -n '4p')
        echo "  $t  -  $name ($board, $arch, ${freq} MHz)"
    done
    echo ""
    echo "Aktueller Target in config.yaml: $(get_config_target)"
    echo ""
    echo "Beispiel:"
    echo "  ./set_target.sh esp32s3    # Wechselt zu ESP32-S3"
    echo ""
}

# Prüfe Argumente
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

TARGET="$1"

# Prüfe, ob Target gültig ist
if ! printf "%s\n" "${VALID_TARGETS[@]}" | grep -qw "$TARGET"; then
    echo "Fehler: Unbekanntes Target '$TARGET'"
    echo "Verfügbare Targets: ${VALID_TARGETS[*]}"
    exit 1
fi

# Hole Board-Info
info=$(get_board_info "$TARGET")
name=$(echo "$info" | sed -n '1p')
board=$(echo "$info" | sed -n '2p')
arch=$(echo "$info" | sed -n '3p')
freq=$(echo "$info" | sed -n '4p')
gpio=$(echo "$info" | sed -n '5p')

echo ""
echo "================================================================"
echo "  Mikrocontroller-Target wechseln"
echo "================================================================"
echo ""
echo "  Target:      $TARGET"
echo "  Board:       $name ($board)"
echo "  Architektur: $arch (${freq} MHz)"
echo "  Trigger-GPIO: $gpio"
echo ""

# 1. config.yaml aktualisieren
echo "[1/4] Aktualisiere config.yaml..."
set_config_target "$TARGET"
echo "       config.yaml: microcontroller.target = $TARGET"

# 2. CMake-Cache löschen
echo "[2/4] Lösche CMake-Cache (build/)..."
if [ -d "build" ]; then
    rm -rf build
    echo "       build/ gelöscht"
else
    echo "       build/ existiert nicht (sauberer Start)"
fi

# 3. ESP-IDF Target setzen
echo "[3/4] Setze ESP-IDF Target auf $TARGET..."
. "$HOME/esp/esp-idf/export.sh" 2>/dev/null
idf.py set-target "$TARGET"

# 4. Projekt bauen
echo "[4/4] Baue Projekt..."
idf.py build

echo ""
echo "================================================================"
echo "  Fertig! Target ist jetzt: $TARGET ($name)"
echo "================================================================"
echo ""
echo "Zum Flashen auf das Board:"
echo "  idf.py flash monitor"
echo ""
