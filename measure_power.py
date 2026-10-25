#!/usr/bin/env python3
"""
Messung des Energieverbrauchs und der Leistungsaufnahme für neuronale Netze auf dem ESP32-C6.

Ablauf:
  1. Trainiert ein einfaches PyTorch-Modell (MLP) auf dem MNIST-Datensatz auf dem Rechner.
  2. Konvertiert das PyTorch-Modell in einen TensorFlow Lite FlatBuffer und C++ Source/Header.
  3. Baut und flasht die ESP32-Firmware (TensorFlow Lite Micro Inferenz).
  4. Misst mit dem angeschlossenen Joulescope (Trigger an IN1 / Pin 4) hochgenau den
     Energieverbrauch und die Leistungsaufnahme während der Inferenz auf dem ESP32.

Voraussetzungen: joulescope, numpy, torch, torchvision, flatbuffers, tflite
Installation:    pip install -r requirements.txt
"""

import argparse
import logging
import os
import subprocess
import sys
import time

# Logging für Joulescope/jsdrv komplett deaktivieren (OFF), um Mutex-Deadlocks im C-Backend (posix.c) zu verhindern
logging.getLogger("jsdrv").setLevel(logging.CRITICAL)
logging.getLogger("pyjoulescope_driver").setLevel(logging.CRITICAL)
logging.getLogger("joulescope").setLevel(logging.CRITICAL)

# Fallback für virtuelle Umgebungen
try:
    import joulescope
    import numpy as np
except ImportError:
    venv_site = os.path.expanduser("~/venv/joulescope/lib/python3.13/site-packages")
    if os.path.exists(venv_site):
        sys.path.insert(0, venv_site)
    try:
        import joulescope
        import numpy as np
    except ImportError:
        print("Fehler: Das 'joulescope'- oder 'numpy'-Paket wurde nicht gefunden.", file=sys.stderr)
        print("Bitte installieren mit: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

# Treiber-Level des Joulescope C-Backends vollständig stummschalten (OFF)
try:
    from joulescope.v1.driver import DriverWrapper
    _wrapper = DriverWrapper()
    if hasattr(_wrapper, "driver") and _wrapper.driver is not None:
        _wrapper.driver.log_level = "OFF"
except Exception:
    pass

from train_mnist import (
    load_config,
    get_model_configs,
    train_all_models,
    export_model_entry_to_main,
    train_and_export,
)
from datasets_hub import (
    list_available_datasets,
    print_available_datasets_table,
)


def find_esp_idf_export() -> str:
    """
    Sucht nach dem 'export.sh' Skript von ESP-IDF im System.
    """
    if "IDF_PATH" in os.environ:
        export_sh = os.path.join(os.environ["IDF_PATH"], "export.sh")
        if os.path.isfile(export_sh):
            return export_sh

    candidates = [
        os.path.expanduser("~/esp/esp-idf/export.sh"),
        os.path.expanduser("~/.platformio/packages/framework-espidf/export.sh"),
        "/opt/esp/idf/export.sh",
        "/opt/esp-idf/export.sh",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def flash_esp32(port: str = None, project_dir: str = None, build: bool = True) -> bool:
    """
    Kompiliert das ESP-IDF Projekt und flasht es auf den ESP32.
    """
    if project_dir is None:
        project_dir = os.path.dirname(os.path.abspath(__file__))

    export_sh = find_esp_idf_export()
    if not export_sh:
        print("[ESP-IDF] Warnung: 'export.sh' für ESP-IDF nicht gefunden. Flashen übersprungen.", file=sys.stderr)
        return False

    port_opt = f"-p {port} " if port else ""
    cmd_action = f"idf.py {port_opt}build flash" if build else f"idf.py {port_opt}flash"
    full_cmd = f". {export_sh} && cd {project_dir} && {cmd_action}"

    print(f"\n[ESP-IDF] Baue und schreibe Programm auf ESP32 ({cmd_action})...")
    res = subprocess.run(["bash", "-c", full_cmd])
    if res.returncode != 0:
        print(f"\n[ESP-IDF] Fehler: Flashen des ESP32 fehlgeschlagen (Exit-Code {res.returncode}).", file=sys.stderr)
        return False

    print("[ESP-IDF] Programm erfolgreich auf ESP32 geschrieben. Starte Messung...\n")
    return True


class StreamingMetricsAccumulator:
    """
    Sammelt elektrische Messwerte (Strom, Spannung, Leistung) im Streaming-Verfahren (Online-Berechnung),
    ohne Rohdatenarrays unbegrenzt im Arbeitsspeicher abzulegen (O(1) RAM-Verbrauch).
    Verhindert Systemüberlastung und OOM-Abstürze bei langen Inferenzreihen.
    """
    def __init__(self, sampling_freq: float, num_inferences: int = None, title: str = "MESSERGEBNISSE"):
        self.sampling_freq = float(sampling_freq)
        self.num_inferences = num_inferences
        self.title = title
        self.num_samples = 0
        self.sum_voltage = 0.0
        self.min_voltage = float("inf")
        self.max_voltage = float("-inf")
        self.sum_current = 0.0
        self.min_current = float("inf")
        self.max_current = float("-inf")
        self.sum_power = 0.0
        self.min_power = float("inf")
        self.max_power = float("-inf")

    def update(self, current_chunk: np.ndarray, voltage_chunk: np.ndarray):
        if len(current_chunk) == 0:
            return
        n = len(current_chunk)
        self.num_samples += n

        cur = current_chunk.astype(np.float64, copy=False)
        vol = voltage_chunk.astype(np.float64, copy=False)
        pwr = cur * vol

        self.sum_voltage += float(np.sum(vol))
        self.sum_current += float(np.sum(cur))
        self.sum_power += float(np.sum(pwr))

        c_min = float(np.min(cur))
        c_max = float(np.max(cur))
        v_min = float(np.min(vol))
        v_max = float(np.max(vol))
        p_min = float(np.min(pwr))
        p_max = float(np.max(pwr))

        if c_min < self.min_current:
            self.min_current = c_min
        if c_max > self.max_current:
            self.max_current = c_max
        if v_min < self.min_voltage:
            self.min_voltage = v_min
        if v_max > self.max_voltage:
            self.max_voltage = v_max
        if p_min < self.min_power:
            self.min_power = p_min
        if p_max > self.max_power:
            self.max_power = p_max

    def compute_and_display(self) -> dict:
        if self.num_samples == 0:
            print("Keine Messwerte erfasst.", file=sys.stderr)
            return {}

        dt = 1.0 / self.sampling_freq
        duration = self.num_samples * dt

        # Energie in Joule (Ws), mJ, uJ und mWh
        energy_j = self.sum_power * dt
        energy_mj = energy_j * 1000.0
        energy_uj = energy_j * 1e6
        energy_mwh = (energy_j / 3600.0) * 1000.0

        # Ladung in Coulomb (As), mAh und uAh
        charge_c = self.sum_current * dt
        charge_mah = (charge_c / 3600.0) * 1000.0
        charge_uah = charge_mah * 1000.0

        avg_voltage = self.sum_voltage / self.num_samples
        min_voltage = self.min_voltage
        max_voltage = self.max_voltage

        avg_current = self.sum_current / self.num_samples
        min_current = self.min_current
        max_current = self.max_current

        avg_power = self.sum_power / self.num_samples
        min_power = self.min_power
        max_power = self.max_power

        print("\n" + "=" * 65)
        print(f"             {self.title}")
        print("=" * 65)
        if duration < 1.0:
            print(f"Dauer der Inferenzphase: {duration * 1000:.2f} ms ({duration:.6f} s)")
        else:
            print(f"Dauer der Inferenzphase: {duration:.4f} s")
        print(f"Abtastrate:              {self.sampling_freq / 1e6:.2f} MSPS ({self.num_samples:,} Samples)")
        print("-" * 65)
        print(f"Spannung (Durchschnitt): {avg_voltage:.4f} V   [min: {min_voltage:.4f} V, max: {max_voltage:.4f} V]")
        print(f"Strom    (Durchschnitt): {avg_current * 1000:.3f} mA ({avg_current:.6f} A)")
        print(f"         [min: {min_current * 1000:.3f} mA, max: {max_current * 1000:.3f} mA]")
        print(f"Leistung (Durchschnitt): {avg_power * 1000:.3f} mW ({avg_power:.6f} W)")
        print(f"         [min: {min_power * 1000:.3f} mW, max: {max_power * 1000:.3f} mW]")
        print("-" * 65)
        print(f"Elektrische Ladung:      {charge_mah:.4f} mAh ({charge_uah:.2f} µAh / {charge_c:.6f} C)")
        print(f"Energieverbrauch gesamt: {energy_j:.6f} J ({energy_mj:.3f} mJ / {energy_mwh:.6f} mWh)")

        energy_per_inf_mj = None
        energy_per_inf_uj = None
        time_per_inf_ms = None
        inferences_per_sec = None
        charge_per_inf_uah = None

        # Zusätzliche Metriken pro Inferenz, falls bekannt
        if self.num_inferences and self.num_inferences > 0:
            energy_per_inf_mj = energy_mj / self.num_inferences
            energy_per_inf_uj = energy_uj / self.num_inferences
            time_per_inf_ms = (duration * 1000.0) / self.num_inferences
            inferences_per_sec = self.num_inferences / duration if duration > 0 else 0.0
            charge_per_inf_uah = charge_uah / self.num_inferences

            print("-" * 65)
            print("          INFERENZ-METRIKEN (TensorFlow Lite Micro)")
            print("-" * 65)
            print(f"Anzahl Inferenzen:       {self.num_inferences}")
            print(f"Zeit pro Inferenz:       {time_per_inf_ms:.3f} ms ({inferences_per_sec:.1f} Inferenzen/s)")
            print(f"Energie pro Inferenz:    {energy_per_inf_mj:.4f} mJ ({energy_per_inf_uj:.2f} µJ)")
            print(f"Ladung pro Inferenz:     {charge_per_inf_uah:.4f} µAh")
        print("=" * 65 + "\n")

        return {
            "duration_s": duration,
            "num_samples": self.num_samples,
            "sampling_freq": self.sampling_freq,
            "avg_voltage_v": avg_voltage,
            "min_voltage_v": min_voltage,
            "max_voltage_v": max_voltage,
            "avg_current_a": avg_current,
            "avg_current_ma": avg_current * 1000.0,
            "min_current_ma": min_current * 1000.0,
            "max_current_ma": max_current * 1000.0,
            "avg_power_w": avg_power,
            "avg_power_mw": avg_power * 1000.0,
            "min_power_mw": min_power * 1000.0,
            "max_power_mw": max_power * 1000.0,
            "energy_j": energy_j,
            "energy_mj": energy_mj,
            "energy_uj": energy_uj,
            "energy_mwh": energy_mwh,
            "charge_c": charge_c,
            "charge_mah": charge_mah,
            "charge_uah": charge_uah,
            "num_inferences": self.num_inferences,
            "energy_per_inf_mj": energy_per_inf_mj,
            "energy_per_inf_uj": energy_per_inf_uj,
            "time_per_inf_ms": time_per_inf_ms,
            "inferences_per_sec": inferences_per_sec,
            "charge_per_inf_uah": charge_per_inf_uah,
        }


def display_results(
    current: np.ndarray,
    voltage: np.ndarray,
    sampling_freq: float,
    num_inferences: int = None,
    title: str = "MESSERGEBNISSE",
) -> dict:
    """
    Berechnet alle elektrischen Messgrößen, gibt sie formatiert auf stdout aus
    und gibt ein Dictionary mit allen berechneten Metriken zurück.
    """
    accumulator = StreamingMetricsAccumulator(
        sampling_freq=sampling_freq,
        num_inferences=num_inferences,
        title=title,
    )
    accumulator.update(current, voltage)
    return accumulator.compute_and_display()


def capture_esp32_serial_metrics(port: str = None, timeout: float = 60.0, baudrate: int = 115200) -> dict:
    """
    Versucht, über die serielle Schnittstelle die Genauigkeit und Inferenz-Metriken
    direkt aus den Log-Ausgaben des ESP32 zu lesen.
    Timeout 60.0 s: Bei 500 Samples und langen Inferenzzeiten (SNN: ~1.3s/Inferenz)
    kann der METRICS-String erst nach mehreren Minuten erscheinen.
    """
    try:
        import serial
        import glob
    except ImportError:
        return {}

    if port is None:
        ports = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
        if not ports:
            return {}
        port = ports[0]

    if not os.path.exists(port):
        return {}

    metrics = {}
    try:
        with serial.Serial(port, baudrate=baudrate, timeout=1.0) as ser:
            ser.reset_input_buffer()
            start_t = time.time()
            while time.time() - start_t < timeout:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                # Format: ESP32_METRICS: ACC=96.00;CORRECT=48;TOTAL=50;TIME_US=482000
                if "ESP32_METRICS:" in line:
                    parts = line.split("ESP32_METRICS:")[1].strip().split(";")
                    for p in parts:
                        if "=" in p:
                            k, v = p.split("=", 1)
                            k = k.strip()
                            v = v.strip()
                            if k == "ACC":
                                metrics["esp32_acc"] = float(v)
                            elif k == "CORRECT":
                                metrics["esp32_correct"] = int(v)
                            elif k == "TOTAL":
                                metrics["esp32_total"] = int(v)
                            elif k == "TIME_US":
                                metrics["esp32_time_us"] = int(v)
                            elif k == "SPIKES":
                                metrics["esp32_spikes"] = int(v)
                            elif k == "SYNOPS":
                                metrics["esp32_synops"] = int(v)
                    if "esp32_acc" in metrics:
                        break
    except Exception:
        pass

    return metrics


def print_comparison_summary(all_results: list):
    """
    Gibt eine übersichtliche tabellarische Zusammenfassung und den Vergleich
    aller gemessenen Modelle (ANN und SNN) auf stdout aus.
    """
    if not all_results:
        return

    print("\n" + "=" * 184)
    print("                                GESAMTVERGLEICH ALLER GEMESSENEN MODELLE (ANN vs. SNN)")
    print("=" * 184)

    header = f"{'Modell-Name':<20} | {'Typ':<5} | {'Netzwerk-Architektur':<38} | {'Host Acc (10k)':<14} | {'ESP32 Acc (Test)':<18} | {'Größe':<9} | {'Zeit/Inf':<10} | {'Energie/Inf':<12} | {'Mittl. Leist.':<14} | {'Spikes gesamt (Ø/Inf)':<24} | {'SynOps gesamt (Ø/Inf)'}"
    sub_hdr = f"{'':<20} | {'':<5} | {'':<38} | {'(%)':<14} | {'(% [korrekt])':<18} | {'(Bytes)':<9} | {'(ms)':<10} | {'(µJ)':<12} | {'(mW)':<14} | {'(Anzahl)':<24} | {'(Anzahl)'}"
    print(header)
    print(sub_hdr)
    print("-" * 184)

    for res in all_results:
        name = res.get("name", "Modell")[:20]
        m_type = str(res.get("type", "ANN")).upper()[:5]
        structure = res.get("structure", "-")[:38]
        host_acc = f"{res['test_acc']:.2f} %" if res.get("test_acc") is not None else "N/A"

        esp32_acc_val = res.get("esp32_acc")
        if esp32_acc_val is None:
            esp32_acc_val = res.get("subset_acc", res.get("subset_test_acc"))

        corr = res.get("esp32_correct", res.get("subset_correct"))
        tot = res.get("esp32_total", res.get("subset_total", res.get("num_test_samples")))

        if esp32_acc_val is not None:
            if corr is not None and tot is not None:
                esp32_acc_str = f"{esp32_acc_val:.2f} % ({corr}/{tot})"
            else:
                esp32_acc_str = f"{esp32_acc_val:.2f} %"
        else:
            esp32_acc_str = "N/A"

        size_str = f"{res['model_size_bytes']:,}" if res.get("model_size_bytes") else "N/A"

        m = res.get("metrics")
        if m:
            num_inf = res.get("num_test_samples", m.get("num_inferences"))
            esp32_time_us = m.get("esp32_time_us")
            energy_per_inf_uj = m.get("energy_per_inf_uj")

            # Prüfe, ob der GPIO-Trigger verpasst wurde: Energie nahe 0 aber esp32_time_us vorhanden
            gpio_missed = (energy_per_inf_uj is not None and energy_per_inf_uj < 0.1 and
                          esp32_time_us is not None and esp32_time_us > 0 and num_inf)

            if esp32_time_us is not None and num_inf:
                t_inf_val = esp32_time_us / 1000.0 / num_inf  # ms pro Inferenz
                t_inf = f"{t_inf_val:.2f} ms"
                # Wenn GPIO-Trigger verpasst, verwende die Durchschnittsleistung aus der gesamten Messdauer
                if gpio_missed:
                    avg_power_mw = m.get("avg_power_mw_total", m.get("avg_power_mw"))
                else:
                    avg_power_mw = m.get("avg_power_mw")
                if avg_power_mw is not None:
                    e_inf_val = avg_power_mw * t_inf_val  # mW * ms = µJ
                    e_inf = f"{e_inf_val:.2f} µJ"
                else:
                    e_inf = f"{m['energy_per_inf_uj']:.2f} µJ" if m.get("energy_per_inf_uj") is not None else (f"{m['energy_uj']:.1f} µJ" if m.get("energy_uj") is not None else "N/A")
            else:
                t_inf = f"{m['time_per_inf_ms']:.2f} ms" if m.get("time_per_inf_ms") is not None else f"{m['duration_s']*1000:.1f} ms"
                e_inf = f"{m['energy_per_inf_uj']:.2f} µJ" if m.get("energy_per_inf_uj") is not None else (f"{m['energy_uj']:.1f} µJ" if m.get("energy_uj") is not None else "N/A")
            p_avg = f"{m['avg_power_mw']:.2f} mW" if m.get("avg_power_mw") is not None else "N/A"
        else:
            t_inf, e_inf, p_avg = "Keine Daten", "Keine Daten", "Keine Daten"

        spike_str = "-"
        synops_str = "-"
        if m_type == "SNN":
            metrics = res.get("metrics") or {}
            sample_count = metrics.get("esp32_total") or res.get("num_test_samples")
            spikes = metrics.get("esp32_spikes")
            synops = metrics.get("esp32_synops")
            if spikes is not None:
                spike_avg = spikes / sample_count if sample_count else None
                spike_str = f"{spikes:,} ({spike_avg:.1f}/Inf)" if spike_avg is not None else f"{spikes:,}"
            else:
                spike_str = "N/A"
            if synops is not None:
                synops_avg = synops / sample_count if sample_count else None
                synops_str = f"{synops:,} ({synops_avg:.1f}/Inf)" if synops_avg is not None else f"{synops:,}"
            else:
                synops_str = "N/A"

        print(f"{name:<20} | {m_type:<5} | {structure:<38} | {host_acc:<14} | {esp32_acc_str:<18} | {size_str:<9} | {t_inf:<10} | {e_inf:<12} | {p_avg:<14} | {spike_str:<24} | {synops_str}")

    print("=" * 184 + "\n")


def measure_triggered(
    device,
    pin: str = "in1",
    timeout: float = 60.0,
    num_inferences: int = None,
    max_duration: float = None,
) -> dict:
    """
    Misst den Energieverbrauch synchron zu einem muster-basierten Triggersignal an IN1 (oder IN0).

    Protokoll (ESP32 sendet diese Muster):
      Start-Muster:  3x HIGH(200ms) mit LOW(100ms) dazwischen -> Messung(HIGH)
      Messung:       HIGH waehrend der gesamten Inferenz (>= 100ms)
      Ende-Muster:   2x HIGH(200ms) mit LOW(100ms) dazwischen -> LOW(500ms)
    
    Start- und Ende-Muster sind durch die Anzahl der Impulse (3 vs. 2) eindeutig unterscheidbar.

    Ein zustandsbasierter Parser erkennt die Muster auch bei verpassten Samples oder
    Chunk-Grenzen. Der ESP32 laeuft in einer Endlosschleife, also stehen mehrere Zyklen
    zur Verfuegung.
    """
    gpi_field = "gpi1" if pin.lower() in ("in1", "gpi1", "1") else "gpi0"
    # Lese den Mikrocontroller aus der config.yaml, um das Board-Label korrekt anzuzeigen
    try:
        import yaml as yaml_mod
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                mc_config = yaml_mod.safe_load(f)
            mcu_target = mc_config.get("microcontroller", {}).get("target", "esp32c6")
            mcu_boards = mc_config.get("microcontroller", {}).get("boards", {})
            mcu_board = mcu_boards.get(mcu_target, {})
            board_name = mcu_board.get("name", "ESP32")
            trigger_gpio = mcu_board.get("trigger_gpio", 4)
        else:
            board_name = "ESP32"
            trigger_gpio = 4
    except Exception:
        board_name = "ESP32"
        trigger_gpio = 4
    channel_label = f"IN1 ({board_name} Pin {trigger_gpio})" if gpi_field == "gpi1" else "IN0"

    # Muster-Toleranzen (in ms) — der ESP32 sendet exakte Zeiten, aber wir erlauben Spielraum
    # Ein Impuls im Muster: HIGH ~200ms (Toleranz 150-300ms), LOW zwischen Impulsen ~100ms (Toleranz 50-200ms)
    PULSE_HIGH_MIN_MS = 150.0
    PULSE_HIGH_MAX_MS = 300.0
    GAP_LOW_MIN_MS = 50.0
    GAP_LOW_MAX_MS = 200.0
    # Die eigentliche Messung dauert mindestens 100ms (Inferenz ist immer laenger)
    MIN_MEASURE_MS = 100.0
    # Nach dem Start-Muster: LOW-Phase vor Messung ~1100ms (100ms Gap + 1000ms Wartezeit)
    # Wir warten auf HIGH nach >= 300ms LOW (um das Start-Muster von der Messung zu trennen)
    START_TO_MEASURE_GAP_MIN_MS = 300.0

    # Berechne maximal zulässige Messdauer (dynamisch nach Anzahl Samples)
    if max_duration is None:
        max_duration = max(60.0, (num_inferences or 50) * 2.0)

    # Puffergröße im Joulescope-Treiber auf 10s begrenzen, um RAM-Verbrauch minimal zu halten
    try:
        device.parameter_set("buffer_duration", 10)
    except Exception:
        pass

    sampling_freq = device.output_sampling_frequency
    samples_per_ms = sampling_freq / 1000.0

    print(f"Abtastrate Joulescope:   {sampling_freq / 1e6:.2f} MSPS")
    print(f"Warte auf Startmuster an {channel_label}...")
    print(f"  Protokoll: Start-Muster (3x Impuls) -> Messung(HIGH) -> Ende-Muster (2x Impuls)")
    if timeout > 0:
        print(f"Trigger-Timeout:         {timeout:.1f} Sekunden (gesamt)")
    if max_duration > 0:
        print(f"Max. Messdauer:          {max_duration:.1f} Sekunden (pro Zyklus)")

    device.start()

    accumulator = StreamingMetricsAccumulator(
        sampling_freq=sampling_freq,
        num_inferences=num_inferences,
        title=f"MESSERGEBNISSE MNIST INFERENZ (Trigger {channel_label})",
    )
    # Zweiter Accumulator für die gesamte Messdauer (inklusive Wartezeit)
    total_accumulator = StreamingMetricsAccumulator(
        sampling_freq=sampling_freq,
        num_inferences=num_inferences,
        title="GESAMTE MESSDAUER",
    )
    last_processed_id = None

    # Zustandsmaschine fuer Muster-Erkennung
    # Start-Muster: 3 Impulse (unterscheidbar vom Ende-Muster mit 2 Impulsen)
    # Staende:
    #   IDLE           - Wartet auf den ersten HIGH-Impuls (Start-Muster Puls 1)
    #   START_PULSE1   - Erster HIGH-Impuls des Start-Musters
    #   START_GAP1     - LOW zwischen Puls 1 und Puls 2 des Start-Musters
    #   START_PULSE2   - Zweiter HIGH-Impuls des Start-Musters
    #   START_GAP2     - LOW zwischen Puls 2 und Puls 3 des Start-Musters
    #   START_PULSE3   - Dritter HIGH-Impuls des Start-Musters
    #   START_WAIT     - LOW nach Start-Muster, wartet auf Messung (HIGH)
    #   MEASURING      - Messung laeuft (HIGH)
    #   END_GAP1       - LOW nach Messung, vor Ende-Muster Puls 1
    #   END_PULSE1     - Erster HIGH-Impuls des Ende-Musters
    #   END_GAP2       - LOW zwischen Puls 1 und Puls 2 des Ende-Musters
    #   END_PULSE2     - Zweiter HIGH-Impuls des Ende-Musters (nach diesem ist die Messung fertig)
    STATE_IDLE = 0
    STATE_START_PULSE1 = 1
    STATE_START_GAP1 = 2
    STATE_START_PULSE2 = 3
    STATE_START_GAP2 = 4
    STATE_START_PULSE3 = 5
    STATE_START_WAIT = 6
    STATE_MEASURING = 7
    STATE_END_GAP1 = 8
    STATE_END_PULSE1 = 9
    STATE_END_GAP2 = 10
    STATE_END_PULSE2 = 11

    state = STATE_IDLE
    state_start_samples = 0  # Sample-Index, an dem der aktuelle Zustand begonnen hat
    last_gpi_value = 0
    t_wait_start = time.time()
    t_measure_start = None
    t_last_progress = 0.0
    retry_count = 0
    pattern_start_idx = 0  # Sample-Index, ab dem das Start-Muster begonnen hat (für Accumulator-Slice)

    def reset_for_retry():
        """Zustand und Accumulator zuruecksetzen fuer einen neuen Versuch."""
        nonlocal state, last_gpi_value, t_measure_start, t_last_progress, accumulator
        state = STATE_IDLE
        last_gpi_value = 0
        t_measure_start = None
        t_last_progress = 0.0
        accumulator = StreamingMetricsAccumulator(
            sampling_freq=sampling_freq,
            num_inferences=num_inferences,
            title=f"MESSERGEBNISSE MNIST INFERENZ (Trigger {channel_label})",
        )

    def ms_to_samples(ms):
        return int(ms * samples_per_ms)

    def samples_to_ms(samples):
        return samples / samples_per_ms

    try:
        while True:
            now = time.time()

            # Timeout-Pruefung
            if state != STATE_MEASURING:
                if timeout > 0 and (now - t_wait_start) > timeout:
                    raise TimeoutError(
                        f"Zeitueberschreitung: Kein gueltiges Startmuster an {channel_label} innerhalb von {timeout:.1f} s empfangen."
                    )
            else:
                elapsed = now - t_measure_start
                if elapsed - t_last_progress >= 5.0:
                    t_last_progress = elapsed
                    print(f"  [Messung aktiv] {elapsed:.1f} s vergangen ({accumulator.num_samples:,} Samples erfasst)...", flush=True)

                if max_duration > 0 and elapsed > max_duration:
                    print(f"\n[Trigger] >>> Timeout: Messung dauerte laenger als {max_duration:.1f} s. <<<")
                    retry_count += 1
                    print(f"[Trigger] Versuch #{retry_count}, warte auf naechstes Startmuster...")
                    reset_for_retry()

            time.sleep(0.01)  # 10 ms Polling-Intervall
            s1, s2 = device.stream_buffer.sample_id_range
            if s1 is None or s2 is None or s2 <= s1:
                continue

            if last_processed_id is None:
                last_processed_id = s1
            if last_processed_id < s1:
                last_processed_id = s1
            if s2 <= last_processed_id:
                continue

            # Datenchunk abrufen
            try:
                data = device.stream_buffer.samples_get(
                    last_processed_id, s2, fields=["current", "voltage", gpi_field]
                )
            except ValueError:
                continue

            cur_chunk = data["signals"]["current"]["value"]
            vol_chunk = data["signals"]["voltage"]["value"]
            gpi_chunk = data["signals"][gpi_field]["value"]
            chunk_len = len(gpi_chunk)
            last_processed_id = s2

            # Total-Accumulator immer mit allen Daten füttern
            total_accumulator.update(cur_chunk, vol_chunk)

            # Zustandsmaschine: Verarbeite den Chunk Sample-fuer-Sample
            chunk_start_global = last_processed_id - chunk_len
            
            # Fuer MEASURING-Zustand: sammle die Indices der Mess-Samples in diesem Chunk
            measure_start_in_chunk = None
            measure_end_in_chunk = None

            for i in range(chunk_len):
                gpi_val = gpi_chunk[i]
                global_idx = chunk_start_global + i
                samples_in_state = global_idx - state_start_samples

                if state == STATE_IDLE:
                    # In IDLE: Jeder HIGH-Wert startet die Mustererkennung.
                    # Ein Rising-Edge-Check (last_gpi_value == 0) wuerde den Start verpassen,
                    # wenn der erste Chunk bereits HIGH zeigt oder der Edge ueber Chunk-Grenzen liegt.
                    if gpi_val == 1:
                        state = STATE_START_PULSE1
                        state_start_samples = global_idx
                        pattern_start_idx = global_idx

                elif state == STATE_START_PULSE1:
                    if gpi_val == 0:
                        pulse_ms = samples_to_ms(samples_in_state)
                        if pulse_ms < PULSE_HIGH_MIN_MS or pulse_ms > PULSE_HIGH_MAX_MS:
                            state = STATE_IDLE
                        else:
                            state = STATE_START_GAP1
                            state_start_samples = global_idx

                elif state == STATE_START_GAP1:
                    if gpi_val == 1:
                        gap_ms = samples_to_ms(samples_in_state)
                        if gap_ms < GAP_LOW_MIN_MS or gap_ms > GAP_LOW_MAX_MS:
                            state = STATE_IDLE
                        else:
                            state = STATE_START_PULSE2
                            state_start_samples = global_idx

                elif state == STATE_START_PULSE2:
                    if gpi_val == 0:
                        pulse_ms = samples_to_ms(samples_in_state)
                        if pulse_ms < PULSE_HIGH_MIN_MS or pulse_ms > PULSE_HIGH_MAX_MS:
                            state = STATE_IDLE
                        else:
                            state = STATE_START_GAP2
                            state_start_samples = global_idx

                elif state == STATE_START_GAP2:
                    if gpi_val == 1:
                        gap_ms = samples_to_ms(samples_in_state)
                        if gap_ms < GAP_LOW_MIN_MS or gap_ms > GAP_LOW_MAX_MS:
                            state = STATE_IDLE
                        else:
                            state = STATE_START_PULSE3
                            state_start_samples = global_idx

                elif state == STATE_START_PULSE3:
                    if gpi_val == 0:
                        pulse_ms = samples_to_ms(samples_in_state)
                        if pulse_ms < PULSE_HIGH_MIN_MS or pulse_ms > PULSE_HIGH_MAX_MS:
                            state = STATE_IDLE
                        else:
                            print(f"\n[Trigger] >>> Start-Muster (3 Impulse) erkannt! Warte auf Messung... <<<")
                            state = STATE_START_WAIT
                            state_start_samples = global_idx

                elif state == STATE_START_WAIT:
                    # Nach dem Start-Muster: Jeder HIGH-Wert nach ausreichender LOW-Wartezeit
                    # startet die Messung — kein Rising-Edge-Check, da der Edge verpasst werden koennte.
                    if gpi_val == 1:
                        wait_ms = samples_to_ms(samples_in_state)
                        if wait_ms < START_TO_MEASURE_GAP_MIN_MS:
                            state = STATE_IDLE
                        else:
                            print(f"[Trigger] >>> Messung gestartet (nach {wait_ms:.0f} ms Wartezeit). <<<")
                            state = STATE_MEASURING
                            state_start_samples = global_idx
                            t_measure_start = time.time()
                            t_last_progress = 0.0
                            measure_start_in_chunk = i  # Messung beginnt bei diesem Sample

                elif state == STATE_MEASURING:
                    if gpi_val == 0:
                        measure_end_in_chunk = i  # Messung endet bei diesem Sample
                        measure_ms = samples_to_ms(samples_in_state)
                        if measure_ms < MIN_MEASURE_MS:
                            retry_count += 1
                            print(f"[Trigger] >>> Messung zu kurz ({measure_ms:.0f} ms), retry... <<<")
                            reset_for_retry()
                            break
                        else:
                            print(f"[Trigger] >>> Messung beendet ({measure_ms:.0f} ms), warte auf Ende-Muster... <<<")
                            state = STATE_END_GAP1
                            state_start_samples = global_idx

                elif state == STATE_END_GAP1:
                    if gpi_val == 1:
                        gap_ms = samples_to_ms(samples_in_state)
                        if gap_ms < GAP_LOW_MIN_MS or gap_ms > GAP_LOW_MAX_MS:
                            retry_count += 1
                            print(f"[Trigger] >>> Ende-Muster nicht erkannt, retry... <<<")
                            reset_for_retry()
                            break
                        else:
                            state = STATE_END_PULSE1
                            state_start_samples = global_idx

                elif state == STATE_END_PULSE1:
                    if gpi_val == 0:
                        pulse_ms = samples_to_ms(samples_in_state)
                        if pulse_ms < PULSE_HIGH_MIN_MS or pulse_ms > PULSE_HIGH_MAX_MS:
                            retry_count += 1
                            print(f"[Trigger] >>> Ende-Muster fehlerhaft, retry... <<<")
                            reset_for_retry()
                            break
                        else:
                            state = STATE_END_GAP2
                            state_start_samples = global_idx

                elif state == STATE_END_GAP2:
                    if gpi_val == 1:
                        gap_ms = samples_to_ms(samples_in_state)
                        if gap_ms < GAP_LOW_MIN_MS or gap_ms > GAP_LOW_MAX_MS:
                            retry_count += 1
                            print(f"[Trigger] >>> Ende-Muster fehlerhaft, retry... <<<")
                            reset_for_retry()
                            break
                        else:
                            state = STATE_END_PULSE2
                            state_start_samples = global_idx

                elif state == STATE_END_PULSE2:
                    if gpi_val == 0:
                        pulse_ms = samples_to_ms(samples_in_state)
                        if pulse_ms < PULSE_HIGH_MIN_MS or pulse_ms > PULSE_HIGH_MAX_MS:
                            retry_count += 1
                            print(f"[Trigger] >>> Ende-Muster unvollstaendig, retry... <<<")
                            reset_for_retry()
                            break
                        else:
                            print(f"[Trigger] >>> Ende-Muster erkannt! Messung abgeschlossen. <<<")
                            raise SystemExit(0)

                last_gpi_value = gpi_val

            # Nach der Sample-fuer-Sample-Verarbeitung: Accumulator mit den Mess-Samples fuettern
            if measure_start_in_chunk is not None:
                if measure_end_in_chunk is not None:
                    # Messung hat in diesem Chunk begonnen UND geendet
                    accumulator.update(cur_chunk[measure_start_in_chunk:measure_end_in_chunk],
                                       vol_chunk[measure_start_in_chunk:measure_end_in_chunk])
                else:
                    # Messung beginnt in diesem Chunk und laeuft weiter
                    accumulator.update(cur_chunk[measure_start_in_chunk:],
                                       vol_chunk[measure_start_in_chunk:])
            elif measure_end_in_chunk is not None:
                # Messung lief bereits vor diesem Chunk und endet in diesem Chunk
                accumulator.update(cur_chunk[:measure_end_in_chunk], vol_chunk[:measure_end_in_chunk])
            elif state == STATE_MEASURING:
                # Messung lief bereits vor diesem Chunk und laeuft weiter (kein Ende in diesem Chunk)
                accumulator.update(cur_chunk, vol_chunk)

    except SystemExit as e:
        if e.code != 0:
            raise
    finally:
        try:
            device.stop()
        except Exception:
            pass
        time.sleep(0.05)

    # Zusammenfassung
    if retry_count > 0:
        print(f"\n[Trigger] {retry_count} fehlerhafte(r) Zyklus(se) wiederholt.")

    # GPIO-Trigger-Messung anzeigen
    metrics = accumulator.compute_and_display()

    # Durchschnittsleistung aus der gesamten Messdauer hinzufügen
    if total_accumulator.num_samples > 0:
        total_dt = 1.0 / sampling_freq
        total_duration = total_accumulator.num_samples * total_dt
        total_avg_power = total_accumulator.sum_power / total_accumulator.num_samples
        metrics["avg_power_mw_total"] = total_avg_power * 1000.0  # mW
        metrics["duration_total_s"] = total_duration
        print(f"[Messung] Durchschnittsleistung (gesamte Dauer): {total_avg_power * 1000:.2f} mW ({total_duration:.2f} s)")

    return metrics


def measure_duration(device, duration: float = 10.0, num_inferences: int = None) -> dict:
    """
    Misst den Energieverbrauch über eine feste Dauer in Sekunden.
    """
    buffer_duration = max(30, int(duration * 2))
    try:
        device.parameter_set("buffer_duration", buffer_duration)
    except Exception:
        pass

    print(f"Messung ueber {duration:.1f} Sekunden gestartet...")
    data = device.read(contiguous_duration=duration)

    current = data[:, 0]
    voltage = data[:, 1]
    sampling_freq = device.output_sampling_frequency

    return display_results(
        current,
        voltage,
        sampling_freq,
        num_inferences=num_inferences,
        title=f"MESSERGEBNISSE ({duration:.1f}s Dauer)",
    )


def main():
    parser = argparse.ArgumentParser(
        description="MNIST PyTorch-Training, TFLite-Export, ESP32-Flashing und Joulescope Energiemessung."
    )
    # Konfigurationsdatei
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Pfad zur Konfigurationsdatei (YAML oder JSON, Standard: config.yaml / config.json)",
    )

    # Trainings-Optionen
    parser.add_argument(
        "-D", "--dataset",
        type=str,
        default=None,
        help="Waehlt den Datensatz (z.B. 'mnist', 'fashion_mnist', 'cifar10', 'speech_commands', 'har_time_series', 'eeg_data', 'ecg_data', 'synthetic_time_series')",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Zeigt alle verfuegbaren Datensaetze und Domaenen in einer Tabelle an",
    )
    parser.add_argument(
        "--train",
        dest="train",
        action="store_true",
        default=True,
        help="PyTorch-Modelle vor dem Flashen trainieren und exportieren (Standard: aktiviert)",
    )
    parser.add_argument(
        "--no-train",
        dest="train",
        action="store_false",
        help="Training ueberspringen und vorhandenes exportiertes Modell verwenden",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=None,
        help="Anzahl der Trainingsepochen fuer PyTorch (ueberschreibt Konfigurationsdatei)",
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=None,
        help="Batch-Groesse fuer das Training (ueberschreibt Konfigurationsdatei)",
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=None,
        dest="learning_rate",
        help="Lernrate fuer das Training (ueberschreibt Konfigurationsdatei)",
    )
    parser.add_argument(
        "--early-stopping",
        dest="early_stopping",
        action="store_true",
        default=None,
        help="Aktiviert Early Stopping bei stagnierender Genauigkeit",
    )
    parser.add_argument(
        "--no-early-stopping",
        dest="early_stopping",
        action="store_false",
        help="Deaktiviert Early Stopping",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Anzahl der Epochen ohne Verbesserung vor vorzeitigem Trainingsabbruch",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Mindestverbesserung der Metrik fuer Early Stopping",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default=None,
        help="Zu ueberwachende Metrik fuer Early Stopping ('test_acc' oder 'test_loss')",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Anzahl der MNIST Testbilder fuer die Inferenz auf dem ESP32 (ueberschreibt Konfigurationsdatei)",
    )
    parser.add_argument(
        "--force-train", "--force-retrain",
        dest="force_train",
        action="store_true",
        default=False,
        help="Erzwingt ein erneutes Training aller Modelle, auch wenn gespeicherte Checkpoints unverändert vorliegen",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Verzeichnis zum automatischen Speichern und Laden trainierter Modell-Checkpoints (Standard: './saved_models')",
    )

    # Post-Training Optimierungs-Optionen
    parser.add_argument(
        "--prune", "--prune-amount",
        type=float,
        default=None,
        dest="prune_amount",
        help="Aktiviert Post-Training Pruning und setzt den Zielanteil entfernter Gewichte (z. B. 0.3 für 30%% Sparsity)",
    )
    parser.add_argument(
        "--prune-method",
        type=str,
        default=None,
        choices=["l1_unstructured", "random_unstructured", "structured", "threshold"],
        help="Pruning-Methode: 'l1_unstructured' (Standard), 'random_unstructured', 'structured' oder 'threshold'",
    )
    parser.add_argument(
        "--quantize", "--quantize-type",
        type=str,
        default=None,
        dest="quantize_type",
        choices=["int8", "int4", "fp16", "dynamic"],
        help="Aktiviert Post-Training Quantisierung der Gewichte ('int8', 'int4', 'fp16', 'dynamic')",
    )
    parser.add_argument(
        "--clustering", "--num-clusters",
        type=int,
        default=None,
        dest="num_clusters",
        help="Aktiviert Weight-Clustering mit k Centroids (z. B. 16 für 4-Bit Codebook)",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=None,
        help="Anzahl an Fine-Tuning Epochen nach dem Pruning zur Genauigkeitswiederherstellung",
    )

    # Mess- und Trigger-Optionen
    parser.add_argument(
        "-p", "--pin",
        type=str,
        default=None,
        help="Joulescope Trigger-Eingangspins: 'in1' (Standard) oder 'in0' (ueberschreibt Konfiguration)",
    )
    parser.add_argument(
        "-t", "--timeout",
        type=float,
        default=None,
        help="Timeout in Sekunden beim Warten auf das Startsignal (0 für unendlich, ueberschreibt Konfiguration)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Maximale Messdauer in Sekunden waehrend der Inferenz (0 für unendlich, ueberschreibt Konfiguration)",
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=None,
        help="Dauer-Modus: Feste Messdauer in Sekunden (ueberschreibt Trigger-Modus, z. B. -d 10.0)",
    )

    # ESP32 Flash-Optionen
    parser.add_argument(
        "--flash",
        dest="flash",
        action="store_true",
        default=True,
        help="ESP32-Programm vor der Messung automatisch bauen und flashen (Standard: aktiviert)",
    )
    parser.add_argument(
        "--no-flash",
        dest="flash",
        action="store_false",
        help="Automatisches Flashen des ESP32 vor der Messung deaktivieren",
    )
    parser.add_argument(
        "-P", "--port",
        type=str,
        default=None,
        help="Serieller Port des ESP32 (z. B. /dev/ttyACM0 oder /dev/ttyUSB0, Standard: automatische Erkennung)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        default=False,
        help="Nur flashen ohne vorheriges 'idf.py build'",
    )
    parser.add_argument(
        "--wait",
        dest="wait_unplug",
        action="store_true",
        default=True,
        help="Nach dem Flashen auf [Enter] warten, um das USB-Kabel abziehen zu koennen (Standard: aktiviert)",
    )
    parser.add_argument(
        "--no-wait",
        dest="wait_unplug",
        action="store_false",
        help="Nicht auf Tastendruck nach dem Flashen warten",
    )

    args = parser.parse_args()

    if args.list_datasets:
        print_available_datasets_table()
        sys.exit(0)

    # Konfiguration laden
    config = load_config(args.config)
    if args.dataset:
        config["dataset"] = args.dataset

    num_test_samples = (
        args.test_samples
        if args.test_samples is not None
        else config.get("evaluation", {}).get("num_test_samples", 500)
    )
    trigger_pin = (
        args.pin
        if args.pin is not None
        else config.get("joulescope", {}).get("trigger_pin", "in1")
    )
    timeout = (
        args.timeout
        if args.timeout is not None
        else config.get("joulescope", {}).get("timeout", 60.0)
    )
    max_duration = (
        args.max_duration
        if args.max_duration is not None
        else config.get("joulescope", {}).get("max_duration", None)
    )

    # -------------------------------------------------------------
    # PHASE 1: Training aller Modelle
    # -------------------------------------------------------------
    if args.train:
        trained_models = train_all_models(
            config_or_path=config,
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            early_stopping=args.early_stopping,
            patience=args.patience,
            min_delta=args.min_delta,
            monitor=args.monitor,
            num_test_samples=args.test_samples,
            force_retrain=args.force_train,
            save_dir=args.save_dir,
            prune_amount=args.prune_amount,
            prune_method=args.prune_method,
            quantize_type=args.quantize_type,
            num_clusters=args.num_clusters,
            fine_tune_epochs=args.fine_tune_epochs,
        )
    else:
        model_configs = get_model_configs(config, dataset_filter=args.dataset)
        trained_models = []
        for m in model_configs:
            trained_models.append({
                "name": m["name"],
                "dataset": m.get("dataset", "mnist"),
                "structure": f"Input {m['model']['input_size']} -> {m['model']['hidden_layers']} -> {m['model']['output_size']}",
                "config": m,
                "test_acc": None,
                "model_size_bytes": None,
                "num_test_samples": num_test_samples,
            })

    if not trained_models:
        print("Keine Modelle zum Messen vorhanden.", file=sys.stderr)
        sys.exit(1)

    total_models = len(trained_models)

    # -------------------------------------------------------------
    # PHASE 2: Nacheinander auf ESP32 flashen und messen
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f">>> PHASE 2: SEQUENTIELLES FLASHEN & JOULESCOPE-MESSUNG ({total_models} Modell{'e' if total_models > 1 else ''}) <<<")
    print("=" * 70)

    devices = joulescope.scan()
    if not devices:
        print("Kein Joulescope-Gerät gefunden. Bitte USB-Verbindung prüfen.", file=sys.stderr)
        sys.exit(1)

    device = devices[0]
    if hasattr(device, "_driver") and hasattr(device._driver, "log_level"):
        try:
            device._driver.log_level = "OFF"
        except Exception:
            pass
    print(f"Verbunden mit Joulescope: {device.device_serial_number or device}\n")

    all_results = []

    try:
        device.open()

        for idx, model_entry in enumerate(trained_models, start=1):
            model_name = model_entry["name"]
            model_samples = model_entry.get("num_test_samples", num_test_samples)

            print("\n" + "#" * 70)
            print(f"### [Modell {idx}/{total_models}]: '{model_name}' ({model_entry.get('structure', '')}) ###")
            print("#" * 70 + "\n")

            # 1. C/C++ Header für dieses Modell nach main/ schreiben
            if "tflite_bytes" in model_entry and "test_dataset" in model_entry:
                export_model_entry_to_main(model_entry, output_dir="main")

            # 2. ESP32 bei Bedarf bauen und flashen
            flash_failed = False
            if args.flash and args.duration is None:
                flash_ok = flash_esp32(port=args.port, build=not args.skip_build)
                if not flash_ok:
                    print(f"[ESP-IDF] Warnung: Flashen für Modell '{model_name}' war nicht erfolgreich. Messung wird übersprungen.", file=sys.stderr)
                    flash_failed = True
                else:
                    if args.wait_unplug:
                        print("\n" + "=" * 65)
                        print(f">>> ESP32 erfolgreich für Modell '{model_name}' geflasht! <<<")
                        print("Sie können jetzt das USB-Kabel vom ESP32 abziehen, damit die")
                        print("USB-Versorgung das Messergebnis nicht beeinflusst.")
                        print("(Hinweis: ESP32 muss weiterhin über Joulescope/3.3V versorgt sein).")
                        print("=" * 65)
                        try:
                            input(f"\nDrücken Sie [Enter], um die Joulescope-Messung für '{model_name}' zu starten...")
                        except (KeyboardInterrupt, EOFError):
                            print("\nMessung durch Benutzer abgebrochen.")
                            break

            if flash_failed:
                continue

            # 3. Joulescope-Messung durchführen
            metrics = {}
            try:
                if args.duration is not None and args.duration > 0:
                    metrics = measure_duration(device, duration=args.duration, num_inferences=model_samples)
                else:
                    metrics = measure_triggered(device, pin=trigger_pin, timeout=timeout, num_inferences=model_samples, max_duration=max_duration)
            except TimeoutError as te:
                print(f"\n[Messung fehlgeschlagen] {te}", file=sys.stderr)
            except KeyboardInterrupt:
                print("\nMessung durch Benutzer abgebrochen.", file=sys.stderr)
                break
            except Exception as e:
                print(f"\n[Messung fehlgeschlagen] Unerwarteter Fehler bei Modell '{model_name}': {e}", file=sys.stderr)

            # 4. Serielle Logs NACH der Messung lesen (USB muss wieder angeschlossen sein)
            serial_metrics = {}
            if args.flash and args.duration is None:
                if args.wait_unplug:
                    print("\n" + "=" * 65)
                    print(f"Bitte schließen Sie das USB-Kabel wieder an den ESP32 an, um")
                    print(f"die Serielle Logs auszulesen.")
                    print("=" * 65)
                    try:
                        input("Drücken Sie [Enter], sobald das USB-Kabel wieder angeschlossen ist...")
                    except (KeyboardInterrupt, EOFError):
                        print("\nMessung durch Benutzer abgebrochen.")
                        break
                serial_metrics = capture_esp32_serial_metrics(port=args.port, timeout=60.0)
                if serial_metrics and "esp32_acc" in serial_metrics:
                    print(f"[ESP32] Ermittelte Test-Genauigkeit (über UART): {serial_metrics['esp32_acc']:.2f}% ({serial_metrics.get('esp32_correct')}/{serial_metrics.get('esp32_total')} korrekt)")

            esp32_acc = model_entry.get("subset_acc", model_entry.get("subset_test_acc"))
            esp32_corr = model_entry.get("subset_correct")
            esp32_tot = model_entry.get("subset_total", model_samples)
            if serial_metrics and "esp32_acc" in serial_metrics:
                esp32_acc = serial_metrics["esp32_acc"]
                esp32_corr = serial_metrics.get("esp32_correct", esp32_corr)
                esp32_tot = serial_metrics.get("esp32_total", esp32_tot)

            # ESP32-Serien-Metriken in die Joulescope-Metriken übernehmen
            if serial_metrics:
                metrics.update({k: v for k, v in serial_metrics.items() if k.startswith("esp32_")})

            if esp32_acc is not None:
                print(f"[ESP32 Test-Accuracy] Modell '{model_name}': {esp32_acc:.2f}% ({esp32_corr}/{esp32_tot} Samples korrekt klassifiziert)")

            all_results.append({
                "name": model_name,
                "type": model_entry.get("type", "ann"),
                "structure": model_entry.get("structure", "-"),
                "test_acc": model_entry.get("test_acc"),
                "subset_acc": model_entry.get("subset_acc", model_entry.get("subset_test_acc")),
                "subset_test_acc": model_entry.get("subset_test_acc", model_entry.get("subset_acc")),
                "subset_correct": model_entry.get("subset_correct"),
                "subset_total": model_entry.get("subset_total"),
                "esp32_acc": esp32_acc,
                "esp32_correct": esp32_corr,
                "esp32_total": esp32_tot,
                "model_size_bytes": model_entry.get("model_size_bytes"),
                "metrics": metrics,
                "num_test_samples": model_samples,
            })

            # Wenn weitere Modelle folgen und wait_unplug aktiv ist:
            if idx < total_models and args.flash and args.wait_unplug and args.duration is None:
                next_model_name = trained_models[idx]["name"]
                print("\n" + "-" * 65)
                print(f"Messung für '{model_name}' abgeschlossen.")
                print(f"Bitte schließen Sie das USB-Kabel wieder an den ESP32 an, um")
                print(f"das nächste Modell ('{next_model_name}') zu flashen.")
                print("-" * 65)
                try:
                    input("Drücken Sie [Enter], sobald das USB-Kabel wieder angeschlossen ist...")
                except (KeyboardInterrupt, EOFError):
                    print("\nMessreihe durch Benutzer beendet.")
                    break

    except KeyboardInterrupt:
        print("\nMessung durch Benutzer abgebrochen.", file=sys.stderr)
    except TimeoutError as te:
        print(f"\n{te}", file=sys.stderr)
    except Exception as e:
        print(f"\nFehler bei der Messung: {e}", file=sys.stderr)
    finally:
        try:
            device.close()
            print("Joulescope-Verbindung getrennt.")
        except Exception:
            pass

    # -------------------------------------------------------------
    # PHASE 3: Gesamtvergleich ausgeben
    # -------------------------------------------------------------
    if all_results:
        print_comparison_summary(all_results)


if __name__ == "__main__":
    main()
