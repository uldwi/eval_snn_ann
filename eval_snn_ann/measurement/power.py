#!/usr/bin/env python3
"""Train, export, flash, and measure ANN/SNN inference on ESP32 with Joulescope.

The workflow can train configured PyTorch models, export supported models to
TFLite/C, build and flash the ESP-IDF firmware, and measure device power.

Requirements: joulescope, numpy, torch, torchvision, flatbuffers, tflite
Install project dependencies with: pip install -r requirements.txt
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Disable Joulescope/jsdrv logging completely (OFF) to prevent mutex deadlocks in the C backend (posix.c)
logging.getLogger("jsdrv").setLevel(logging.CRITICAL)
logging.getLogger("pyjoulescope_driver").setLevel(logging.CRITICAL)
logging.getLogger("joulescope").setLevel(logging.CRITICAL)

# Fallback for virtual environments
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
        print("Error: The 'joulescope' or 'numpy' package was not found.", file=sys.stderr)
        print("Install it with: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

# Silence the Joulescope C backend at the driver level (OFF)
try:
    from joulescope.v1.driver import DriverWrapper
    _wrapper = DriverWrapper()
    if hasattr(_wrapper, "driver") and _wrapper.driver is not None:
        _wrapper.driver.log_level = "OFF"
except Exception:
    pass

from .. import (
    load_config,
    get_model_configs,
    train_all_models,
    export_model_entry_to_main,
    train_and_export,
)
from ..data import (
    list_available_datasets,
    print_available_datasets_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_esp_idf_export() -> str:
    """
    Find the ESP-IDF 'export.sh' script on the system.
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
    Build the ESP-IDF project and flash it to the ESP32.
    """
    if project_dir is None:
        project_dir = str(PROJECT_ROOT)

    export_sh = find_esp_idf_export()
    if not export_sh:
        print("[ESP-IDF] Warning: ESP-IDF 'export.sh' not found. Skipping flash.", file=sys.stderr)
        return False

    port_opt = f"-p {port} " if port else ""
    cmd_action = f"idf.py {port_opt}build flash" if build else f"idf.py {port_opt}flash"
    full_cmd = f". {export_sh} && cd {project_dir} && {cmd_action}"

    print(f"\n[ESP-IDF] Building and flashing firmware to ESP32 ({cmd_action})...")
    res = subprocess.run(["bash", "-c", full_cmd])
    if res.returncode != 0:
        print(f"\n[ESP-IDF] Failed to flash ESP32 (exit code {res.returncode}).", file=sys.stderr)
        return False

    print("[ESP-IDF] Firmware flashed successfully. Starting measurement...\n")
    return True


class StreamingMetricsAccumulator:
    """
    Accumulate electrical measurements (current, voltage, power) as a stream,
    without keeping unbounded raw-data arrays in memory (O(1) RAM usage).
    This avoids excessive memory use during long inference runs.
    """
    def __init__(self, sampling_freq: float, num_inferences: int = None, title: str = "MEASUREMENT RESULTS"):
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
            print("No measurement samples were collected.", file=sys.stderr)
            return {}

        dt = 1.0 / self.sampling_freq
        duration = self.num_samples * dt

        # Energy in joules (W·s), mJ, uJ, and mWh
        energy_j = self.sum_power * dt
        energy_mj = energy_j * 1000.0
        energy_uj = energy_j * 1e6
        energy_mwh = (energy_j / 3600.0) * 1000.0

        # Charge in coulombs (A·s), mAh, and uAh
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
            print(f"Inference duration: {duration * 1000:.2f} ms ({duration:.6f} s)")
        else:
            print(f"Inference duration: {duration:.4f} s")
        print(f"Sample rate:              {self.sampling_freq / 1e6:.2f} MSPS ({self.num_samples:,} samples)")
        print("-" * 65)
        print(f"Voltage (average): {avg_voltage:.4f} V   [min: {min_voltage:.4f} V, max: {max_voltage:.4f} V]")
        print(f"Current (average): {avg_current * 1000:.3f} mA ({avg_current:.6f} A)")
        print(f"         [min: {min_current * 1000:.3f} mA, max: {max_current * 1000:.3f} mA]")
        print(f"Power (average): {avg_power * 1000:.3f} mW ({avg_power:.6f} W)")
        print(f"         [min: {min_power * 1000:.3f} mW, max: {max_power * 1000:.3f} mW]")
        print("-" * 65)
        print(f"Electric charge:      {charge_mah:.4f} mAh ({charge_uah:.2f} µAh / {charge_c:.6f} C)")
        print(f"Total energy: {energy_j:.6f} J ({energy_mj:.3f} mJ / {energy_mwh:.6f} mWh)")

        energy_per_inf_mj = None
        energy_per_inf_uj = None
        time_per_inf_ms = None
        inferences_per_sec = None
        charge_per_inf_uah = None

        # Additional per-inference metrics, when available
        if self.num_inferences and self.num_inferences > 0:
            energy_per_inf_mj = energy_mj / self.num_inferences
            energy_per_inf_uj = energy_uj / self.num_inferences
            time_per_inf_ms = (duration * 1000.0) / self.num_inferences
            inferences_per_sec = self.num_inferences / duration if duration > 0 else 0.0
            charge_per_inf_uah = charge_uah / self.num_inferences

            print("-" * 65)
            print("          INFERENCE METRICS (TensorFlow Lite Micro)")
            print("-" * 65)
            print(f"Number of inferences:       {self.num_inferences}")
            print(f"Time per inference:       {time_per_inf_ms:.3f} ms ({inferences_per_sec:.1f} inferences/s)")
            print(f"Energy per inference:    {energy_per_inf_mj:.4f} mJ ({energy_per_inf_uj:.2f} µJ)")
            print(f"Charge per inference:     {charge_per_inf_uah:.4f} µAh")
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
    title: str = "MEASUREMENT RESULTS",
) -> dict:
    """
    Compute electrical metrics, print them to stdout, and return them as a dictionary.
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
    Read accuracy and inference metrics from ESP32 serial output.
    With 500 samples and long inference times, the metrics line may appear
    several minutes after the process starts.
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
    Print a summary table comparing all measured ANN and SNN models.
    """
    if not all_results:
        return

    print("\n" + "=" * 184)
    print("                                SUMMARY OF ALL MEASURED MODELS (ANN vs. SNN)")
    print("=" * 184)

    header = f"{'Model name':<20} | {'Type':<5} | {'Network architecture':<38} | {'Host accuracy (10k)':<14} | {'ESP32 accuracy (test)':<18} | {'Size':<9} | {'Time/inf':<10} | {'Energy/inf':<12} | {'Avg. power':<14} | {'Total spikes (avg/inf)':<24} | {'Total SynOps (avg/inf)'}"
    sub_hdr = f"{'':<20} | {'':<5} | {'':<38} | {'(%)':<14} | {'(% [correct])':<18} | {'(bytes)':<9} | {'(ms)':<10} | {'(µJ)':<12} | {'(mW)':<14} | {'(count)':<24} | {'(count)'}"
    print(header)
    print(sub_hdr)
    print("-" * 184)

    for res in all_results:
        name = res.get("name", "model")[:20]
        m_type = str(res.get("type", "ANN")).upper()[:5]
        structure = res.get("structure", "-")[:38]
        host_acc = f"{res['test_acc']:.2f} %" if res.get("test_acc") is not None else "N/A"

        esp32_acc_val = res.get("esp32_acc")
        if esp32_acc_val is None and res.get("deployment_supported", True):
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

            # Check whether the GPIO trigger was missed: near-zero energy but esp32_time_us is present
            gpio_missed = (energy_per_inf_uj is not None and energy_per_inf_uj < 0.1 and
                          esp32_time_us is not None and esp32_time_us > 0 and num_inf)

            if esp32_time_us is not None and num_inf:
                t_inf_val = esp32_time_us / 1000.0 / num_inf  # ms pro Inferenz
                t_inf = f"{t_inf_val:.2f} ms"
                # If the GPIO trigger was missed, use average power across the full measurement duration
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
            t_inf, e_inf, p_avg = "N/A", "N/A", "N/A"

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
    Measure energy in sync with a pattern-based trigger signal on IN1 (or IN0).

    Protocol sent by the ESP32:
      Start pattern: 3x HIGH (200 ms), separated by LOW (100 ms) -> measurement (HIGH)
      Measurement:   HIGH for the duration of inference (>= 100 ms)
      End pattern:   2x HIGH (200 ms), separated by LOW (100 ms) -> LOW (500 ms)
    
    The start and end patterns are distinguished by their pulse counts (3 vs. 2).

    A state-based parser recognizes patterns even when samples are missed or
    they cross chunk boundaries. The ESP32 runs continuously, allowing retries.
    """
    gpi_field = "gpi1" if pin.lower() in ("in1", "gpi1", "1") else "gpi0"
    # Read the microcontroller from config.yaml to display the correct board label
    try:
        import yaml as yaml_mod
        config_path = str(PROJECT_ROOT / "config.yaml")
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

    # Pattern timing tolerances (ms); allow some margin around the ESP32 pulse durations
    # Pattern pulse: HIGH ~200 ms (150-300 ms tolerance); LOW between pulses ~100 ms (50-200 ms tolerance)
    PULSE_HIGH_MIN_MS = 150.0
    PULSE_HIGH_MAX_MS = 300.0
    GAP_LOW_MIN_MS = 50.0
    GAP_LOW_MAX_MS = 200.0
    # The measurement itself must last at least 100 ms (inference always takes longer)
    MIN_MEASURE_MS = 100.0
    # After the start pattern: ~1100 ms LOW before measurement (100 ms gap + 1000 ms delay)
    # Wait for HIGH after >= 300 ms LOW to distinguish measurement from the start pattern
    START_TO_MEASURE_GAP_MIN_MS = 300.0

    # Compute the maximum allowed measurement duration based on sample count
    if max_duration is None:
        max_duration = max(60.0, (num_inferences or 50) * 2.0)

    # Limit the Joulescope driver buffer to 10 s to keep memory use low
    try:
        device.parameter_set("buffer_duration", 10)
    except Exception:
        pass

    sampling_freq = device.output_sampling_frequency
    samples_per_ms = sampling_freq / 1000.0

    print(f"Joulescope sample rate:   {sampling_freq / 1e6:.2f} MSPS")
    print(f"Waiting for start pattern on {channel_label}...")
    print(f"  Protocol: start pattern (3 pulses) -> measurement (HIGH) -> end pattern (2 pulses)")
    if timeout > 0:
        print(f"Trigger timeout:         {timeout:.1f} seconds (total)")
    if max_duration > 0:
        print(f"Maximum measurement duration: {max_duration:.1f} seconds (per cycle)")

    device.start()

    accumulator = StreamingMetricsAccumulator(
        sampling_freq=sampling_freq,
        num_inferences=num_inferences,
        title=f"MEASUREMENT RESULTS MNIST INFERENZ (Trigger {channel_label})",
    )
    # Second accumulator for the full duration, including wait time
    total_accumulator = StreamingMetricsAccumulator(
        sampling_freq=sampling_freq,
        num_inferences=num_inferences,
        title="GESAMTE MESSDAUER",
    )
    last_processed_id = None

    # State machine for pattern recognition
    # Start pattern: 3 pulses (distinct from the 2-pulse end pattern)
    # States:
    #   IDLE           - Wait for the first HIGH pulse (start pattern pulse 1)
    #   START_PULSE1   - First HIGH pulse of the start pattern
    #   START_GAP1     - LOW between start pattern pulses 1 and 2
    #   START_PULSE2   - Second HIGH pulse of the start pattern
    #   START_GAP2     - LOW between start pattern pulses 2 and 3
    #   START_PULSE3   - Third HIGH pulse of the start pattern
    #   START_WAIT     - LOW after the start pattern; wait for measurement (HIGH)
    #   MEASURING      - Measurement active (HIGH)
    #   END_GAP1       - LOW after measurement, before end pattern pulse 1
    #   END_PULSE1     - First HIGH pulse of the end pattern
    #   END_GAP2       - LOW between end pattern pulses 1 and 2
    #   END_PULSE2     - Second HIGH pulse of the end pattern (measurement ends after this)
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
    state_start_samples = 0  # Sample index where the current state began
    last_gpi_value = 0
    t_wait_start = time.time()
    t_measure_start = None
    t_last_progress = 0.0
    retry_count = 0
    pattern_start_idx = 0  # Sample index where the start pattern began (for accumulator slicing)

    def reset_for_retry():
        """Reset state and accumulators for a new attempt."""
        nonlocal state, last_gpi_value, t_measure_start, t_last_progress, accumulator
        state = STATE_IDLE
        last_gpi_value = 0
        t_measure_start = None
        t_last_progress = 0.0
        accumulator = StreamingMetricsAccumulator(
            sampling_freq=sampling_freq,
            num_inferences=num_inferences,
            title=f"MEASUREMENT RESULTS MNIST INFERENZ (Trigger {channel_label})",
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
                        f"Timeout: no valid start pattern received on {channel_label} within {timeout:.1f} s."
                    )
            else:
                elapsed = now - t_measure_start
                if elapsed - t_last_progress >= 5.0:
                    t_last_progress = elapsed
                    print(f"  [Measurement active] {elapsed:.1f} s elapsed ({accumulator.num_samples:,} samples collected)...", flush=True)

                if max_duration > 0 and elapsed > max_duration:
                    print(f"\n[Trigger] >>> Timeout: measurement exceeded {max_duration:.1f} s. <<<")
                    retry_count += 1
                    print(f"[Trigger] Attempt #{retry_count}; waiting for the next start pattern...")
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

            # Get the next data chunk
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

            # Always feed all data to the total accumulator
            total_accumulator.update(cur_chunk, vol_chunk)

            # State machine: process the chunk sample by sample
            chunk_start_global = last_processed_id - chunk_len
            
            # In MEASURING state, collect measurement sample indices from this chunk
            measure_start_in_chunk = None
            measure_end_in_chunk = None

            for i in range(chunk_len):
                gpi_val = gpi_chunk[i]
                global_idx = chunk_start_global + i
                samples_in_state = global_idx - state_start_samples

                if state == STATE_IDLE:
                    # In IDLE, any HIGH value begins pattern recognition.
                    # A rising-edge check (last_gpi_value == 0) could miss the start
                    # if the first chunk is already HIGH or the edge crosses chunk boundaries.
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
                            print(f"\n[Trigger] >>> Start pattern (3 pulses) detected. Waiting for measurement... <<<")
                            state = STATE_START_WAIT
                            state_start_samples = global_idx

                elif state == STATE_START_WAIT:
                    # After the start pattern, any HIGH value after a sufficient LOW interval
                    # starts measurement; no rising-edge check is used because an edge could be missed.
                    if gpi_val == 1:
                        wait_ms = samples_to_ms(samples_in_state)
                        if wait_ms < START_TO_MEASURE_GAP_MIN_MS:
                            state = STATE_IDLE
                        else:
                            print(f"[Trigger] >>> Measurement started (after {wait_ms:.0f} ms delay). <<<")
                            state = STATE_MEASURING
                            state_start_samples = global_idx
                            t_measure_start = time.time()
                            t_last_progress = 0.0
                            measure_start_in_chunk = i  # Measurement starts at this sample

                elif state == STATE_MEASURING:
                    if gpi_val == 0:
                        measure_end_in_chunk = i  # Measurement ends at this sample
                        measure_ms = samples_to_ms(samples_in_state)
                        if measure_ms < MIN_MEASURE_MS:
                            retry_count += 1
                            print(f"[Trigger] >>> Measurement too short ({measure_ms:.0f} ms); retrying... <<<")
                            reset_for_retry()
                            break
                        else:
                            print(f"[Trigger] >>> Measurement ended ({measure_ms:.0f} ms); waiting for end pattern... <<<")
                            state = STATE_END_GAP1
                            state_start_samples = global_idx

                elif state == STATE_END_GAP1:
                    if gpi_val == 1:
                        gap_ms = samples_to_ms(samples_in_state)
                        if gap_ms < GAP_LOW_MIN_MS or gap_ms > GAP_LOW_MAX_MS:
                            retry_count += 1
                            print(f"[Trigger] >>> End pattern not detected; retrying... <<<")
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
                            print(f"[Trigger] >>> Invalid end pattern; retrying... <<<")
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
                            print(f"[Trigger] >>> Invalid end pattern; retrying... <<<")
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
                            print(f"[Trigger] >>> Incomplete end pattern; retrying... <<<")
                            reset_for_retry()
                            break
                        else:
                            print(f"[Trigger] >>> End pattern detected. Measurement complete. <<<")
                            raise SystemExit(0)

                last_gpi_value = gpi_val

            # After processing each sample, feed measurement samples to the accumulator
            if measure_start_in_chunk is not None:
                if measure_end_in_chunk is not None:
                    # Measurement both started and ended in this chunk
                    accumulator.update(cur_chunk[measure_start_in_chunk:measure_end_in_chunk],
                                       vol_chunk[measure_start_in_chunk:measure_end_in_chunk])
                else:
                    # Measurement starts in this chunk and continues
                    accumulator.update(cur_chunk[measure_start_in_chunk:],
                                       vol_chunk[measure_start_in_chunk:])
            elif measure_end_in_chunk is not None:
                # Measurement started before this chunk and ends in this chunk
                accumulator.update(cur_chunk[:measure_end_in_chunk], vol_chunk[:measure_end_in_chunk])
            elif state == STATE_MEASURING:
                # Measurement started before this chunk and continues (no end in this chunk)
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
        print(f"\n[Trigger] Retried {retry_count} invalid cycle(s).")

    # Display GPIO-triggered measurement
    metrics = accumulator.compute_and_display()

    # Add average power over the full measurement duration
    if total_accumulator.num_samples > 0:
        total_dt = 1.0 / sampling_freq
        total_duration = total_accumulator.num_samples * total_dt
        total_avg_power = total_accumulator.sum_power / total_accumulator.num_samples
        metrics["avg_power_mw_total"] = total_avg_power * 1000.0  # mW
        metrics["duration_total_s"] = total_duration
        print(f"[Measurement] Average power (total duration): {total_avg_power * 1000:.2f} mW ({total_duration:.2f} s)")

    return metrics


def measure_duration(device, duration: float = 10.0, num_inferences: int = None) -> dict:
    """
    Measure energy consumption for a fixed duration in seconds.
    """
    buffer_duration = max(30, int(duration * 2))
    try:
        device.parameter_set("buffer_duration", buffer_duration)
    except Exception:
        pass

    print(f"Starting measurement for {duration:.1f} seconds...")
    data = device.read(contiguous_duration=duration)

    current = data[:, 0]
    voltage = data[:, 1]
    sampling_freq = device.output_sampling_frequency

    return display_results(
        current,
        voltage,
        sampling_freq,
        num_inferences=num_inferences,
        title=f"MEASUREMENT RESULTS ({duration:.1f}s Duration)",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train and export models, flash ESP32 firmware, and measure power with a Joulescope."
    )
    # Configuration file
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to the configuration file (YAML or JSON; default: config.yaml / config.json)",
    )

    # Training options
    parser.add_argument(
        "-D", "--dataset",
        type=str,
        default=None,
        help="Select a dataset (e.g. mnist, fashion_mnist, cifar10, speech_commands, har_time_series, eeg_data, ecg_data)",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and domains",
    )
    parser.add_argument(
        "--train",
        dest="train",
        action="store_true",
        default=True,
        help="Train and export PyTorch models before flashing (enabled by default)",
    )
    parser.add_argument(
        "--no-train",
        dest="train",
        action="store_false",
        help="Skip training and use existing exported model files",
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=None,
        help="Number of PyTorch training epochs (overrides the config file)",
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides the config file)",
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=None,
        dest="learning_rate",
        help="Training learning rate (overrides the config file)",
    )
    parser.add_argument(
        "--early-stopping",
        dest="early_stopping",
        action="store_true",
        default=None,
        help="Enable early stopping when accuracy stops improving",
    )
    parser.add_argument(
        "--no-early-stopping",
        dest="early_stopping",
        action="store_false",
        help="Disable early stopping",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Number of epochs without improvement before stopping",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=None,
        help="Minimum metric improvement required for early stopping",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default=None,
        help="Metric to monitor for early stopping (test_acc or test_loss)",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Number of test samples for ESP32 inference (overrides the config file)",
    )
    parser.add_argument(
        "--force-train", "--force-retrain",
        dest="force_train",
        action="store_true",
        default=False,
        help="Retrain all models even when matching checkpoints exist",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory for saving and loading model checkpoints (default: ./saved_models)",
    )

    # Post-training optimization options
    parser.add_argument(
        "--prune", "--prune-amount",
        type=float,
        default=None,
        dest="prune_amount",
        help="Enable post-training pruning and set the target fraction of removed weights (e.g. 0.3 for 30%% sparsity)",
    )
    parser.add_argument(
        "--prune-method",
        type=str,
        default=None,
        choices=["l1_unstructured", "random_unstructured", "structured", "threshold"],
        help="Pruning method: l1_unstructured (default), random_unstructured, structured, or threshold",
    )
    parser.add_argument(
        "--quantize", "--quantize-type",
        type=str,
        default=None,
        dest="quantize_type",
        choices=["int8", "int4", "fp16", "dynamic"],
        help="Enable post-training weight quantization (int8, int4, fp16, or dynamic)",
    )
    parser.add_argument(
        "--clustering", "--num-clusters",
        type=int,
        default=None,
        dest="num_clusters",
        help="Enable weight clustering with k centroids (e.g. 16 for a 4-bit codebook)",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=None,
        help="Number of fine-tuning epochs after pruning to recover accuracy",
    )

    # Measurement and trigger options
    parser.add_argument(
        "-p", "--pin",
        type=str,
        default=None,
        help="Joulescope trigger input pin: in1 (default) or in0 (overrides the config file)",
    )
    parser.add_argument(
        "-t", "--timeout",
        type=float,
        default=None,
        help="Timeout in seconds while waiting for the trigger (0 means unlimited; overrides the config file)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Maximum measurement duration in seconds (0 means unlimited; overrides the config file)",
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=None,
        help="Fixed-duration mode in seconds (overrides trigger mode; e.g. -d 10.0)",
    )

    # ESP32 flashing options
    parser.add_argument(
        "--flash",
        dest="flash",
        action="store_true",
        default=True,
        help="Build and flash the ESP32 firmware before measuring (enabled by default)",
    )
    parser.add_argument(
        "--no-flash",
        dest="flash",
        action="store_false",
        help="Disable automatic ESP32 flashing before measurement",
    )
    parser.add_argument(
        "-P", "--port",
        type=str,
        default=None,
        help="ESP32 serial port (e.g. /dev/ttyACM0 or /dev/ttyUSB0; default: auto-detect)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        default=False,
        help="Flash without running 'idf.py build' first",
    )
    parser.add_argument(
        "--wait",
        dest="wait_unplug",
        action="store_true",
        default=True,
        help="Wait for Enter after flashing so the USB cable can be disconnected (enabled by default)",
    )
    parser.add_argument(
        "--no-wait",
        dest="wait_unplug",
        action="store_false",
        help="Do not wait for a key press after flashing",
    )

    args = parser.parse_args()

    if args.list_datasets:
        print_available_datasets_table()
        sys.exit(0)

    # Load configuration
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
    # PHASE 1: Train all models
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
        print("No models available for measurement.", file=sys.stderr)
        sys.exit(1)

    total_models = len(trained_models)

    # -------------------------------------------------------------
    # PHASE 2: Flash and measure each model on ESP32
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f">>> PHASE 2: SEQUENTIAL FLASHING & JOULESCOPE MEASUREMENT ({total_models} model{'s' if total_models != 1 else ''}) <<<")
    print("=" * 70)

    devices = joulescope.scan()
    if not devices:
        print("No Joulescope device found. Check the USB connection.", file=sys.stderr)
        sys.exit(1)

    device = devices[0]
    if hasattr(device, "_driver") and hasattr(device._driver, "log_level"):
        try:
            device._driver.log_level = "OFF"
        except Exception:
            pass
    print(f"Connected to Joulescope: {device.device_serial_number or device}\n")

    all_results = []

    try:
        device.open()

        for idx, model_entry in enumerate(trained_models, start=1):
            model_name = model_entry["name"]
            model_samples = model_entry.get("num_test_samples", num_test_samples)

            if not model_entry.get("deployable", True):
                print(f"[ESP32/Joulescope] Model '{model_name}' is host-only; skipping export and device measurement.")
                all_results.append({
                    "name": model_name,
                    "type": model_entry.get("type", "ann"),
                    "structure": model_entry.get("structure", "-"),
                    "test_acc": model_entry.get("test_acc"),
                    "subset_acc": model_entry.get("subset_acc", model_entry.get("subset_test_acc")),
                    "subset_test_acc": model_entry.get("subset_test_acc", model_entry.get("subset_acc")),
                    "subset_correct": model_entry.get("subset_correct"),
                    "subset_total": model_entry.get("subset_total"),
                    "esp32_acc": None,
                    "esp32_correct": None,
                    "esp32_total": None,
                    "model_size_bytes": model_entry.get("model_size_bytes"),
                    "metrics": {},
                    "num_test_samples": model_samples,
                    "deployment_supported": False,
                })
                continue

            print("\n" + "#" * 70)
            print(f"### [Model {idx}/{total_models}]: '{model_name}' ({model_entry.get('structure', '')}) ###")
            print("#" * 70 + "\n")

            # 1. Write this model's C/C++ headers to main/
            if "tflite_bytes" in model_entry and "test_dataset" in model_entry:
                export_model_entry_to_main(model_entry, output_dir="main")

            # 2. Build and flash the ESP32 if needed
            flash_failed = False
            if args.flash and args.duration is None:
                flash_ok = flash_esp32(port=args.port, build=not args.skip_build)
                if not flash_ok:
                    print(f"[ESP-IDF] Warning: Failed to flash model '{model_name}'. Skipping measurement.", file=sys.stderr)
                    flash_failed = True
                else:
                    if args.wait_unplug:
                        print("\n" + "=" * 65)
                        print(f">>> ESP32 flashed successfully for model '{model_name}'! <<<")
                        print("You can now disconnect the USB cable from the ESP32 so that")
                        print("USB power does not affect the measurement.")
                        print("(Note: The ESP32 must remain powered by the Joulescope/3.3 V supply.)")
                        print("=" * 65)
                        try:
                            input(f"\nPress [Enter] to start the Joulescope measurement for '{model_name}'...")
                        except (KeyboardInterrupt, EOFError):
                            print("\nMeasurement cancelled by user.")
                            break

            if flash_failed:
                continue

            # 3. Run the Joulescope measurement
            metrics = {}
            try:
                if args.duration is not None and args.duration > 0:
                    metrics = measure_duration(device, duration=args.duration, num_inferences=model_samples)
                else:
                    metrics = measure_triggered(device, pin=trigger_pin, timeout=timeout, num_inferences=model_samples, max_duration=max_duration)
            except TimeoutError as te:
                print(f"\n[Measurement failed] {te}", file=sys.stderr)
            except KeyboardInterrupt:
                print("\nMeasurement cancelled by user.", file=sys.stderr)
                break
            except Exception as e:
                print(f"\n[Measurement failed] Unexpected error for model '{model_name}': {e}", file=sys.stderr)

            # 4. Read serial logs after measurement (USB must be reconnected)
            serial_metrics = {}
            if args.flash and args.duration is None:
                if args.wait_unplug:
                    print("\n" + "=" * 65)
                    print(f"Reconnect the USB cable to the ESP32 to")
                    print(f"read the serial logs.")
                    print("=" * 65)
                    try:
                        input("Press [Enter] once the USB cable is reconnected...")
                    except (KeyboardInterrupt, EOFError):
                        print("\nMeasurement cancelled by user.")
                        break
                serial_metrics = capture_esp32_serial_metrics(port=args.port, timeout=60.0)
                if serial_metrics and "esp32_acc" in serial_metrics:
                    print(f"[ESP32] Test accuracy read over UART: {serial_metrics['esp32_acc']:.2f}% ({serial_metrics.get('esp32_correct')}/{serial_metrics.get('esp32_total')} correct)")

            esp32_acc = model_entry.get("subset_acc", model_entry.get("subset_test_acc"))
            esp32_corr = model_entry.get("subset_correct")
            esp32_tot = model_entry.get("subset_total", model_samples)
            if serial_metrics and "esp32_acc" in serial_metrics:
                esp32_acc = serial_metrics["esp32_acc"]
                esp32_corr = serial_metrics.get("esp32_correct", esp32_corr)
                esp32_tot = serial_metrics.get("esp32_total", esp32_tot)

            # Merge ESP32 serial metrics into the Joulescope metrics
            if serial_metrics:
                metrics.update({k: v for k, v in serial_metrics.items() if k.startswith("esp32_")})

            if esp32_acc is not None:
                print(f"[ESP32 test accuracy] Model '{model_name}': {esp32_acc:.2f}% ({esp32_corr}/{esp32_tot} samples classified correctly)")

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
                "deployment_supported": model_entry.get("deployable", True),
            })

            # If more models follow and wait_unplug is enabled:
            if idx < total_models and args.flash and args.wait_unplug and args.duration is None:
                next_model_name = trained_models[idx]["name"]
                print("\n" + "-" * 65)
                print(f"Measurement for '{model_name}' complete.")
                print(f"Reconnect the USB cable to the ESP32 to")
                print(f"flash the next model ('{next_model_name}').")
                print("-" * 65)
                try:
                    input("Press [Enter] once the USB cable is reconnected...")
                except (KeyboardInterrupt, EOFError):
                    print("\nMeasurement sequence stopped by user.")
                    break

    except KeyboardInterrupt:
        print("\nMeasurement cancelled by user.", file=sys.stderr)
    except TimeoutError as te:
        print(f"\n{te}", file=sys.stderr)
    except Exception as e:
        print(f"\nMeasurement error: {e}", file=sys.stderr)
    finally:
        try:
            device.close()
            print("Joulescope disconnected.")
        except Exception:
            pass

    # -------------------------------------------------------------
    # PHASE 3: Print the comparison summary
    # -------------------------------------------------------------
    if all_results:
        print_comparison_summary(all_results)


if __name__ == "__main__":
    main()
