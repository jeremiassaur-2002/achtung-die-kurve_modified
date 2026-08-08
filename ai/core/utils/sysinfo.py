"""Hardware-Snapshot des Laufs - wird neben das Timing gelegt.

Ohne diese Angabe ist eine Timing-Tabelle wertlos: "world_model: 4h" heisst auf
einer RTX 4090 etwas voellig anderes als auf einer L4, und beim Vergleich zweier
Laeufe (v1_0 gegen v1_1, oder derselbe Lauf bei anderem Anbieter) will man
wissen, ob der Unterschied vom Code oder von der Maschine kommt.

Bewusst ohne Zusatz-Abhaengigkeit: torch liefert GPU-Name und VRAM, der Rest
kommt aus der Standardbibliothek. Faellt torch aus, ist das kein Fehler - dann
steht eben `cuda: false` drin.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess


def _nvidia_smi() -> str | None:
    """Ein einzelner nvidia-smi-Aufruf fuer den Treiber-Stand. `nvidia-smi` fehlt
    auf CPU-Maschinen komplett, deshalb erst pruefen statt Exception fangen."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip() or None


def collect() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cuda": False,
    }
    smi = _nvidia_smi()
    if smi:
        info["nvidia_smi"] = smi
    try:
        import torch
    except ImportError:
        return info
    info["torch"] = torch.__version__
    info["cuda"] = bool(torch.cuda.is_available())
    if info["cuda"]:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = props.name
        info["gpu_vram_gb"] = round(props.total_memory / 1024**3, 1)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
    return info


def summary_line() -> str:
    """Eine Zeile fuers Log beim Start - reicht, um beim Durchscrollen der
    Konsole sofort zu sehen, ob man versehentlich auf CPU trainiert."""
    i = collect()
    if i.get("cuda"):
        return f"[sysinfo] {i.get('gpu_name')} ({i.get('gpu_vram_gb')} GB) x{i.get('gpu_count')} | {i['cpu_count']} CPU | torch {i.get('torch')}"
    return f"[sysinfo] KEINE GPU sichtbar - Training laeuft auf CPU | {i['cpu_count']} CPU | torch {i.get('torch', 'n/a')}"
