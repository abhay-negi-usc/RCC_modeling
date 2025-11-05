#!/usr/bin/env python3
"""
Record ATI Ethernet F/T (Axia / Net F/T) data over UDP (RDT protocol).

Changes from previous version:
- Uses a CONFIG dict (no argparse)
- Auto-generates output filename based on timestamp (and IP), e.g.:
  ./logs/ati_192-168-10-133_2025-11-04_01-12-09.csv
"""

import csv
import os
import socket
import struct
import sys
import time
from datetime import datetime, timezone
from urllib.request import urlopen
import xml.etree.ElementTree as ET

# =========================
# User configuration
# =========================
CONFIG = {
    "ip": "192.168.10.133",        # Sensor IP
    "out_dir": "./FTS/data",           # Directory to write CSVs (created if missing)
    "file_prefix": "ati",          # Prefix for filename
    "tare": True,                  # Send software bias (tare) before streaming
    "rate": 0,                     # 0 = continuous (until Ctrl-C). Otherwise number of samples to request.
    "timeout": 2.0,                # Socket receive timeout (seconds)
    "single_block": True,          # Use single-block mode (one record per UDP packet)
    "utc_timestamps": True,        # Write ISO timestamps in UTC (Z). If False, uses local time.
}

# =========================
# RDT protocol constants
# =========================
RDT_PORT = 49152
RDT_HEADER = 0x1234
CMD_STOP = 0x0000
CMD_START_SINGLE = 0x0001
CMD_START_MULTI  = 0x0003
CMD_BIAS = 0x0042

# Requests: header (uint16), command (uint16), sample_count (uint32) [big-endian]
REQ_STRUCT = struct.Struct('>HHI')

# Records: rdt_seq (u32), ft_seq (u32), status (u32),
#          Fx,Fy,Fz,Tx,Ty,Tz (all int32 counts), big-endian
REC_STRUCT = struct.Struct('>IIIiiiiii')


def get_counts_per_units(ip):
    """
    Fetch Counts-per-Force (cpf) and Counts-per-Torque (cpt) from netftapi2.xml.
    Tries cfgcpf/cfgcpt then calcpf/calcpt.
    """
    url = f'http://{ip}/netftapi2.xml'
    with urlopen(url, timeout=5) as resp:
        xml = resp.read()
    root = ET.fromstring(xml)

    cpf = root.findtext('.//cfgcpf') or root.findtext('.//calcpf')
    cpt = root.findtext('.//cfgcpt') or root.findtext('.//calcpt')

    if cpf is None or cpt is None:
        raise RuntimeError("Counts-per-Force/Torque not found in netftapi2.xml (cfgcpf/cfgcpt or calcpf/calcpt).")

    return float(int(cpf)), float(int(cpt))


def send_command(sock, ip, cmd, sample_count=0):
    pkt = REQ_STRUCT.pack(RDT_HEADER, cmd, sample_count)
    sock.sendto(pkt, (ip, RDT_PORT))


def make_output_path(cfg):
    """
    Build an output CSV path using timestamp and IP:
    <out_dir>/<file_prefix>_<ip_underscored>_<YYYY-MM-DD_HH-MM-SS>.csv
    """
    os.makedirs(cfg["out_dir"], exist_ok=True)
    ip_safe = cfg["ip"].replace('.', '-')
    now = datetime.now(timezone.utc) if cfg.get("utc_timestamps", True) else datetime.now()
    stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    fname = f'{cfg["file_prefix"]}_{ip_safe}_{stamp}.csv'
    return os.path.join(cfg["out_dir"], fname)


def main(cfg):
    ip = cfg["ip"]

    # Determine start command mode
    start_cmd = CMD_START_SINGLE if cfg.get("single_block", True) else CMD_START_MULTI

    # Get scaling (Counts-per-Force/Torque)
    cpf = cpt = None
    try:
        cpf, cpt = get_counts_per_units(ip)
        print(f"[info] Scaling: Counts-per-Force={cpf:.0f}, Counts-per-Torque={cpt:.0f}")
    except Exception as e:
        print(f"[warn] Could not read netftapi2.xml for scaling: {e}")
        print("[warn] Falling back to raw counts only (no unit conversion).")

    # Output path
    out_path = make_output_path(cfg)
    print(f"[info] Writing to: {out_path}")

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(cfg["timeout"])

    # Optional tare
    if cfg["tare"]:
        print("[info] Sending software Bias (tare).")
        send_command(sock, ip, CMD_BIAS, 0)
        time.sleep(0.1)

    # Start streaming
    print("[info] Starting RDT stream. Press Ctrl-C to stop.")
    send_command(sock, ip, start_cmd, cfg["rate"])

    # CSV header
    fieldnames = [
        'iso_time','t_epoch',
        'rdt_seq','ft_seq','status',
        'Fx','Fy','Fz','Tx','Ty','Tz'
    ]

    tz = timezone.utc if cfg.get("utc_timestamps", True) else None
    last_rdt = None
    packets = 0
    t0 = time.time()

    try:
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                data, _ = sock.recvfrom(4096)
                now = time.time()

                # If we somehow get multi-record data, iterate in strides
                if len(data) >= REC_STRUCT.size * 2:
                    for off in range(0, len(data) - REC_STRUCT.size + 1, REC_STRUCT.size):
                        rec = REC_STRUCT.unpack_from(data, off)
                        _write_record(writer, rec, now, cpf, cpt, last_rdt, tz)
                        last_rdt = rec[0]
                        packets += 1
                else:
                    rec = REC_STRUCT.unpack(data)
                    _write_record(writer, rec, now, cpf, cpt, last_rdt, tz)
                    last_rdt = rec[0]
                    packets += 1

    except KeyboardInterrupt:
        pass
    except socket.timeout:
        print("[warn] Receive timeout; no packets received in the last period.")
    finally:
        try:
            send_command(sock, ip, CMD_STOP, 0)
        except Exception:
            pass

    dt = time.time() - t0
    print(f"[done] Wrote {packets} packets to {out_path} in {dt:.1f}s.")


def _write_record(writer, rec, t_now, cpf, cpt, last_rdt, tzinfo):
    rdt_seq, ft_seq, status, fx_c, fy_c, fz_c, tx_c, ty_c, tz_c = rec

    if last_rdt is not None and rdt_seq != (last_rdt + 1):
        missed = (rdt_seq - (last_rdt + 1)) & 0xFFFFFFFF
        if missed:
            print(f"[warn] Missed {missed} RDT record(s): last={last_rdt}, now={rdt_seq}")

    if cpf and cpt:
        fx, fy, fz = fx_c / cpf, fy_c / cpf, fz_c / cpf
        tx, ty, tzv = tx_c / cpt, ty_c / cpt, tz_c / cpt
    else:
        fx, fy, fz, tx, ty, tzv = fx_c, fy_c, fz_c, tx_c, ty_c, tz_c

    # Timestamp (ISO 8601)
    dt_obj = datetime.fromtimestamp(t_now, tz=tzinfo) if tzinfo else datetime.fromtimestamp(t_now)
    iso_str = dt_obj.isoformat().replace("+00:00", "Z") if tzinfo else dt_obj.isoformat()

    row = {
        'iso_time': iso_str,
        't_epoch': f"{t_now:.6f}",
        'rdt_seq': rdt_seq,
        'ft_seq' : ft_seq,
        'status' : status,
        'Fx': fx, 'Fy': fy, 'Fz': fz,
        'Tx': tx, 'Ty': ty, 'Tz': tzv
    }
    writer.writerow(row)


if __name__ == '__main__':
    try:
        main(CONFIG)
    except KeyboardInterrupt:
        sys.exit(0)
