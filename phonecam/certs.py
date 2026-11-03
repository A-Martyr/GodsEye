"""A self-signed certificate covering this machine's LAN addresses.

Why this file exists at all: a browser will not hand a page its camera unless
the page is a secure context, and a LAN address over plain http is not one. No
amount of care on the phone page changes that - without TLS, getUserMedia
simply refuses, and the phones cannot be cameras.

So the laptop serves https with a certificate it signs itself. The phone shows
one warning, which is accepted once per phone and then remembered. That is the
whole cost, and it buys a LAN path with no cloud round-trip in it.

Generated with the openssl binary rather than the `cryptography` package,
which is not installed in this project's environments. openssl ships with Git
for Windows, so it is already present on the demo machine.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from phonecam import settings

CERT = settings.CERT_DIR / "cert.pem"
KEY = settings.CERT_DIR / "key.pem"
STAMP = settings.CERT_DIR / "issued_for.json"


def _openssl() -> str | None:
    """A usable openssl, preferring one on PATH then Git for Windows'."""
    import shutil
    found = shutil.which("openssl")
    if found:
        return found
    for guess in (r"C:\Program Files\Git\usr\bin\openssl.exe",
                  r"C:\Program Files\Git\mingw64\bin\openssl.exe"):
        if Path(guess).exists():
            return guess
    return None


def lan_addresses() -> list[str]:
    """Every address a phone might reach this machine on."""
    try:
        from core import reach
        found = [ip for ip, _label in reach.local_addresses()]
    except Exception:
        found = []
    return found or ["127.0.0.1"]


def ensure(force: bool = False) -> tuple[Path, Path] | None:
    """(cert, key), generating them if missing or if this machine's IP moved.

    The IP list is stamped beside the certificate because a certificate issued
    for yesterday's DHCP lease fails on today's, and the browser error for that
    looks identical to every other TLS failure.
    """
    addresses = lan_addresses()
    if not force and CERT.exists() and KEY.exists() and STAMP.exists():
        try:
            if json.loads(STAMP.read_text()).get("addresses") == addresses:
                return CERT, KEY
        except (OSError, ValueError):
            pass

    tool = _openssl()
    if tool is None:
        print("[phonecam] openssl not found - cannot make a certificate. Serve over "
              "the Cloudflare tunnel instead (share.bat), or install Git for "
              "Windows.", file=sys.stderr)
        return None

    settings.CERT_DIR.mkdir(parents=True, exist_ok=True)
    san = ",".join(dict.fromkeys([f"IP:{ip}" for ip in addresses]
                                 + ["IP:127.0.0.1", "DNS:localhost"]))
    # Our own config, and OPENSSL_CONF pointed at it. Windows machines
    # routinely carry an OPENSSL_CONF left by some other installer - on this
    # one it names a PostgreSQL path that does not exist, and openssl then
    # refuses to start at all. Supplying the config removes that dependency
    # instead of hoping the machine's is sane.
    conf = settings.CERT_DIR / "openssl.cnf"
    conf.write_text(
        "[req]\ndistinguished_name = dn\nx509_extensions = v3\nprompt = no\n"
        "[dn]\nCN = GodsEye PhoneCam\nO = GodsEye\n"
        f"[v3]\nsubjectAltName = {san}\nbasicConstraints = CA:FALSE\n"
        "keyUsage = digitalSignature, keyEncipherment\n"
        "extendedKeyUsage = serverAuth\n", encoding="utf-8")

    env = dict(os.environ, OPENSSL_CONF=str(conf))
    cmd = [tool, "req", "-x509", "-newkey", "rsa:2048",
           "-keyout", str(KEY), "-out", str(CERT),
           "-days", str(settings.CERT_DAYS), "-nodes", "-config", str(conf)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90,
                              env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[phonecam] openssl failed: {exc}", file=sys.stderr)
        return None
    if proc.returncode != 0 or not CERT.exists():
        print(f"[phonecam] openssl failed:\n{proc.stderr[:400]}", file=sys.stderr)
        return None

    STAMP.write_text(json.dumps({"addresses": addresses}), encoding="utf-8")
    print(f"[phonecam] certificate issued for {', '.join(addresses)}",
          file=sys.stderr)
    return CERT, KEY


if __name__ == "__main__":
    out = ensure(force="--force" in sys.argv)
    print(out if out else "failed")
