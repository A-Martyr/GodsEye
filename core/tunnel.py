"""The public tunnel - the only route that reaches someone who is not here.

A Cloudflare "quick" tunnel: cloudflared dials out to trycloudflare.com, is
given a throwaway https hostname, and forwards whatever arrives there to our
local port. Two properties are what make it the right tool for a demo. It
needs no account and no configuration, and because the connection is outbound
it needs nothing opened on the firewall and nothing forwarded on the router -
the two things nobody can arrange in a venue they walked into an hour ago.

Two consequences to hold on to:

  - The hostname is new every run. Any link handed out before a restart is
    dead afterwards, so the link has to be re-shared, not remembered.

  - It publishes a whole origin, not a path. Everything the local port serves
    becomes reachable from the internet, which for this app would mean the
    entire ANPR API and its docs. api/main.py narrows that back down to the
    judge console; this module deliberately does not pretend the tunnel is
    doing any narrowing itself.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import config

# Quick tunnels are always https and always on this domain. Anchored so a
# stray URL in a log line cannot be mistaken for the hostname we were given.
_URL_RE = re.compile(r"https://[a-z0-9][-a-z0-9]*\.trycloudflare\.com")

_NO_WINDOW = {"creationflags": 0x08000000} if sys.platform == "win32" else {}


@dataclass
class Status:
    running: bool = False
    url: str = ""
    detail: str = ""


def available() -> str | None:
    """The cloudflared binary, or None if it is not installed."""
    return shutil.which(config.TUNNEL_BIN)


def current_url() -> str:
    """The public URL a running tunnel left behind, or "".

    Read from a file because the tunnel does not belong to either process that
    needs the answer. The API serves the console, the dashboard displays the
    link, and cloudflared is started by neither - it is a third process, so a
    file on disk is the only thing all three can see.
    """
    try:
        url = config.TUNNEL_URL_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return url if _URL_RE.fullmatch(url) else ""


def clear_url() -> None:
    try:
        config.TUNNEL_URL_FILE.unlink()
    except OSError:
        pass


def start(port: int | None = None, wait_s: float | None = None) -> Status:
    """Bring a tunnel up and return once it has a public hostname.

    cloudflared prints its progress, including the hostname, on **stderr** -
    stdout stays empty - so stderr is what gets read. Roughly five seconds
    pass between launch and the hostname appearing.

    It is worth knowing that cloudflared will happily hand back a URL for a
    local port with nothing listening on it: the tunnel is up, the origin
    behind it is simply broken. So the caller must not read "we got a URL" as
    "the demo is reachable" - the API has to be running too.
    """
    port = port or int(config.API_BASE.rsplit(":", 1)[-1].strip("/") or 8000)
    wait_s = config.TUNNEL_STARTUP_S if wait_s is None else wait_s

    binary = available()
    if not binary:
        return Status(detail=(
            f"{config.TUNNEL_BIN} is not installed or not on PATH. Install it from "
            "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/"
            "downloads/ - it is a single executable and needs no account."))

    existing = current_url()
    if existing and _alive():
        return Status(running=True, url=existing, detail="Already running.")

    clear_url()
    config.TUNNEL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = config.TUNNEL_LOG_FILE.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [binary, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=log, stderr=subprocess.STDOUT, text=True, **_NO_WINDOW)

    # Poll the log rather than reading the pipe: the pipe is being written to
    # the file, and a reader competing with it loses lines. Polling a file is
    # also what lets a tunnel started by tunnel.bat be picked up by the
    # dashboard, which never sees the process at all.
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return Status(detail=_explain_exit())
        try:
            found = _URL_RE.search(config.TUNNEL_LOG_FILE.read_text(
                encoding="utf-8", errors="replace"))
        except OSError:
            found = None
        if found:
            url = found.group(0)
            config.TUNNEL_URL_FILE.write_text(url, encoding="utf-8")
            return Status(running=True, url=url)
        time.sleep(0.4)

    proc.terminate()
    return Status(detail=(
        f"cloudflared did not produce a public address within {wait_s:.0f}s. "
        "That is what no internet connection looks like from here - a quick "
        "tunnel has to reach Cloudflare to be given a hostname."))


def _explain_exit() -> str:
    """Turn cloudflared's exit into something an operator can act on."""
    try:
        tail = config.TUNNEL_LOG_FILE.read_text(encoding="utf-8", errors="replace")[-800:]
    except OSError:
        tail = ""
    low = tail.lower()
    if "failed to request quick tunnel" in low or "dial tcp" in low or "lookup" in low:
        return ("cloudflared could not reach Cloudflare. Check this machine has "
                "working internet - a quick tunnel cannot be created offline.")
    return f"cloudflared exited before printing an address. Last output:\n{tail.strip()[-400:]}"


def _alive() -> bool:
    """Is a cloudflared actually running right now?

    Asked of the operating system rather than remembered in a variable,
    because the tunnel usually belongs to a different process - started by
    tunnel.bat or demo.bat - than the one asking.
    """
    if sys.platform != "win32":
        try:
            return bool(subprocess.run(["pgrep", "-f", "cloudflared"],
                                       capture_output=True, timeout=5).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            return False
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq cloudflared.exe", "/NH"],
                             capture_output=True, text=True, timeout=8, **_NO_WINDOW)
        return "cloudflared.exe" in out.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def status() -> Status:
    """What the tunnel is doing, asked fresh rather than remembered."""
    url, alive = current_url(), _alive()
    if alive and url:
        return Status(running=True, url=url)
    if alive:
        return Status(running=True, detail="cloudflared is running but has not "
                                           "published an address yet.")
    if url:
        # A stale file outlives the process it came from, and a dead URL shown
        # as live is exactly the failure this whole change exists to remove.
        clear_url()
    return Status()


def stop() -> bool:
    """Stop every cloudflared on this machine.

    By image name, not by port: a quick tunnel makes an outbound connection
    and listens on nothing, so stop.bat's approach of finding the process
    holding a port cannot see it. Left alone it keeps a public address
    pointing at this laptop long after the demo has finished.
    """
    clear_url()
    try:
        if sys.platform == "win32":
            done = subprocess.run(["taskkill", "/IM", "cloudflared.exe", "/F", "/T"],
                                  capture_output=True, text=True, timeout=15, **_NO_WINDOW)
        else:
            done = subprocess.run(["pkill", "-f", "cloudflared"],
                                  capture_output=True, text=True, timeout=15)
        return done.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _main() -> int:
    """share.bat's body: bring the tunnel up, print the link, hold it open.

    Blocks rather than returning, so the console window's lifetime is the
    link's lifetime. That is the mental model an operator can hold under
    pressure - the window is open, the link works - and it means a closed
    window does not silently leave the demo published.
    """
    from core import reach

    result = start()
    if not result.running:
        print(f"\n  {result.detail}\n")
        return 1

    link = f"{result.url}/j/{reach.token()}"
    print(f"\n  Public judge console:\n\n    {link}\n")
    print("  Press Ctrl+C to take it down.\n", flush=True)
    try:
        while _alive():
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        stop()
        print("\n  Tunnel stopped - that link is now dead.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
