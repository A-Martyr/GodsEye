"""How an outsider reaches this demo - every route, honestly assessed.

There are exactly two ways a phone gets to a laptop: the two are on the same
network, or something on the public internet relays between them. Everything
here is one of those two, plus the truthful answer about whether it is
actually working.

The old code guessed one LAN address and then "checked" it by connecting to
that address from this same machine. That connection never leaves the host, so
it never meets the firewall the judge's phone would meet - it passed whenever
the socket was bound, which is to say always. A green tick that cannot go red
is worse than no tick, because the demo looks fine right up until someone
else's phone is holding it. So nothing in this module claims reachability it
has not established, and where it genuinely cannot know - a venue access point
that isolates its clients is invisible from in here - it says so.
"""
from __future__ import annotations

import ipaddress
import secrets
import socket
import subprocess
import sys
from dataclasses import dataclass, field

import config

# Windows console tools are spawned with no window; without this a .bat-started
# dashboard flashes a console for every probe.
_NO_WINDOW = {"creationflags": 0x08000000} if sys.platform == "win32" else {}


# --- the shared secret ---------------------------------------------------
def token() -> str:
    """The token that makes the public link safe to hand out.

    Created on first use and then reused, because a link that stops working
    when the API restarts is a link you cannot send anyone in advance. Both
    the API and the dashboard call this and must get the same answer, which is
    why it lives in a file rather than in either process.
    """
    path = config.JUDGE_TOKEN_FILE
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except OSError:
        pass
    fresh = secrets.token_urlsafe(9)          # ~72 bits, short enough for a URL
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fresh, encoding="utf-8")
    except OSError:
        pass                                   # in-memory for this process only
    return fresh


def rotate_token() -> str:
    """Burn the current token and issue a new one.

    The revoke button. Every link already handed out stops working, which is
    the point - a tunnel URL pasted into a group chat outlives the demo.
    """
    with_suppress = config.JUDGE_TOKEN_FILE
    try:
        with_suppress.unlink()
    except OSError:
        pass
    return token()


def is_public_host(host_header: str) -> bool:
    """Did this request arrive on a public address rather than the LAN?

    The Host header is the discriminator, and it is the right one: a phone in
    the room asks for 192.168.x.x, and a browser coming through the tunnel
    asks for something.trycloudflare.com. The client IP cannot be used instead
    because cloudflared connects to us over loopback, so every tunnelled
    request looks local no matter where it came from.
    """
    host = (host_header or "").strip().lower()
    if not host:
        return False
    if host.startswith("["):                   # bracketed IPv6 literal
        host = host[1:host.find("]")] if "]" in host else host[1:]
    elif host.count(":") == 1:
        host = host.rsplit(":", 1)[0]
    if not host or host in ("localhost", "127.0.0.1", "::1") or host.endswith(".local"):
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return True                            # a hostname, i.e. not the LAN
    return not (addr.is_private or addr.is_loopback or addr.is_link_local)


# --- local addresses -----------------------------------------------------
@dataclass
class Candidate:
    """One way in, and what we actually know about it."""

    kind: str                                  # "tunnel" | "lan" | "local"
    url: str
    label: str
    detail: str = ""
    works_off_lan: bool = False
    warnings: list[str] = field(default_factory=list)


def _api_port() -> int:
    tail = config.API_BASE.rsplit(":", 1)[-1].strip("/")
    return int(tail) if tail.isdigit() else 8000


def local_addresses() -> list[tuple[str, str]]:
    """Every usable IPv4 on this machine, as (address, interface name).

    All of them, not the one the old code guessed. The guess - open a UDP
    socket towards 8.8.8.8 and read back the local end - returns the address
    with the default route, which is the wrong one whenever the judge is not
    on that particular network: Ethernet plugged in while the phone is on
    Wi-Fi, a VPN adapter holding the route, or a hotspot whose clients live on
    a different subnet than the upstream link.

    169.254.x.x is dropped. Windows puts that on any adapter that asked for
    DHCP and got no answer - an unplugged Ethernet port, an idle Wi-Fi Direct
    adapter - so it is a sign of an interface that is not connected to
    anything, and offering it as a link wastes the one attempt a judge makes.
    """
    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    if sys.platform == "win32":
        # PowerShell knows which adapter each address belongs to; naming the
        # interface is what lets the operator tell "the Wi-Fi one" from "the
        # Ethernet one" while a judge is waiting.
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "Get-NetIPAddress -AddressFamily IPv4 | "
                 "Select-Object -ExpandProperty IPAddress -First 40 | ForEach-Object { $_ }; "
                 "'---'; "
                 "Get-NetIPAddress -AddressFamily IPv4 | "
                 "ForEach-Object { $_.IPAddress + '|' + $_.InterfaceAlias }"],
                capture_output=True, text=True, timeout=12, **_NO_WINDOW)
            body = out.stdout.split("---", 1)[-1]
            for line in body.splitlines():
                if "|" not in line:
                    continue
                ip, _, iface = line.strip().partition("|")
                if ip and ip not in seen and _usable(ip):
                    seen.add(ip)
                    found.append((ip, iface.strip() or "network"))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    if not found:                              # non-Windows, or PowerShell blocked
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ip = info[4][0]
                if ip not in seen and _usable(ip):
                    seen.add(ip)
                    found.append((ip, "network"))
        except OSError:
            pass

    # Keep the default-route address first: when several are live it is still
    # the most likely one to be the network the room is on.
    primary = _primary_address()
    found.sort(key=lambda pair: (pair[0] != primary, pair[0]))
    return found


def _usable(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (addr.is_loopback or addr.is_link_local or addr.is_multicast
                or addr.is_unspecified)


def _primary_address() -> str:
    """The address with the default route, or "" if there is none."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))             # no packet leaves; this only
        return s.getsockname()[0]              # selects the outbound interface
    except OSError:
        return ""
    finally:
        s.close()


# --- the firewall --------------------------------------------------------
@dataclass
class Firewall:
    """What Windows Firewall will do to an inbound connection on our port."""

    profile: str = "unknown"
    blocks_inbound: bool = False
    rule_present: bool = False
    checked: bool = False

    @property
    def verdict(self) -> str:
        """"open", "unverified" or "unknown" - never a guess dressed as a fact.

        There is no "blocked", and that is deliberate. Windows will also let a
        connection in through a rule attached to the *program* rather than the
        port, and this machine has exactly that: an allow rule for python.exe,
        scoped to the Public profile. So an absent port rule does not prove the
        judge cannot get in - it only proves we cannot promise they can.
        Enumerating program rules to find out costs minutes, which is far too
        slow to sit behind a dashboard that reruns every few seconds.

        Saying "unverified" where the truth is unverified is the whole point of
        this module. The old code's mistake was not that it was pessimistic or
        optimistic, but that it stated something it had not checked.
        """
        if not self.checked:
            return "unknown"
        if self.rule_present or not self.blocks_inbound:
            return "open"
        return "unverified"


RULE_NAME = "GodsEye demo"


def firewall_state(port: int | None = None) -> Firewall:
    """Whether an inbound connection would survive the firewall.

    Both queries are exact-match netsh lookups and return in about a tenth of
    a second. Enumerating every rule instead takes minutes on a normal
    machine, which is far too slow to run behind a dashboard that reruns
    itself every few seconds.

    The subtle case this exists to catch: a rule that allows python.exe on the
    Public profile does nothing the moment Windows decides the venue Wi-Fi is
    Private, which it does whenever someone answers yes to "make this PC
    discoverable". Same laptop, same code, same network - and the judge's
    phone stops getting through.
    """
    port = port or _api_port()
    fw = Firewall()
    if sys.platform != "win32":
        return fw
    try:
        cur = subprocess.run(["netsh", "advfirewall", "show", "currentprofile"],
                             capture_output=True, text=True, timeout=8, **_NO_WINDOW)
        text = cur.stdout
        for line in text.splitlines():
            low = line.lower()
            if "profile settings" in low:
                fw.profile = line.split()[0].strip()
            elif "firewall policy" in low:
                fw.blocks_inbound = "blockinbound" in low.replace(" ", "")
        fw.checked = cur.returncode == 0

        rule = subprocess.run(["netsh", "advfirewall", "firewall", "show", "rule",
                               f"name={RULE_NAME} (TCP {port})"],
                              capture_output=True, text=True, timeout=8, **_NO_WINDOW)
        fw.rule_present = rule.returncode == 0 and "No rules match" not in rule.stdout
    except (OSError, subprocess.SubprocessError):
        fw.checked = False
    return fw


def firewall_fix_command(ports: tuple[int, ...] = ()) -> str:
    """The command that opens the ports, to be run in an elevated shell.

    Not run for the operator: adding a firewall rule needs Administrator, and
    a dashboard that silently asks for elevation mid-demo is worse than one
    that prints the line to paste. All profiles are named explicitly so the
    rule survives Windows reclassifying the network.
    """
    ports = ports or (_api_port(), 8501)
    return " & ".join(
        f'netsh advfirewall firewall add rule name="{RULE_NAME} (TCP {p})" '
        f'dir=in action=allow protocol=TCP localport={p} profile=any'
        for p in ports)


# --- putting it together -------------------------------------------------
def candidates(tunnel_url: str | None = None) -> list[Candidate]:
    """Every link worth showing, best first.

    Ordered by how far each one reaches rather than by how nice it is: the
    tunnel first because it is the only one that works for someone who is not
    in the building, then the LAN addresses, then loopback - which is not a
    judge link at all, but is the honest last resort when the laptop itself
    has to be handed over.
    """
    out: list[Candidate] = []
    port = _api_port()
    tok = token()

    if tunnel_url:
        out.append(Candidate(
            kind="tunnel",
            url=f"{tunnel_url.rstrip('/')}/j/{tok}",
            label="Public link - works from any network",
            detail="Routed through Cloudflare. Send it to anyone, anywhere; they "
                   "do not need to be on your Wi-Fi.",
            works_off_lan=True,
            warnings=["The address changes every time the tunnel restarts."]))

    fw = firewall_state(port)
    for ip, iface in local_addresses():
        warns: list[str] = []
        if fw.verdict == "unverified":
            warns.append(f"Unverified: the {fw.profile} profile blocks inbound "
                         f"connections by default and there is no rule for this port. "
                         f"It may still be allowed for python itself - test it on a "
                         f"phone before you rely on it.")
        elif fw.verdict == "unknown":
            warns.append("Could not read the firewall state, so this is unverified.")
        warns.append("Only works if their device is on this same network, and some "
                     "venue Wi-Fi blocks devices from talking to each other.")
        out.append(Candidate(
            kind="lan",
            url=f"http://{ip}:{port}/j",
            label=f"Same network - {iface}",
            detail="No internet needed. Their phone must be on this network.",
            works_off_lan=False,
            warnings=warns))

    out.append(Candidate(
        kind="local",
        url=f"http://127.0.0.1:{port}/j",
        label="This machine only",
        detail="Open it on the demo laptop and hand the laptop over. Always works, "
               "needs no network at all.",
        works_off_lan=False))
    return out
