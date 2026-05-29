from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


PUBLIC_HTTP_SCHEMES = {"http", "https"}
LOCAL_HOSTNAMES = {"localhost", "localhost.localdomain"}


def public_http_url_error(url: str, *, resolve_dns: bool = True) -> str:
    """Return an error string when a URL is unsafe for crawler fetches."""
    clean_url = str(url or "").strip()
    if not clean_url:
        return "URL is empty."

    parsed = urlparse(clean_url)
    if parsed.scheme.lower() not in PUBLIC_HTTP_SCHEMES:
        return "Only http and https URLs are allowed."
    if not parsed.hostname:
        return "URL must include a hostname."
    if parsed.username or parsed.password:
        return "URLs with embedded credentials are not allowed."

    hostname = parsed.hostname.strip("[]").rstrip(".").lower()
    if hostname in LOCAL_HOSTNAMES or hostname.endswith(".localhost"):
        return "Localhost URLs are not allowed."
    if _is_blocked_ip(hostname):
        return "Private, local, link-local, reserved, multicast, and unspecified IPs are not allowed."
    if resolve_dns:
        resolved_error = _resolved_host_error(hostname)
        if resolved_error:
            return resolved_error
    return ""


def is_public_http_url(url: str, *, resolve_dns: bool = True) -> bool:
    return not public_http_url_error(url, resolve_dns=resolve_dns)


def _resolved_host_error(hostname: str) -> str:
    try:
        addresses = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        return f"Hostname could not be resolved: {error}."

    for address in addresses:
        ip_value = address[4][0]
        if _is_blocked_ip(ip_value):
            return "Hostname resolves to a private, local, link-local, reserved, multicast, or unspecified IP."
    return ""


def _is_blocked_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )
