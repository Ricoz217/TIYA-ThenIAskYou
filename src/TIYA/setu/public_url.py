from __future__ import annotations

import asyncio
import ipaddress
import re
from urllib.parse import urlparse

import aiohttp

from .remote import RemoteLoginUnavailable

DEFAULT_IPV4_ENDPOINTS = (
    "https://api.ipify.org",
    "https://ddns.oray.com/checkip",
    "https://ip.3322.net",
    "https://4.ipw.cn",
    "https://v4.yinghualuo.cn/bejson",
)
DEFAULT_IPV6_ENDPOINTS = (
    "https://api6.ipify.org",
    "https://6.ipw.cn",
)
_IP_CANDIDATE_PATTERN = re.compile(r"[0-9A-Fa-f:.]+")


def _format_public_base_url(address: str, port: int) -> str:
    ip = ipaddress.ip_address(address.strip())
    host = f"[{ip.compressed}]" if ip.version == 6 else ip.compressed
    return f"http://{host}:{port}"


def select_public_bind_host(
        public_url: str,
        configured_host: str,
        *,
        auto_discovered: bool,
) -> str:
    host = configured_host.strip() or "0.0.0.0"
    if not auto_discovered or host != "0.0.0.0":
        return host

    public_host = urlparse(public_url).hostname
    try:
        if public_host and ipaddress.ip_address(public_host).version == 6:
            return "::"
    except ValueError:
        pass
    return host


async def _fetch_public_ip(
        endpoint: str,
        timeout: float,
) -> str:
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
            trust_env=False,
    ) as session:
        async with session.get(
                endpoint,
                headers={"User-Agent": "TIYA-Pixiv-Login/1"},
        ) as response:
            response.raise_for_status()
            return await response.text()


def _extract_global_ip(payload: str, version: int) -> str | None:
    for candidate in _IP_CANDIDATE_PATTERN.findall(payload):
        candidate = candidate.strip(".:")
        if not candidate:
            continue
        try:
            ip = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if ip.version == version and ip.is_global:
            return ip.compressed
    return None


async def _probe_endpoint(
        endpoint: str,
        version: int,
        timeout: float,
) -> str:
    payload = await _fetch_public_ip(endpoint, timeout)
    address = _extract_global_ip(payload, version)
    if address is None:
        raise ValueError(f"响应中没有公网 IPv{version}")
    return address


async def _probe_named_endpoint(
        endpoint: str,
        version: int,
        timeout: float,
) -> tuple[str, str | None, str | None]:
    try:
        address = await _probe_endpoint(endpoint, version, timeout)
    except (
        aiohttp.ClientError,
        TimeoutError,
        OSError,
        ValueError,
    ) as exc:
        return endpoint, None, str(exc)
    return endpoint, address, None


async def _race_endpoints(
        endpoints: tuple[str, ...],
        version: int,
        timeout: float,
) -> tuple[str | None, list[str]]:
    tasks = {
        asyncio.create_task(
            _probe_named_endpoint(endpoint, version, timeout)
        )
        for endpoint in endpoints
        if isinstance(endpoint, str) and endpoint.startswith(("http://", "https://"))
    }
    errors: list[str] = []
    try:
        for task in asyncio.as_completed(tasks):
            endpoint, address, error = await task
            if address is None:
                errors.append(f"{endpoint}: {error}")
                continue
            return address, errors
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    return None, errors


async def discover_public_login_url(
        port: int,
        *,
        timeout: float = 5,
        ipv4_endpoints: tuple[str, ...] | list[str] | None = None,
        ipv6_endpoints: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Discover a direct public IP without using the Pixiv proxy."""
    if not 1 <= port <= 65535:
        raise RemoteLoginUnavailable("公网登录端口无效")

    ipv4_sources = tuple(ipv4_endpoints or DEFAULT_IPV4_ENDPOINTS)
    ipv6_sources = tuple(ipv6_endpoints or DEFAULT_IPV6_ENDPOINTS)
    address, errors = await _race_endpoints(
        ipv4_sources,
        version=4,
        timeout=timeout,
    )
    if address is not None:
        return _format_public_base_url(address, port)

    address, ipv6_errors = await _race_endpoints(
        ipv6_sources,
        version=6,
        timeout=timeout,
    )
    errors.extend(ipv6_errors)
    if address is not None:
        return _format_public_base_url(address, port)

    raise RemoteLoginUnavailable(
        "无法自动获取公网 IP: " + "; ".join(errors)
    )
