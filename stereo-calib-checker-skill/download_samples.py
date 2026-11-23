# -*- coding: utf-8 -*-
"""下载 OpenCV 官方样例双目标定数据（left01~14 / right01~14，9x6 内角点）。

多源自动回退：直连 GitHub 失败时依次尝试镜像加速源。
安全约束：仅允许 https + 固定白名单域名，解析并阻断私网/环回/链路本地地址，
重定向目标二次校验，防 SSRF / DNS rebinding。
"""
import argparse
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ALLOWED_HOSTS = {
    "raw.githubusercontent.com",
    "ghfast.top",
    "gh-proxy.com",
    "ghproxy.net",
}
SOURCES = [
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/{name}",
    "https://ghfast.top/https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/{name}",
    "https://gh-proxy.com/raw.githubusercontent.com/opencv/opencv/4.x/samples/data/{name}",
    "https://ghproxy.net/https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/{name}",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (stereo-calib-checker sample downloader)"}


def validate_url(url):
    """校验 URL：https + 白名单域名 + 解析结果全为公网地址。"""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname is None:
        raise ValueError(f"仅支持 https URL: {url}")
    if parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"非白名单域名: {parsed.hostname}")
    infos = socket.getaddrinfo(parsed.hostname, 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast):
            raise ValueError(f"解析到内网/保留地址，已阻断: {ip}")


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """允许重定向，但每个跳转目标都重新过白名单与私网校验。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: A003
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url, dst):
    validate_url(url)
    req = urllib.request.Request(url, headers=HEADERS)
    opener = urllib.request.build_opener(SafeRedirect)
    with opener.open(req, timeout=30) as resp:
        data = resp.read()
    if not data:
        raise ValueError("响应为空")
    dst.write_bytes(data)


def main():
    ap = argparse.ArgumentParser(description="下载 OpenCV 样例标定数据到 ./data/")
    ap.add_argument("--out", default="data", help="输出目录（默认 data）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for i in range(1, 15):
        for side in ("left", "right"):
            name = f"{side}{i:02d}.jpg"
            dst = out / name
            if dst.exists() and dst.stat().st_size > 0:
                print(f"跳过（已存在）: {name}")
                ok += 1
                continue
            done = False
            for src in SOURCES:
                try:
                    fetch(src.format(name=name), dst)
                    print(f"已下载: {name}")
                    ok += 1
                    done = True
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
            if not done:
                print(f"失败: {name} ({last})")
    print(f"\n完成：成功下载 {ok} 个文件，保存在 {out}/")


if __name__ == "__main__":
    main()
