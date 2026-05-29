from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from src.codex_oauth.auth import CodexAuthStore, CodexDeviceAuthClient


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    store = CodexAuthStore()
    client = CodexDeviceAuthClient()
    start = client.start()
    print(f"Open: {start.verification_uri}")
    print(f"Code: {start.user_code}")
    print("Waiting for browser approval...")

    while True:
        poll = client.poll(device_auth_id=start.device_auth_id, user_code=start.user_code)
        if poll.status == "connected" and poll.credentials is not None:
            store.write_credentials(poll.credentials)
            print(f"Connected. Credentials saved to {store.path}")
            return
        time.sleep(args.poll_interval or start.interval)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="codex-oauth-login",
        description="Connect Codex OAuth for job_finder.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0,
        help="Override the device-code polling interval in seconds.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
