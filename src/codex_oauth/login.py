from __future__ import annotations

import time

from codex_oauth.auth import CodexAuthStore, CodexDeviceAuthClient


def main() -> None:
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
        time.sleep(start.interval)


if __name__ == "__main__":
    main()
