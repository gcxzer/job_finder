from __future__ import annotations

import unittest
from unittest.mock import patch

from src import docker_backend


class DockerBackendLifecycleTests(unittest.TestCase):
    def test_attached_container_is_not_removed_or_stopped_on_close(self) -> None:
        container = _FakeContainer(status="running")
        fake_docker = _FakeDockerModule(_FakeClient(container))

        with patch.object(docker_backend, "docker", fake_docker):
            backend = docker_backend.DockerBackend(container_id="existing-container", auto_remove=True)
            backend.close()

        self.assertFalse(container.removed)
        self.assertFalse(container.stopped)

    def test_owned_container_is_removed_on_close_when_auto_remove_enabled(self) -> None:
        container = _FakeContainer(status="running")
        fake_docker = _FakeDockerModule(_FakeClient(container))

        with patch.object(docker_backend, "docker", fake_docker):
            backend = docker_backend.DockerBackend(auto_remove=True)
            backend.close()

        self.assertTrue(container.removed)
        self.assertFalse(container.stopped)

    def test_ls_info_quotes_path(self) -> None:
        container = _FakeContainer(status="running")
        fake_docker = _FakeDockerModule(_FakeClient(container))

        with patch.object(docker_backend, "docker", fake_docker):
            backend = docker_backend.DockerBackend(auto_remove=False)
            backend.ls_info("/tmp/a; touch /tmp/pwned")
            backend.close()

        self.assertEqual(
            container.commands[-1],
            ["bash", "-c", "ls -la '/tmp/a; touch /tmp/pwned'"],
        )


class _FakeContainer:
    id = "fake-container-id"

    def __init__(self, status: str) -> None:
        self.status = status
        self.started = False
        self.removed = False
        self.stopped = False
        self.commands: list[list[str]] = []

    def start(self) -> None:
        self.started = True
        self.status = "running"

    def exec_run(self, cmd: list[str], workdir: str, demux: bool) -> tuple[int, bytes]:
        self.commands.append(cmd)
        return 0, b""

    def remove(self, force: bool = False) -> None:
        self.removed = force

    def stop(self) -> None:
        self.stopped = True


class _FakeContainers:
    def __init__(self, container: _FakeContainer) -> None:
        self._container = container

    def get(self, container_id: str) -> _FakeContainer:
        return self._container

    def run(self, *args: object, **kwargs: object) -> _FakeContainer:
        return self._container


class _FakeImages:
    def get(self, image: str) -> object:
        return object()


class _FakeClient:
    def __init__(self, container: _FakeContainer) -> None:
        self.containers = _FakeContainers(container)
        self.images = _FakeImages()


class _FakeDockerModule:
    def __init__(self, client: _FakeClient) -> None:
        self._client = client

    def from_env(self) -> _FakeClient:
        return self._client


if __name__ == "__main__":
    unittest.main()
