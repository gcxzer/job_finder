from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import preflight
from src.configs import CONFIG


class PreflightTests(unittest.TestCase):
    def test_preflight_pulls_missing_docker_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_client = _FakeDockerClient(image_exists=False)
            config = _test_config(Path(tmpdir), image="python:test-slim")

            with (
                patch.object(preflight, "docker", _FakeDockerModule(fake_client)),
                patch.dict(
                    os.environ,
                    {
                        "DOCKER_HOST": "unix:///tmp/docker.sock",
                        "DOCKER_CONFIG": str(Path(tmpdir) / ".docker"),
                    },
                ),
            ):
                result = preflight.run_preflight_checks(config)

        self.assertTrue(fake_client.images.pulled)
        self.assertIn("Docker daemon reachable: test-version", result.messages)
        self.assertIn("Docker image pulled: python:test-slim", result.messages)

    def test_preflight_fails_when_docker_daemon_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _test_config(Path(tmpdir))

            with (
                patch.object(preflight, "docker", _FailingDockerModule(RuntimeError("connection refused"))),
                patch.dict(os.environ, {"DOCKER_HOST": "unix:///tmp/docker.sock", "DOCKER_CONFIG": str(Path(tmpdir))}),
            ):
                with self.assertRaises(preflight.PreflightError) as context:
                    preflight.run_preflight_checks(config)

        self.assertIn("Docker daemon is not reachable", str(context.exception))
        self.assertIn("Start Docker Desktop", str(context.exception))

    def test_preflight_fails_when_configured_credential_helper_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            docker_config = Path(tmpdir) / ".docker"
            docker_config.mkdir()
            (docker_config / "config.json").write_text('{"credsStore": "missing-test-helper"}', encoding="utf-8")
            config = _test_config(Path(tmpdir))

            with (
                patch.object(preflight, "docker", _FakeDockerModule(_FakeDockerClient(image_exists=True))),
                patch.dict(
                    os.environ,
                    {
                        "DOCKER_HOST": "unix:///tmp/docker.sock",
                        "DOCKER_CONFIG": str(docker_config),
                    },
                ),
            ):
                with self.assertRaises(preflight.PreflightError) as context:
                    preflight.run_preflight_checks(config)

        self.assertIn("docker-credential-missing-test-helper", str(context.exception))
        self.assertIn("missing from PATH", str(context.exception))


def _test_config(root: Path, *, image: str = "python:3.12-slim"):
    return CONFIG.model_copy(
        update={
            "docker": CONFIG.docker.model_copy(update={"image": image, "container_id": None}),
            "workspace": CONFIG.workspace.model_copy(
                update={
                    "root_dir": root / "workspace",
                    "logs_dir": root / "runs" / "logs",
                }
            ),
        }
    )


class _FakeImageNotFound(Exception):
    pass


class _FakeImages:
    def __init__(self, *, image_exists: bool) -> None:
        self.image_exists = image_exists
        self.pulled = False

    def get(self, image: str) -> object:
        if not self.image_exists:
            raise _FakeImageNotFound(f"No such image: {image}")
        return object()

    def pull(self, image: str) -> object:
        self.pulled = True
        self.image_exists = True
        return object()


class _FakeDockerClient:
    def __init__(self, *, image_exists: bool) -> None:
        self.images = _FakeImages(image_exists=image_exists)

    def version(self) -> dict[str, str]:
        return {"Version": "test-version"}


class _FakeDockerModule:
    def __init__(self, client: _FakeDockerClient) -> None:
        self.client = client

    def from_env(self) -> _FakeDockerClient:
        return self.client


class _FailingDockerModule:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def from_env(self) -> _FakeDockerClient:
        raise self.error


if __name__ == "__main__":
    unittest.main()
