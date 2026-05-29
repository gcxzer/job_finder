from __future__ import annotations

import io
import shlex
import tarfile
import time
from typing import Optional

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

try:
    import docker
    from docker.errors import NotFound
except ImportError:
    docker = None
    NotFound = Exception


class DockerBackend(BaseSandbox):
    """Docker sandbox backend for DeepAgents.

    This follows the project reference implementation: start or attach to a
    local Docker container, execute commands inside it, and transfer files with
    Docker archive APIs.
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        container_id: Optional[str] = None,
        auto_remove: bool = True,
        cpu_quota: int = 50000,
        memory_limit: str = "512m",
        network_disabled: bool = False,
        working_dir: str = "/workspace",
        volumes: dict[str, dict[str, str]] | None = None,
    ) -> None:
        if docker is None:
            raise ImportError("docker package is not installed. Install it with `uv add docker`.")

        self.client = docker.from_env()
        self.image = image
        self.auto_remove = auto_remove
        self.working_dir = working_dir
        self.volumes = volumes or {}
        self._container = None
        self._owns_container = False

        try:
            if container_id:
                try:
                    self._container = self.client.containers.get(container_id)
                    if self._container.status != "running":
                        self._container.start()
                except NotFound as error:
                    raise RuntimeError(f"Container {container_id} not found.") from error
            else:
                try:
                    self.client.images.get(image)
                except NotFound:
                    print(f"Pulling image {image}...")
                    self.client.images.pull(image)

                self._container = self.client.containers.run(
                    image,
                    command="tail -f /dev/null",
                    detach=True,
                    tty=True,
                    cpu_quota=cpu_quota,
                    mem_limit=memory_limit,
                    network_disabled=network_disabled,
                    working_dir=working_dir,
                    volumes=self.volumes,
                )
                self._owns_container = True

            self.execute(f"mkdir -p {working_dir}", workdir="/")
        except Exception as error:
            raise RuntimeError(f"Failed to start/attach Docker container: {error}") from error

    @property
    def id(self) -> str:
        return self._container.id if self._container else "unknown"

    def execute(
        self,
        command: str,
        workdir: Optional[str] = None,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        if not self._container:
            return ExecuteResponse(output="Container not running", exit_code=1, truncated=False)
        if timeout is not None and timeout < 0:
            return ExecuteResponse(output=f"Invalid timeout: {timeout}", exit_code=1, truncated=False)

        try:
            execution_workdir = workdir if workdir is not None else self.working_dir
            exec_command = _timeout_command(command, timeout)
            exec_result = self._container.exec_run(
                cmd=["bash", "-c", exec_command],
                workdir=execution_workdir,
                demux=False,
            )
            exit_code, output = exec_result

            if not isinstance(exit_code, int):
                output_str = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)
                return ExecuteResponse(
                    output=f"Docker exec failed internally. Exit code: {exit_code}\nOutput: {output_str}",
                    exit_code=1,
                    truncated=False,
                )

            return ExecuteResponse(
                output=output.decode("utf-8", errors="replace"),
                exit_code=exit_code,
                truncated=False,
            )
        except Exception as error:
            return ExecuteResponse(
                output=f"Error executing command: {error}",
                exit_code=1,
                truncated=False,
            )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        if not self._container:
            return [FileUploadResponse(path=path, error="permission_denied") for path, _ in files]

        responses: list[FileUploadResponse] = []
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for path, content in files:
                arcname = path.lstrip("/") if path.startswith("/") else path
                info = tarfile.TarInfo(name=arcname)
                info.size = len(content)
                info.mtime = time.time()
                tar.addfile(info, io.BytesIO(content))
                responses.append(FileUploadResponse(path=path, error=None))

        tar_stream.seek(0)

        try:
            self._container.put_archive(path="/", data=tar_stream)
        except Exception:
            return [FileUploadResponse(path=path, error="permission_denied") for path, _ in files]

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        if not self._container:
            return [FileDownloadResponse(path=path, error="permission_denied") for path in paths]

        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                bits, _stat = self._container.get_archive(path)

                file_content = io.BytesIO()
                for chunk in bits:
                    file_content.write(chunk)
                file_content.seek(0)

                with tarfile.open(fileobj=file_content, mode="r") as tar:
                    member = tar.next()
                    if member is None:
                        responses.append(FileDownloadResponse(path=path, error="file_not_found"))
                        continue
                    if member.isdir():
                        responses.append(FileDownloadResponse(path=path, error="is_directory"))
                        continue

                    extracted = tar.extractfile(member)
                    if extracted:
                        responses.append(
                            FileDownloadResponse(path=path, content=extracted.read(), error=None)
                        )
                    else:
                        responses.append(FileDownloadResponse(path=path, error="file_not_found"))
            except NotFound:
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
            except Exception as error:
                error_msg = str(error).lower()
                normalized = "permission_denied" if "permission" in error_msg else "invalid_path"
                responses.append(FileDownloadResponse(path=path, content=None, error=normalized))

        return responses

    def ls_info(self, path: str) -> list[dict]:
        if not self._container:
            return []

        try:
            result = self.execute(f"ls -la {shlex.quote(path)}")
            if result.exit_code != 0:
                return []

            entries = []
            for line in result.output.strip().split("\n")[1:]:
                parts = line.split(maxsplit=8)
                if len(parts) < 9:
                    continue

                entries.append(
                    {
                        "permissions": parts[0],
                        "links": parts[1],
                        "owner": parts[2],
                        "group": parts[3],
                        "size": parts[4],
                        "date": f"{parts[5]} {parts[6]} {parts[7]}",
                        "name": parts[8],
                    }
                )

            return entries
        except Exception:
            return []

    def close(self) -> None:
        if self._container:
            try:
                if self._owns_container:
                    if self.auto_remove:
                        self._container.remove(force=True)
                    else:
                        self._container.stop()
            except Exception:
                pass
            self._container = None


def _timeout_command(command: str, timeout: int | None) -> str:
    if timeout is None or timeout == 0:
        return command
    return f"timeout {int(timeout)}s bash -lc {shlex.quote(command)}"
