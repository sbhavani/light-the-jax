import platform
import re
import subprocess
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Set

from pip._vendor.packaging.version import InvalidVersion, Version


class ComputationBackend(ABC):
    @property
    @abstractmethod
    def local_specifier(self) -> str:
        pass

    @abstractmethod
    def __lt__(self, other: Any) -> bool:
        pass

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ComputationBackend):
            return self.local_specifier == other.local_specifier
        elif isinstance(other, str):
            return self.local_specifier == other
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.local_specifier)

    def __repr__(self) -> str:
        return self.local_specifier

    @classmethod
    def from_str(cls, string: str) -> "ComputationBackend":
        parse_error = ValueError(f"Unable to parse {string} into a computation backend")
        string = string.strip().lower()
        if string == "cpu":
            return CPUBackend()
        elif string.startswith("cu"):
            match = re.match(r"^cu(da)?(?P<version>[\d.]+)$", string)
            if match is None:
                raise parse_error

            version = match.group("version")
            if "." in version:
                major, minor = version.split(".")
            else:
                major = version[:-1]
                minor = version[-1]

            return CUDABackend(int(major), int(minor))
        elif string.startswith("rocm"):
            match = re.match(r"^rocm(?P<version>[\d.]+)$", string)
            if match is None:
                raise parse_error

            parts = match["version"].split(".")
            if len(parts) not in {2, 3}:
                raise parse_error

            return ROCmBackend(*[int(part) for part in parts])
        else:
            raise parse_error


class CPUBackend(ComputationBackend):
    @property
    def local_specifier(self) -> str:
        return "cpu"

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, ComputationBackend):
            return NotImplemented

        return True


class CUDABackend(ComputationBackend):
    def __init__(self, major: int, minor: int) -> None:
        self.major = major
        self.minor = minor

    @property
    def local_specifier(self) -> str:
        return f"cu{self.major}{self.minor}"

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, CPUBackend):
            return False
        elif isinstance(other, ROCmBackend):
            raise TypeError("Refusing to order a CUDA and a ROCm computation backend.")
        elif not isinstance(other, CUDABackend):
            return NotImplemented

        return (self.major, self.minor) < (other.major, other.minor)


class ROCmBackend(ComputationBackend):
    def __init__(self, major, minor, patch=None):
        self.major = major
        self.minor = minor
        self.patch = patch

    @property
    def local_specifier(self) -> str:
        parts = [self.major, self.minor]
        if self.patch is not None:
            parts.append(self.patch)
        return f"rocm{'.'.join(str(part) for part in parts)}"

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, CPUBackend):
            return False
        elif isinstance(other, CUDABackend):
            raise TypeError("Refusing to order a ROCm and a CUDA computation backend.")
        elif not isinstance(other, ROCmBackend):
            return NotImplemented

        if (self.major, self.minor) < (other.major, other.minor):
            return True
        elif (self.major, self.minor) > (other.major, other.minor):
            return False

        if self.patch is not None and other.patch is None:
            return False
        elif self.patch is None and other.patch is not None:
            return True
        else:
            return self.patch < other.patch


def _detect_nvidia_driver_version() -> Optional[Version]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return Version(result.stdout.splitlines()[-1])
    except (FileNotFoundError, subprocess.CalledProcessError, InvalidVersion):
        return None


_MINIMUM_DRIVER_VERSIONS = {
    "Linux": {
        # JAX supports CUDA 12.x with newer drivers
        Version("12.3"): Version("530.30.02"),
        Version("12.2"): Version("525.60.13"),
        Version("12.1"): Version("525.60.13"),
        Version("12.0"): Version("525.60.13"),
        # JAX supports CUDA 11.8 with these drivers
        Version("11.8"): Version("450.80.02"),
    },
    "Windows": {
        # JAX supports CUDA 12.x with newer drivers
        Version("12.3"): Version("531.18"),
        Version("12.2"): Version("528.33"),
        Version("12.1"): Version("528.33"),
        Version("12.0"): Version("528.33"),
        # JAX supports CUDA 11.8 with these drivers
        Version("11.8"): Version("452.39"),
    },
}


def _detect_compatible_cuda_backends() -> List[CUDABackend]:
    driver_version = _detect_nvidia_driver_version()
    if not driver_version:
        return []

    minimum_driver_versions = _MINIMUM_DRIVER_VERSIONS.get(platform.system())
    if not minimum_driver_versions:
        return []

    return [
        CUDABackend(cuda_version.major, cuda_version.minor)
        for cuda_version, minimum_driver_version in minimum_driver_versions.items()
        if driver_version >= minimum_driver_version
    ]


def detect_compatible_computation_backends() -> Set[ComputationBackend]:
    return {*_detect_compatible_cuda_backends(), CPUBackend()}
