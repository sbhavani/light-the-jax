import platform
import re
import subprocess
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Set

from pip._vendor.packaging.version import InvalidVersion, Version

# Set up logger
logger = logging.getLogger(__name__)


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
        elif not isinstance(other, CUDABackend):
            return NotImplemented

        return (self.major, self.minor) < (other.major, other.minor)


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

    # Get the supported CUDA versions from the dictionary
    supported_cuda_versions = list(minimum_driver_versions.keys())
    
    # Check if we need to detect installed CUDA version via nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=cuda_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        cuda_version_str = result.stdout.strip()
        if cuda_version_str:
            # Format is typically "12.7" for CUDA 12.7
            parts = cuda_version_str.split('.')
            if len(parts) == 2:
                installed_cuda_major = int(parts[0])
                installed_cuda_minor = int(parts[1])
                
                # Check if installed CUDA version is greater than all supported versions
                is_newer = True
                for cuda_version in supported_cuda_versions:
                    cuda_major = cuda_version.major
                    cuda_minor = cuda_version.minor
                    
                    if installed_cuda_major < cuda_major or (installed_cuda_major == cuda_major and installed_cuda_minor <= cuda_minor):
                        is_newer = False
                        break
                
                # If we have a newer CUDA version than all supported versions, use the latest supported version
                if is_newer and supported_cuda_versions:
                    # Sort versions to find the latest supported
                    latest_supported = max(supported_cuda_versions)
                    
                    # Check if the driver is compatible with this version
                    if driver_version >= minimum_driver_versions[latest_supported]:
                        logger.info(f"Detected CUDA {installed_cuda_major}.{installed_cuda_minor} which is newer than all supported versions. "
                              f"Using latest supported version: CUDA {latest_supported.major}.{latest_supported.minor}")
                        # Return only the latest supported version
                        return [CUDABackend(latest_supported.major, latest_supported.minor)]
    except (subprocess.SubprocessError, ValueError, IndexError):
        # If anything goes wrong with CUDA detection, fall back to driver version detection
        pass

    # Original logic - return all compatible backends based on driver version
    return [
        CUDABackend(cuda_version.major, cuda_version.minor)
        for cuda_version, minimum_driver_version in minimum_driver_versions.items()
        if driver_version >= minimum_driver_version
    ]


def detect_compatible_computation_backends() -> Set[ComputationBackend]:
    return {*_detect_compatible_cuda_backends(), CPUBackend()}
