# `light-the-jax`

[![BSD-3-Clause License](https://img.shields.io/github/license/pmeier/light-the-torch)](https://opensource.org/licenses/BSD-3-Clause)

`light-the-jax` is a small utility that wraps `pip` to ease the installation process
for JAX and related distributions like `jaxlib`, `flax`, `optax`, and more, as well
as third-party packages that depend on them. It auto-detects compatible CUDA versions
from the local setup and installs the correct JAX binaries without user
interference.

- [Why do I need it?](#why-do-i-need-it)
- [How do I install it?](#how-do-i-install-it)
- [How do I use it?](#how-do-i-use-it)
- [How does it work?](#how-does-it-work)
- [Is it safe?](#is-it-safe)
- [How do I contribute?](#how-do-i-contribute)

## Why do I need it?

JAX distributions like `jax` and `jaxlib` are fully `pip install`'able, but installing
JAX with GPU support requires additional steps:

1. You need to know what CUDA version is compatible with your NVIDIA driver
2. You need to find the correct JAX wheel URL for your specific CUDA version and Python version
3. Installation can be complex, especially for users new to ML frameworks or GPU computing

JAX hosts pre-built wheels with GPU support on Google Cloud Storage rather than PyPI. This means
you need to use special installation commands like:

```shell
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

While this works, it has several downsides:

1. You need to know what CUDA version (e.g., `cuda12`) is supported on your system
2. You need to remember the specific flags and URLs for installation
3. The installation process differs from standard Python packages

If you want a simple `pip install` experience for JAX with GPU support, `light-the-jax` is for you.

## Current Compatibility Matrix

The table below shows the current CUDA version compatibility for JAX and related libraries that can be installed using `light-the-jax`:

| JAX Version | CUDA Version | Compatible NVIDIA Driver | Compatible Libraries |
|-------------|--------------|--------------------------|----------------------|
| Latest      | CUDA 12.3    | >= 530.30.02 (Linux)     | jax, jaxlib, flax, optax, numpyro |
|             |              | >= 531.18 (Windows)      |                      |
| Latest      | CUDA 12.2    | >= 525.60.13 (Linux)     | jax, jaxlib, flax, optax, numpyro |
|             |              | >= 528.33 (Windows)      |                      |
| Latest      | CUDA 12.1    | >= 525.60.13 (Linux)     | jax, jaxlib, flax, optax, numpyro |
|             |              | >= 528.33 (Windows)      |                      |
| Latest      | CUDA 12.0    | >= 525.60.13 (Linux)     | jax, jaxlib, flax, optax, numpyro |
|             |              | >= 528.33 (Windows)      |                      |
| Latest      | CUDA 11.8    | >= 450.80.02 (Linux)     | jax, jaxlib, flax, optax, numpyro |
|             |              | >= 452.39 (Windows)      |                      |
| Latest      | CPU          | N/A                      | jax, jaxlib, flax, optax, numpyro |

Notes:
- The latest JAX wheels have CUDA 12.3 support and are backward compatible with CUDA 12.1+
- For older CUDA versions, corresponding JAX wheels are used
- JAX no longer supports GPUs with Compute Capability < 5.2. Maxwell or newer is required and Turing or newer is recommended.

## How do I install it?

Since `light-the-jax` is not yet published on PyPI, you can install it directly from GitHub:

```shell
pip install git+https://github.com/sbhavani/light-the-jax.git
```

Since it depends on `pip` and it might be upgraded during installation,
[Windows users](https://pip.pypa.io/en/stable/installation/#upgrading-pip) should
install it with:

```shell
py -m pip install git+https://github.com/sbhavani/light-the-jax.git
```

Once published to PyPI, installation will be as simple as:

```shell
pip install light-the-jax
```

## How do I use it?

After `light-the-jax` is installed you can use its CLI interface `ltj` as a drop-in
replacement for `pip`:

```shell
ltj install jax jaxlib
```

You can also install JAX ecosystem libraries:

```shell
ltj install jax jaxlib flax optax
```

In fact, `ltj` is `pip` with a few added options:

- By default, `ltj` uses the local NVIDIA driver version to select the correct binary
  for you. If your CUDA version is newer than all supported JAX versions (e.g., CUDA 12.7),
  it will automatically use the latest supported version (e.g., CUDA 12.3). 
  
  You can also pass the `--jax-computation-backend` option to manually specify
  the computation backend you want to use:

  ```shell
  ltj install --jax-computation-backend=cu121 jax jaxlib
  ```

  For CPU-only installations, `--cpuonly` is available as a shorthand for 
  `--jax-computation-backend=cpu`.

  ```shell
  ltj install --cpuonly jax jaxlib
  ```

  In addition, the computation backend can also be set through the
  `LTJ_JAX_COMPUTATION_BACKEND` environment variable. It will only be honored if
  no CLI option for the computation backend is specified.

- By default, `ltj` installs stable JAX binaries. To install nightly builds, pass 
  the `--jax-channel` option:

  ```shell
  ltj install --jax-channel=nightly jax jaxlib
  ```

  If `--jax-channel` is not passed, using `pip`'s builtin `--pre` option implies
  `--jax-channel=test`.

Of course, you are not limited to installing only JAX distributions. The tool also works
when installing packages that depend on JAX:

```shell
ltj install --jax-computation-backend=cpu numpyro
```

## How does it work?

The authors of `pip` **do not condone** the use of `pip` internals as they might break
without warning. As a result, `pip` has no capability for plugins to hook into
specific tasks.

`light-the-jax` works by monkey-patching `pip` internals at runtime:

- While searching for a download link for a JAX distribution, `light-the-jax`
  replaces the default search index with the official JAX storage bucket URL. This is
  equivalent to calling `pip install` with the `--find-links` option only for JAX
  distributions.
- While evaluating possible installation candidates, `light-the-jax` culls
  binaries incompatible with the hardware.

## Is it safe?

JAX is developed and maintained by Google and NVIDIA, and its wheels are hosted on Google Cloud Storage,
which provides a secure distribution mechanism. `light-the-jax` does not modify this security
model - it simply automates the process of finding and installing the correct wheels for your
system.

The tool follows the same security practices as the original `light-the-torch`:
- Third-party dependencies are pulled from PyPI only if they are specifically requested and pinned
- The regular JAX installation channels are used for JAX packages

## How do I contribute?

All contributions are appreciated, whether code or not. 