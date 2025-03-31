import contextlib
import dataclasses
import enum
import functools
import itertools
import optparse
import os
import re
import sys
import unittest.mock
from typing import List, Set
from unittest import mock

import pip._internal.cli.cmdoptions
from pip._internal.index.collector import CollectedSources
from pip._internal.index.package_finder import CandidateEvaluator
from pip._internal.index.sources import build_source
from pip._internal.models.search_scope import SearchScope

import light_the_jax as ltj

from . import _cb as cb
from ._utils import apply_fn_patch


class Channel(enum.Enum):
    STABLE = enum.auto()
    TEST = enum.auto()
    NIGHTLY = enum.auto()

    @classmethod
    def from_str(cls, string):
        return cls[string.upper()]


JAX_DISTRIBUTIONS = {
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "orbax",
    "chex",
    "dm-haiku",
    "diffrax",
    "equinox",
    "brax",
    "jraph",
    "maxtext",
    "numpyro",
    "pax",
    "t5x",
}

THIRD_PARTY_PACKAGES = {
    "Jinja2",
    "MarkupSafe",
    "Pillow",
    "certifi",
    "charset-normalizer",
    "colorama",
    "filelock",
    "fsspec",
    "idna",
    "mpmath",
    "networkx",
    "numpy",
    "packaging",
    "requests",
    "sympy",
    "tqdm",
    "typing-extensions",
    "urllib3",
    "tensorflow",
    "scipy",
    "absl-py",
}


def patch(pip_main):
    @functools.wraps(pip_main)
    def wrapper(argv=None):
        if argv is None:
            argv = sys.argv[1:]

        with apply_patches(argv):
            return pip_main(argv)

    return wrapper


# adapted from https://stackoverflow.com/a/9307174
class PassThroughOptionParser(optparse.OptionParser):
    def __init__(self):
        super().__init__(add_help_option=False)

    def _process_args(self, largs, rargs, values):
        while rargs:
            try:
                super()._process_args(largs, rargs, values)
            except (optparse.BadOptionError, optparse.AmbiguousOptionError) as error:
                largs.append(error.opt_str)


@dataclasses.dataclass
class LttOptions:
    computation_backends: Set[cb.ComputationBackend] = dataclasses.field(
        default_factory=lambda: {cb.CPUBackend()}
    )
    channel: Channel = Channel.STABLE

    @staticmethod
    def computation_backend_parser_options():
        return [
            optparse.Option(
                "--jax-computation-backend",
                help=(
                    "Computation backend for compiled JAX distributions, "
                    "e.g. 'cu118', 'cu121', or 'cpu'. "
                    "Multiple computation backends can be passed as a comma-separated "
                    "list, e.g 'cu118,cu121'. "
                    "If not specified, the computation backend is detected from the "
                    "available hardware, preferring CUDA over CPU."
                ),
            ),
            optparse.Option(
                "--cpuonly",
                action="store_true",
                help=(
                    "Shortcut for '--jax-computation-backend=cpu'. "
                    "If '--computation-backend' is used simultaneously, "
                    "it takes precedence over '--cpuonly'."
                ),
            ),
        ]

    @staticmethod
    def channel_parser_option() -> optparse.Option:
        return optparse.Option(
            "--jax-channel",
            help=(
                "Channel to download JAX distributions from, e.g. 'stable' , "
                "'test', and 'nightly'. "
                "If not specified, defaults to 'stable' unless '--pre' is given in "
                "which case it defaults to 'test'."
            ),
        )

    @staticmethod
    def _parse(argv):
        parser = PassThroughOptionParser()

        for option in LttOptions.computation_backend_parser_options():
            parser.add_option(option)
        parser.add_option(LttOptions.channel_parser_option())
        parser.add_option("--pre", dest="pre", action="store_true")

        opts, _ = parser.parse_args(argv)
        return opts

    @classmethod
    def from_pip_argv(cls, argv: List[str]):
        if not argv or argv[0] != "install":
            return cls()

        opts = cls._parse(argv)

        if opts.jax_computation_backend is not None:
            cbs = {
                cb.ComputationBackend.from_str(string.strip())
                for string in opts.jax_computation_backend.split(",")
            }
        elif opts.cpuonly:
            cbs = {cb.CPUBackend()}
        elif "LTJ_JAX_COMPUTATION_BACKEND" in os.environ:
            cbs = {
                cb.ComputationBackend.from_str(string.strip())
                for string in os.environ["LTJ_JAX_COMPUTATION_BACKEND"].split(",")
            }
        else:
            cbs = cb.detect_compatible_computation_backends()

        if opts.jax_channel is not None:
            channel = Channel.from_str(opts.jax_channel)
        elif opts.pre:
            channel = Channel.TEST
        else:
            channel = Channel.STABLE

        return cls(cbs, channel)


@contextlib.contextmanager
def apply_patches(argv):
    options = LttOptions.from_pip_argv(argv)

    patches = [
        patch_cli_version(),
        patch_cli_options(),
        patch_link_collection_with_supply_chain_attack_mitigation(
            options.computation_backends, options.channel
        ),
        patch_candidate_selection(options.computation_backends),
    ]

    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)

        yield stack


@contextlib.contextmanager
def patch_cli_version():
    with apply_fn_patch(
        "pip",
        "_internal",
        "cli",
        "main_parser",
        "get_pip_version",
        postprocessing=lambda input, output: f"ltt {ltj.__version__} from {ltj.__path__[0]}\n{output}",
    ):
        yield


@contextlib.contextmanager
def patch_cli_options():
    def postprocessing(input, output):
        for option in LttOptions.computation_backend_parser_options():
            input.cmd_opts.add_option(option)

    index_group = pip._internal.cli.cmdoptions.index_group

    with apply_fn_patch(
        "pip",
        "_internal",
        "cli",
        "cmdoptions",
        "add_target_python_options",
        postprocessing=postprocessing,
    ):
        with unittest.mock.patch.dict(index_group):
            options = index_group["options"].copy()
            options.append(LttOptions.channel_parser_option)
            index_group["options"] = options
            yield


def get_index_urls(computation_backends, channel):
    urls = []
    
    # Add base JAX releases URL
    urls.append("https://storage.googleapis.com/jax-releases/jax_releases.html")
    
    # Get Python version for appropriate wheel selection
    py_version = f"{sys.version_info.major}{sys.version_info.minor}"
    
    # Add JAX URLs for NVIDIA GPUs
    for backend in sorted(computation_backends):
        if isinstance(backend, cb.CUDABackend):
            # JAX uses cuda version in the form cuda12 instead of cu12
            cuda_version = f"cuda{backend.major}{backend.minor}"
            
            # Add appropriate wheel URLs for the current Python version
            if channel == Channel.NIGHTLY:
                # Add nightly URLs if needed
                urls.append(f"https://storage.googleapis.com/jax-releases/nightly/{cuda_version}")
            else:
                # Add stable URLs
                urls.append(f"https://storage.googleapis.com/jax-releases/{cuda_version}")
    
    return urls


@contextlib.contextmanager
def patch_link_collection_with_supply_chain_attack_mitigation(
    computations_backends, channel
):
    def is_pinned(requirement):
        if requirement.req is None:
            return False

        return requirement.is_pinned

    @contextlib.contextmanager
    def context(input):
        with patch_link_collection(
            computations_backends,
            channel,
            {
                requirement.name
                for requirement in input.root_reqs
                if requirement.user_supplied and is_pinned(requirement)
            },
        ):
            yield

    with apply_fn_patch(
        "pip",
        "_internal",
        "resolution",
        "resolvelib",
        "resolver",
        "Resolver",
        "resolve",
        context=context,
    ):
        yield


@contextlib.contextmanager
def patch_link_collection(computation_backends, channel, user_supplied_pinned_packages):
    search_scope = SearchScope(
        find_links=[],
        index_urls=get_index_urls(computation_backends, channel),
        no_index=False,
    )

    @contextlib.contextmanager
    def context(input):
        if not (
            input.project_name in JAX_DISTRIBUTIONS
            or (
                input.project_name in THIRD_PARTY_PACKAGES
                and input.project_name not in user_supplied_pinned_packages
            )
        ):
            yield
            return

        with mock.patch.object(input.self, "search_scope", search_scope):
            yield

    def postprocessing(input, output):
        if input.project_name not in JAX_DISTRIBUTIONS:
            return output

        if channel != Channel.STABLE:
            return output

        # Some stable binaries are not hosted on the JAX indices. We check if this
        # is the case for the current distribution.
        for remote_file_source in output.index_urls:
            candidates = list(remote_file_source.page_candidates())

            # Cache the candidates, so `pip` doesn't has to retrieve them again later.
            remote_file_source.page_candidates = lambda: iter(candidates)

            # If there are any candidates on the JAX indices, we continue normally.
            if candidates:
                return output

        # In case the distribution is not present on the JAX indices, we fall back
        # to PyPI.
        _, pypi_file_source = build_source(
            SearchScope(
                find_links=[],
                index_urls=["https://pypi.org/simple"],
                no_index=False,
            ).get_index_urls_locations(input.project_name)[0],
            candidates_from_page=input.candidates_from_page,
            page_validator=input.self.session.is_secure_origin,
            expand_dir=False,
            cache_link_parsing=False,
        )

        return CollectedSources(find_links=[], index_urls=[pypi_file_source])

    with apply_fn_patch(
        "pip",
        "_internal",
        "index",
        "collector",
        "LinkCollector",
        "collect_sources",
        context=context,
        postprocessing=postprocessing,
    ):
        yield


@contextlib.contextmanager
def patch_candidate_selection(computation_backends):
    # Update pattern to recognize both JAX patterns
    computation_backend_link_pattern = re.compile(
        r"/(?P<computation_backend>(cpu|cu\d+|cuda\d+))/"
    )
    # JAX specific pattern for extract from wheel filename
    jax_cuda_pattern = re.compile(
        r"jaxlib-[\d.]+\+(?P<computation_backend>cuda\d+)-"
    )

    def extract_local_specifier(candidate):
        local = candidate.version.local

        # Make sure that local actually is a computation backend identifier
        if local is not None:
            try:
                cb.ComputationBackend.from_str(local)
            except ValueError:
                local = None

        if local is None:
            # Check for JAX package with CUDA in filename
            if candidate.name == "jaxlib":
                jax_match = jax_cuda_pattern.search(candidate.link.filename)
                if jax_match:
                    cuda_str = jax_match["computation_backend"]
                    # Convert cuda12 to cu12 format
                    if cuda_str.startswith("cuda"):
                        major_minor = cuda_str[4:]  # Extract "12" from "cuda12"
                        if len(major_minor) == 2:
                            major, minor = major_minor[0], major_minor[1]
                            local = f"cu{major}{minor}"
                            return local
            
            # Standard JAX pattern
            match = computation_backend_link_pattern.search(candidate.link.comes_from)
            local = match["computation_backend"] if match else "any"
            
            # Convert cuda12 to cu12 format if needed
            if local and local.startswith("cuda"):
                major_minor = local[4:]  # Extract "12" from "cuda12"
                if len(major_minor) == 2:
                    major, minor = major_minor[0], major_minor[1]
                    local = f"cu{major}{minor}"

        # Early JAX distributions used the "any" local specifier to indicate a
        # pure Python binary. This was changed to no local specifier later.
        # Setting this to "cpu" is technically not correct as it will exclude this
        # binary if a non-CPU backend is requested. Still, this is probably the
        # right thing to do, since the user requested a specific backend and
        # although this binary will work with it, it was not compiled against it.
        if local == "any":
            local = "cpu"

        return local

    def preprocessing(input):
        if not input.candidates:
            return

        candidates = iter(input.candidates)
        candidate = next(candidates)

        if candidate.name not in JAX_DISTRIBUTIONS:
            # At this stage all candidates have the same name. Thus, if the first is
            # not a JAX distribution, we don't need to check the rest and can
            # return without changes.
            return

        input.candidates = [
            candidate
            for candidate in itertools.chain([candidate], candidates)
            if extract_local_specifier(candidate) in computation_backends
        ]

    vanilla_sort_key = CandidateEvaluator._sort_key

    def patched_sort_key(candidate_evaluator, candidate):
        # At this stage all candidates have the same name. Thus, we don't need to
        # mirror the exact key structure that the vanilla sort keys have.
        return (
            vanilla_sort_key(candidate_evaluator, candidate)
            if candidate.name not in JAX_DISTRIBUTIONS
            else (
                cb.ComputationBackend.from_str(extract_local_specifier(candidate)),
                candidate.version.base_version,
            )
        )

    with apply_fn_patch(
        "pip",
        "_internal",
        "index",
        "package_finder",
        "CandidateEvaluator",
        "get_applicable_candidates",
        preprocessing=preprocessing,
    ):
        with unittest.mock.patch.object(
            CandidateEvaluator, "_sort_key", new=patched_sort_key
        ):
            yield
