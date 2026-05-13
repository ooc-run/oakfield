# Backends

`/backends` contains the execution abstraction used to run KernelIR
packages against bound fields.

## Backend Contract

The public contract is in `include/oakfield/backend.h`:

- initialize a `SimBackend`
- bind fields through a `KernelIR` package
- launch the package
- inspect `last_error`
- destroy the backend

Backends advertise feature bits so callers can reject unsupported KernelIR
packages before launch.

## Support Matrix

| Backend | Status | Notes |
| --- | --- | --- |
| CPU | Reference/supported | Built into `simcore` in the standalone CMake package. This is the correctness reference for tests and examples. |
| CUDA | Experimental/optional | Enable with `OAKFIELD_ENABLE_CUDA=ON` when the CUDA Toolkit, driver library, and NVRTC are available. Runtime requests return `SIM_RESULT_NOT_FOUND` in builds without CUDA support. |
| Metal | Experimental/optional | Enable with `OAKFIELD_ENABLE_METAL=ON` on Apple platforms with the Metal and Foundation frameworks. Runtime requests return `SIM_RESULT_NOT_FOUND` in builds without Metal support. |

## CMake Options

- `OAKFIELD_ENABLE_OPENMP`: optional CPU backend accelerator when OpenMP is available.
- `OAKFIELD_ENABLE_VDSP`: optional CPU backend accelerator on Apple platforms.
- `OAKFIELD_ENABLE_CUDA`: compiles the CUDA backend and links CUDA driver/NVRTC
  libraries when the CUDA Toolkit is available.
- `OAKFIELD_ENABLE_METAL`: compiles the Metal backend and links Metal/Foundation on
  Apple platforms.

## Contributor Notes

- Keep CPU behavior as the correctness reference.
- Gate optional backends behind clear CMake options and tests.
- Keep CUDA/Metal marked experimental until builds compile them and
  CTest covers at least initialization, fallback behavior, and a pointwise
  KernelIR smoke path.
- Document any KernelIR node or scalar-domain limitations in the backend README
  and generated API docs.
- Keep backend-private state in private headers or source files.
