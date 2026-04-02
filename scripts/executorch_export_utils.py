from pathlib import Path


def resolve_output_path(default_filename: str, backend: str, output: str | None) -> str:
    if output:
        return output

    path = Path(default_filename)
    if backend == "coreml":
        return str(path.with_name(f"{path.stem}_coreml{path.suffix}"))
    return str(path)


def lower_to_executorch(exported_program, backend: str):
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        from executorch.exir import to_edge

        edge_manager = to_edge(exported_program)
        edge_manager = edge_manager.to_backend(XnnpackPartitioner())
        return edge_manager.to_executorch()

    if backend == "coreml":
        try:
            import coremltools as ct
            from executorch.backends.apple.coreml.compiler import CoreMLBackend
            from executorch.backends.apple.coreml.partition import CoreMLPartitioner
            from executorch.exir import to_edge_transform_and_lower
        except ImportError as exc:
            raise RuntimeError(
                "CoreML export requires the Apple CoreML backend dependencies. "
                "Install ExecuTorch's CoreML requirements on macOS before using "
                "--backend coreml."
            ) from exc

        return to_edge_transform_and_lower(
            exported_program,
            partitioner=[
                CoreMLPartitioner(
                    compile_specs=[
                        CoreMLBackend.generate_minimum_deployment_target_compile_spec(
                            ct.target.iOS16
                        ),
                        CoreMLBackend.generate_compute_unit_compile_spec(
                            ct.ComputeUnit.ALL
                        ),
                        CoreMLBackend.generate_compute_precision_compile_spec(
                            ct.precision.FLOAT16
                        ),
                    ]
                )
            ],
        ).to_executorch()

    raise ValueError(f"Unsupported backend: {backend}")


def write_program(et_program, output_path: str) -> None:
    with open(output_path, "wb") as f:
        if hasattr(et_program, "write_to_file"):
            et_program.write_to_file(f)
        else:
            f.write(et_program.buffer)
