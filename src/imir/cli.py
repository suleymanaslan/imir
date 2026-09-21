"""Single-image and directory inference with per-output provenance."""

import argparse
import json
from pathlib import Path

from .images import IMAGE_SUFFIXES


def plan_outputs(source: Path, destination: Path, recursive=False, overwrite=False):
    source, destination = source.expanduser().resolve(), destination.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {source}")
        if destination.suffix.lower() != ".png":
            raise ValueError("For a single image, --output must be a .png file")
        jobs = [(source, destination)]
    elif source.is_dir():
        if destination == source or source in destination.parents:
            raise ValueError("Output directory must be outside the input directory")
        files = source.rglob("*") if recursive else source.glob("*")
        images = sorted(p for p in files if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        # Keep original extension in the name: x.jpg and x.png never collide.
        jobs = [(p, destination / p.relative_to(source).parent / f"{p.name}.png") for p in images]
    else:
        raise FileNotFoundError(f"Input does not exist: {source}")
    if not jobs:
        raise ValueError("No supported images found")
    for input_path, output_path in jobs:
        sidecar = output_path.with_suffix(".json")
        if output_path.resolve() == input_path.resolve():
            raise ValueError("Output cannot replace an input image")
        for target in (output_path, sidecar):
            if target.exists() and (not overwrite or target.is_dir()):
                raise FileExistsError(
                    f"Output exists: {target}; choose another path or --overwrite"
                )
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description="ImIR image-instructed restoration")
    parser.add_argument(
        "--checkpoint", required=True, help="Local bundle directory or Hub model ID"
    )
    parser.add_argument("--revision", help="Hub checkpoint commit or tag")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", help="Only for task-aware checkpoints")
    parser.add_argument(
        "--scale", type=float, help="Instruction residual scale; checkpoint default"
    )
    parser.add_argument("--steps", type=int, help="Sampling steps; checkpoint default")
    parser.add_argument("--seed", type=int, default=0, help="Same independent seed for every image")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--base-model-path", type=Path, help="Local snapshot of the configured backbone"
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-generation-size", action="store_true")
    args = parser.parse_args(argv)
    try:
        jobs = plan_outputs(args.input, args.output, args.recursive, args.overwrite)
        import torch

        from .pipeline import ImIRPipeline

        pipe = ImIRPipeline.from_pretrained(
            args.checkpoint,
            revision=args.revision,
            device=args.device,
            torch_dtype=getattr(torch, args.dtype),
            cpu_offload=args.cpu_offload,
            local_files_only=args.local_files_only,
            base_model_path=args.base_model_path,
        )
        for source, destination in jobs:
            result = pipe(
                source,
                task=args.task,
                instruction_scale=args.scale,
                num_inference_steps=args.steps,
                seed=args.seed,
                restore_original_size=not args.keep_generation_size,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            result.image.save(destination, format="PNG")
            result.metadata["input_file"] = str(source)
            destination.with_suffix(".json").write_text(
                json.dumps(result.metadata, indent=2) + "\n"
            )
            print(destination)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"imir: {error}\n")


if __name__ == "__main__":
    main()
