#!/usr/bin/env python
"""Apply the lighteval vllm_model.py DP fix to whichever lighteval install
is importable by the current Python interpreter.

Background: stock lighteval unconditionally sets
distributed_executor_backend="ray" whenever data_parallel_size > 1, even for
tensor_parallel_size == 1. Each Ray-remote DP worker then tries to build its
own nested Ray executor for a single-GPU engine that needs no distributed
executor at all, which hangs/conflicts. The fix restricts that override to
the case that actually needs it: tensor_parallel_size > 1.

Idempotent: safe to run multiple times, and safe if the file is already
patched (detected via importlib, not guessed via version number).

Usage: run with the SAME interpreter that has lighteval installed, e.g.:
    /path/to/venv/bin/python patch_lighteval_dp.py
    conda run -n lighteval python patch_lighteval_dp.py
"""
import importlib.util
import sys

ORIGINAL = '''        if config.data_parallel_size > 1:
            self.model_args["distributed_executor_backend"] = "ray"
            self._batch_size = "auto"'''

PATCHED = '''        if config.data_parallel_size > 1:
            if config.tensor_parallel_size > 1:
                self.model_args["distributed_executor_backend"] = "ray"
            self._batch_size = "auto"'''


def main():
    spec = importlib.util.find_spec("lighteval.models.vllm.vllm_model")
    if spec is None or spec.origin is None:
        print("ERROR: lighteval.models.vllm.vllm_model not importable with "
              f"this interpreter ({sys.executable}). Is lighteval installed here?",
              file=sys.stderr)
        sys.exit(1)

    target = spec.origin
    print(f"Target file: {target}")
    content = open(target, "r").read()

    if PATCHED in content:
        print("Already patched. Nothing to do.")
        return

    if ORIGINAL not in content:
        print("ERROR: expected original snippet not found -- this lighteval "
              "version's vllm_model.py differs from the one this patch was "
              "written against (lighteval==0.12.0). Inspect it manually "
              "around 'distributed_executor_backend' before patching.",
              file=sys.stderr)
        sys.exit(1)

    backup = target + ".bak.pre_dp_patch"
    try:
        with open(backup, "x") as f:
            f.write(content)
        print(f"Backup written: {backup}")
    except FileExistsError:
        print(f"Backup already exists, leaving it as-is: {backup}")

    patched_content = content.replace(ORIGINAL, PATCHED)
    with open(target, "w") as f:
        f.write(patched_content)
    print("Patch applied.")


if __name__ == "__main__":
    main()
