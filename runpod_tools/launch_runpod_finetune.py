#!/usr/bin/env python3
"""Launch a Runpod pod, fine-tune Qwen3-1.5B with Unsloth, and sync artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import runpod


LOGGER = logging.getLogger("runpod_launch")


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def load_env_file(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def load_env_file(path: Path) -> Dict[str,str]

def redact_env(env: Dict[str, Any]) -> Dict[str, Any]:
    redacted = {}
    for key, value in env.items():
        if key.upper().endswith("TOKEN") or key.upper().endswith("KEY"):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = value
    return redacted


def scrub_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if key.upper().endswith("TOKEN") or key.upper().endswith("KEY") else scrub_secrets(val))
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [scrub_secrets(item) for item in value]
    if isinstance(value, str):
        if "HF_TOKEN=" in value:
            return "HF_TOKEN=<redacted>"
        if "PUBLIC_KEY=" in value:
            return "PUBLIC_KEY=<redacted>"
    return value


def resolve_env_value(key: str, env_file: Dict[str, str]) -> Optional[str]:
    return os.environ.get(key) or env_file.get(key)


def resolve_public_key(path: Optional[str]) -> Optional[str]:
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.extend(
        [
            Path.home() / ".ssh/id_ed25519.pub",
            Path.home() / ".ssh/id_rsa.pub",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text().strip()
    return None


def normalize_gpu_type_id(value: str) -> str:
    cleaned = value.strip()
    for prefix in ("nvidia ", "geforce ", "amd "):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned


def resolve_gpu_type_id(value: str) -> str:
    normalized = normalize_gpu_type_id(value)
    try:
        gpus = runpod.get_gpus()
    except Exception as exc:
        raise RuntimeError(f"Failed to query Runpod GPU list: {exc}") from exc

    if LOGGER.isEnabledFor(logging.DEBUG):
        gpu_names = [
            f"{gpu.get('id')} | {gpu.get('displayName')}" for gpu in gpus
        ]
        LOGGER.debug("Available GPUs: %s", gpu_names)

    for gpu in gpus:
        gpu_id = str(gpu.get("id", "")).strip()
        display = str(gpu.get("displayName", "")).strip()
        if normalized.lower() in {gpu_id.lower(), display.lower()}:
            return gpu_id

    raise ValueError(
        "No GPU found matching "
        f"'{value}'. Run `runpod.get_gpus()` to see valid GPU IDs."
    )


def resolve_create_fn():
    if hasattr(runpod, "create_pod"):
        return runpod.create_pod
    if hasattr(runpod, "pods") and hasattr(runpod.pods, "create"):
        return runpod.pods.create
    raise RuntimeError("Unsupported runpod SDK: missing create_pod.")


def adapt_payload(payload: Dict[str, Any], param_names: set[str]) -> Dict[str, Any]:
    payload = dict(payload)

    rename_map = {
        "image_name": "image",
        "gpu_type_id": "gpu_type",
        "container_disk_in_gb": "container_disk_gb",
        "volume_in_gb": "volume_gb",
        "public_ssh_key": "ssh_public_key",
    }

    for src, dst in rename_map.items():
        if src in payload and src not in param_names and dst in param_names:
            payload[dst] = payload.pop(src)

    if "public_ssh_key" in payload and "public_ssh_key" not in param_names:
        ssh_value = payload.pop("public_ssh_key")
        for alt in ("ssh_public_key", "ssh_key"):
            if alt in param_names:
                payload[alt] = ssh_value
                break
        else:
            LOGGER.warning("SDK does not accept SSH key; SSH access may be unavailable.")

    filtered = {k: v for k, v in payload.items() if k in param_names}
    dropped = set(payload) - set(filtered)
    if dropped:
        LOGGER.debug("Dropping unsupported create_pod args: %s", sorted(dropped))
    return filtered


def get_pod(pod_id: str) -> Dict[str, Any]:
    if hasattr(runpod, "get_pod"):
        return runpod.get_pod(pod_id)
    if hasattr(runpod, "pods") and hasattr(runpod.pods, "get"):
        return runpod.pods.get(pod_id)
    raise RuntimeError("Unsupported runpod SDK: missing get_pod.")


def is_pod_ready(pod: Dict[str, Any]) -> bool:
    runtime = pod.get("runtime") or {}
    status = runtime.get("status") or pod.get("status") or pod.get("desiredStatus")
    if status:
        LOGGER.debug("Pod status: %s", status)
    return str(status).upper() in {"RUNNING", "READY"}


def resolve_ssh_endpoint(pod: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    runtime = pod.get("runtime") or {}
    ssh_endpoint = runtime.get("sshEndpoint")
    if isinstance(ssh_endpoint, str) and ":" in ssh_endpoint:
        host, port = ssh_endpoint.split(":", 1)
        return host, int(port)

    host = runtime.get("publicIp") or runtime.get("podAddress") or runtime.get("ip")
    ports = runtime.get("ports") or pod.get("ports") or []
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Runtime ports payload: %s", ports)
    for entry in ports:
        if isinstance(entry, str):
            if entry.startswith("22"):
                return host, 22
            continue
        if isinstance(entry, dict):
            if entry.get("privatePort") == 22 or entry.get("type") == "ssh":
                return host or entry.get("ip"), int(entry.get("publicPort", 22))
    if host:
        return host, 22
    return None, None


def resolve_ssh_gateway_user(pod: Dict[str, Any]) -> Optional[str]:
    runtime = pod.get("runtime") or {}
    user = runtime.get("podHostId")
    if user:
        return str(user)
    machine = pod.get("machine") or {}
    host_id = machine.get("podHostId")
    if host_id:
        return str(host_id)
    return None


def run_cmd(cmd: list[str], debug: bool = False) -> None:
    if debug:
        LOGGER.debug("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def wait_for_ssh(host: str, port: int, user: str, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            run_cmd(
                [
                    "ssh",
                    "-p",
                    str(port),
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "ConnectTimeout=5",
                    f"{user}@{host}",
                    "echo",
                    "ssh-ready",
                ],
                debug=True,
            )
            return
        except subprocess.CalledProcessError:
            LOGGER.debug("SSH not ready yet; retrying in 10s.")
            time.sleep(10)
    raise TimeoutError(f"SSH not reachable at {host}:{port}")


def wait_for_ssh_any(
    direct: Optional[Tuple[str, int, str]],
    gateway: Optional[Tuple[str, int, str]],
    timeout_s: int = 600,
) -> Tuple[str, int, str]:
    deadline = time.time() + timeout_s
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        if direct:
            host, port, user = direct
            LOGGER.debug("Trying direct SSH attempt %d: %s@%s:%s", attempts, user, host, port)
            try:
                wait_for_ssh(host, port, user, timeout_s=10)
                return host, port, user
            except TimeoutError:
                LOGGER.debug("Direct SSH attempt failed.")
        if gateway:
            host, port, user = gateway
            LOGGER.debug("Trying gateway SSH attempt %d: %s@%s:%s", attempts, user, host, port)
            try:
                wait_for_ssh(host, port, user, timeout_s=10)
                return host, port, user
            except TimeoutError:
                LOGGER.debug("Gateway SSH attempt failed.")
        time.sleep(5)
    raise TimeoutError("SSH not reachable via direct or gateway endpoints.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Runpod fine-tuning for Qwen3-1.5B with Unsloth."
    )
    parser.add_argument("--pod-name", default="qwen3-1.7b-finetune", help="Pod name.")
    parser.add_argument("--gpu-type", default="RTX 5090", help="GPU type id or name.")
    parser.add_argument(
        "--image",
        default="runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04",
        help="Docker image for the pod.",
    )
    parser.add_argument("--cloud-type", default="COMMUNITY", help="Runpod cloud type.")
    parser.add_argument("--volume-gb", type=int, default=50, help="Pod volume in GB.")
    parser.add_argument(
        "--container-disk-gb",
        type=int,
        default=30,
        help="Container disk size in GB.",
    )
    parser.add_argument("--data", default="finetune-data.jsonl", help="Local dataset.")
    parser.add_argument(
        "--remote-base",
        default="/workspace/bound-datagen",
        help="Remote workspace path.",
    )
    parser.add_argument(
        "--local-output",
        default="finetuned_models/qwen3-1.7b-unsloth",
        help="Local output directory for artifacts.",
    )
    parser.add_argument("--ssh-user", default="root", help="SSH username.")
    parser.add_argument(
        "--ssh-mode",
        choices=["auto", "direct", "gateway"],
        default="auto",
        help="SSH connection mode (direct tcp or ssh.runpod.io gateway).",
    )
    parser.add_argument("--ssh-host", default=None, help="Override SSH host.")
    parser.add_argument("--ssh-port", type=int, default=None, help="Override SSH port.")
    parser.add_argument("--ssh-key", default=None, help="SSH public key path.")
    parser.add_argument("--timeout-min", type=int, default=20, help="Pod ready timeout.")
    parser.add_argument("--debug", action="store_true", help="Verbose logging.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions only.")
    args = parser.parse_args()

    configure_logging(args.debug)

    LOGGER.debug("Python: %s", os.sys.version)
    LOGGER.debug("Args: %s", args)

    env_file = load_env_file(Path(".env"))
    runpod_key = resolve_env_value("RUNPOD_API_KEY", env_file)
    hf_token = resolve_env_value("HF_TOKEN", env_file)

    if not runpod_key:
        raise RuntimeError("RUNPOD_API_KEY not found in environment or .env")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found in environment or .env")

    LOGGER.debug("RUNPOD_API_KEY present: %s", bool(runpod_key))
    LOGGER.debug("HF_TOKEN present: %s", bool(hf_token))

    runpod.api_key = runpod_key
    LOGGER.debug("Runpod SDK module: %s", runpod)
    LOGGER.debug("Runpod SDK attrs: create_pod=%s pods=%s", hasattr(runpod, "create_pod"), hasattr(runpod, "pods"))

    ssh_key = resolve_public_key(args.ssh_key)
    if not ssh_key:
        raise RuntimeError("SSH public key not found; provide --ssh-key")
    LOGGER.debug("SSH public key detected (length=%d).", len(ssh_key))

    gpu_type_id = resolve_gpu_type_id(args.gpu_type)
    payload = {
        "name": args.pod_name,
        "image_name": args.image,
        "gpu_type_id": gpu_type_id,
        "cloud_type": args.cloud_type,
        "container_disk_in_gb": args.container_disk_gb,
        "volume_in_gb": args.volume_gb,
        "env": {"HF_TOKEN": hf_token},
        "docker_args": "sleep infinity",
        "ports": "22/tcp",
        "public_ssh_key": ssh_key,
        "start_ssh": True,
    }

    LOGGER.info("Creating pod with GPU type: %s", gpu_type_id)
    LOGGER.debug(
        "Pod payload: %s",
        json.dumps(
            {**payload, "env": redact_env(payload.get("env", {}))}, indent=2
        ),
    )
    if args.dry_run:
        LOGGER.info("Dry run enabled. Payload:\n%s", json.dumps(payload, indent=2))
        return

    create_fn = resolve_create_fn()
    import inspect

    LOGGER.debug("create_pod fn: %s", create_fn)
    LOGGER.debug("create_pod signature: %s", inspect.signature(create_fn))
    param_names = set(getattr(create_fn, "__signature__", None).parameters) if hasattr(create_fn, "__signature__") else set()
    if not param_names:
        param_names = set(inspect.signature(create_fn).parameters)
    LOGGER.debug("create_pod param names: %s", sorted(param_names))
    adapted = adapt_payload(payload, param_names)
    LOGGER.debug("Adapted pod payload: %s", json.dumps(adapted, indent=2))
    pod = create_fn(**adapted)
    LOGGER.debug("Create pod response: %s", json.dumps(scrub_secrets(pod), indent=2))
    pod_id = pod.get("id") or pod.get("podId") or pod.get("pod_id")
    if not pod_id:
        raise RuntimeError(f"Could not determine pod id from response: {pod}")
    LOGGER.info("Pod created: %s", pod_id)

    timeout = time.time() + args.timeout_min * 60
    LOGGER.info("Waiting for pod readiness...")
    while time.time() < timeout:
        pod = get_pod(pod_id)
        LOGGER.debug("Pod status payload: %s", json.dumps(scrub_secrets(pod), indent=2))
        if is_pod_ready(pod):
            break
        LOGGER.debug("Pod not ready yet; sleeping 15s.")
        time.sleep(15)
    else:
        raise TimeoutError("Timed out waiting for pod to become ready.")

    remote_base = args.remote_base.rstrip("/")
    host = args.ssh_host
    port = args.ssh_port
    ssh_timeout = time.time() + 10 * 60
    while time.time() < ssh_timeout and (not host or not port):
        auto_host, auto_port = resolve_ssh_endpoint(pod)
        host = host or auto_host
        port = port or auto_port
        if host and port:
            break
        LOGGER.debug("SSH endpoint not available yet; sleeping 15s.")
        time.sleep(15)
        pod = get_pod(pod_id)
        LOGGER.debug(
            "Pod payload while waiting for SSH: %s",
            json.dumps(scrub_secrets(pod), indent=2),
        )
    if not host or not port:
        raise RuntimeError(
            "Unable to resolve SSH endpoint; pass --ssh-host/--ssh-port."
        )

    ssh_user = args.ssh_user
    direct = None
    gateway = None
    if args.ssh_mode != "gateway":
        direct = (host, port, ssh_user)
        LOGGER.info("Direct SSH candidate: %s@%s:%s", ssh_user, host, port)

    if args.ssh_mode in {"gateway", "auto"}:
        gateway_user = resolve_ssh_gateway_user(pod)
        if not gateway_user:
            LOGGER.warning(
                "Unable to resolve gateway SSH user from pod data; skipping gateway."
            )
        else:
            gateway = ("ssh.runpod.io", 22, gateway_user)
            LOGGER.info(
                "Gateway SSH candidate: %s@%s:%s", gateway_user, "ssh.runpod.io", 22
            )

    LOGGER.debug("Remote base: %s", remote_base)
    LOGGER.info("Waiting for SSH to accept connections (trying both endpoints)...")
    if args.ssh_mode == "direct":
        wait_for_ssh(host, port, ssh_user, timeout_s=600)
    elif args.ssh_mode == "gateway":
        host, port, ssh_user = gateway
        wait_for_ssh(host, port, ssh_user, timeout_s=600)
    else:
        host, port, ssh_user = wait_for_ssh_any(direct, gateway, timeout_s=600)

    rsync_target = f"{ssh_user}@{host}:{remote_base}"
    ssh_base = [
        "ssh",
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        f"{ssh_user}@{host}",
    ]

    run_cmd(ssh_base + ["mkdir", "-p", remote_base], debug=args.debug)
    run_cmd(
        [
            "rsync",
            "-avv",
            "--progress",
            "-e",
            f"ssh -p {port} -o StrictHostKeyChecking=no",
            "finetune-data.jsonl",
            "runpod_tools/",
            "datagen/",
            rsync_target + "/",
        ],
        debug=args.debug,
    )

    LOGGER.info("Starting remote training...")
    run_cmd(
        ssh_base + [f"cd {remote_base} && bash runpod_tools/setup_and_train.sh"],
        debug=args.debug,
    )

    local_output = Path(args.local_output)
    local_output.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "rsync",
            "-avv",
            "--progress",
            "-e",
            f"ssh -p {port} -o StrictHostKeyChecking=no",
            f"{ssh_user}@{host}:{remote_base}/output/qwen3-1.7b-unsloth/",
            str(local_output) + "/",
        ],
        debug=args.debug,
    )

    LOGGER.info("Artifacts synced to %s", local_output)


if __name__ == "__main__":
    main()
