"""Real CUDA kernel/gradient smoke test. No model or dataset is loaded."""
from datetime import datetime
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import time
from zoneinfo import ZoneInfo

import torch

root = Path(__file__).resolve().parents[2]
state = json.loads((root / ".local/bert_crf/stage1a_state.json").read_text(encoding="utf-8"))
target = Path(state["evidence"]) / "gpu_smoke.json"
report = {
    "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
    "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(), "scope": "Tensor arithmetic, matmul and autograd only; no model training",
    "versions": {name: metadata.version(name) for name in
                 ("torch", "transformers", "tokenizers", "huggingface-hub", "numpy", "pytest")},
}
try:
    assert torch.cuda.is_available(), "CUDA unavailable"
    report.update(gpu_name=torch.cuda.get_device_name(0),
                  compute_capability=torch.cuda.get_device_capability(0),
                  compiled_arches=torch.cuda.get_arch_list(),
                  gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1729)
    report["seed"] = 1729
    t0 = time.perf_counter()
    a = torch.arange(64, device="cuda", dtype=torch.float32)
    actual = ((a + 2) * 3).sum()
    torch.cuda.synchronize()
    assert actual.item() == 6432.0
    cpu_x = torch.randn(128, 256, requires_grad=True)
    cpu_w = torch.randn(256, 64, requires_grad=True)
    gpu_x = cpu_x.detach().to("cuda").requires_grad_()
    gpu_w = cpu_w.detach().to("cuda").requires_grad_()
    cpu_loss = (cpu_x @ cpu_w).square().mean()
    gpu_loss = (gpu_x @ gpu_w).square().mean()
    cpu_loss.backward()
    gpu_loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(gpu_loss).item()
    assert torch.allclose(cpu_loss.detach(), gpu_loss.detach().cpu(), rtol=1e-4, atol=1e-5)
    assert torch.allclose(cpu_x.grad, gpu_x.grad.cpu(), rtol=1e-4, atol=1e-5)
    assert torch.allclose(cpu_w.grad, gpu_w.grad.cpu(), rtol=1e-4, atol=1e-5)
    report.update(status="PASS", tensor_sum=actual.item(), matmul_shape=[[128,256],[256,64]],
                  loss_cpu=cpu_loss.item(), loss_cuda=gpu_loss.item(),
                  x_grad_max_abs_difference=(cpu_x.grad-gpu_x.grad.cpu()).abs().max().item(),
                  w_grad_max_abs_difference=(cpu_w.grad-gpu_w.grad.cpu()).abs().max().item(),
                  compute_elapsed_seconds=time.perf_counter()-t0,
                  peak_memory_allocated_bytes=torch.cuda.max_memory_allocated())
except Exception as exc:
    report.update(status="FAIL", error_type=type(exc).__name__, error=str(exc))
    raise
finally:
    with target.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
