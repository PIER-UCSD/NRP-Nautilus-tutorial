# Nautilus for the Shi Lab — a practical tutorial

How to run GPU experiments on [NRP Nautilus](https://nrp.ai) using our lab
namespace, written from real runs (profiling + RL training, Aug 2026). Every
command here was actually used; every failure in §8 actually happened.

**What Nautilus is:** a national Kubernetes cluster with free shared GPUs.
You don't SSH into machines — you describe *what to run* in a YAML file,
submit it with `kubectl`, and the cluster finds a machine for you.

**Our namespace:** `shi-group-ece-advanced`.
**Quota reality:** A100 / H100 / H200 / GH200 = **0** (hard). What we can use
is the free pool: RTX 2080 Ti, T4, A10, RTX 3090/4090, A40, RTX A6000, L40,
L40S — up to 48 GB. Fine for ≤20B models and all profiling; not for 70B.

---

## Step 0 — Get access (once)

1. Create an account at [portal.nrp-nautilus.io](https://portal.nrp-nautilus.io)
   (institutional login: UCSD).
2. Ask the namespace admin (Yuanyuan or the current student admin) to add you
   to `shi-group-ece-advanced`.
3. Install `kubectl` (`brew install kubectl` on macOS).
4. Download your config from the portal ("Get Config") and place it at
   `~/.kube/config`.
5. Verify:

```bash
kubectl config current-context     # should print: nautilus
kubectl get pods                   # empty list is fine; an error is not
```

## Step 1 — The five concepts you actually need

| Concept | What it is | When you use it |
|---|---|---|
| **Pod** | one running container | interactive work, debugging |
| **Job** | a pod that runs to completion (with retries) | reproducible batch runs |
| **emptyDir** | scratch disk that **dies with the pod** | temp files, code you `kubectl cp` in |
| **PVC** | persistent disk that **survives pods** | checkpoints, HF cache, datasets |
| **Secret** | encrypted key-value in the namespace | HF_TOKEN, WANDB_API_KEY |

Rules of the namespace (Nautilus policy, they enforce it):
- `requests` must equal `limits` for CPU/memory (ratio ≤ 1.2).
- Bare interactive pods are reaped after ~6 h of no GPU use. Jobs can run longer.
- Always delete your pods/jobs when done — GPUs are shared.

## Step 2 — The two ways to launch work

### A. Interactive pod (develop & debug) — `examples/interactive_pod.yaml`

The pod starts, installs base deps, prints `READY_FOR_CODE`, and sleeps. You
copy your code in and drive it by hand. This is the right mode while your
script still has bugs.

```bash
kubectl apply -f examples/interactive_pod.yaml
kubectl get pods -w                          # wait for Running (Ctrl+C to stop watching)
kubectl logs my-gpu-pod | tail -2            # wait for READY_FOR_CODE

kubectl cp my_script.py my-gpu-pod:/workspace/my_script.py
kubectl exec my-gpu-pod -- python /workspace/my_script.py

kubectl cp my-gpu-pod:/output ./results      # bring results home
kubectl delete pod my-gpu-pod                # ALWAYS
```

### B. Batch job (reproducible runs) — `examples/batch_job.yaml`

A Job retries on failure (`backoffLimit`) and is the record of *exactly what
ran*. Use it once the script works.

```bash
kubectl apply -f examples/batch_job.yaml
kubectl logs -f job/my-experiment            # stream output
# when finished:
POD=$(kubectl get pods -l job-name=my-experiment -o jsonpath='{.items[0].metadata.name}')
kubectl cp $POD:/output ./results
kubectl delete job my-experiment
```

**Which one?** Develop in A until the script runs clean end-to-end, then
freeze it into B. (In practice we often stay in A with `nohup` — see Step 4.)

## Step 3 — Choosing a GPU

You select GPUs by *node affinity* on the product name. List what exists:

```bash
kubectl get nodes -L nvidia.com/gpu.product | awk '{print $NF}' | sort | uniq -c | sort -rn | head -20
```

In your YAML (see the examples), list every product that fits your model —
the scheduler picks whichever has a free card:

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: nvidia.com/gpu.product
              operator: In
              values: [NVIDIA-A10, NVIDIA-A40, NVIDIA-L40, NVIDIA-L40S]
```

Sizing cheat-sheet (bf16 weights ≈ 2 bytes/param, plus activations/KV):

| Model | Inference fits on | Full fine-tune / RL fits on |
|---|---|---|
| ≤1.5B | 2080 Ti (11 GB), T4 | 48 GB class (A6000/A40/L40/L40S) |
| 7B | 24 GB class (A10, 3090, 4090) | multi-GPU or LoRA |
| 20B (MXFP4) | 24–48 GB class | — |

To **exclude a broken node** (we hit one with a corrupted NVIDIA driver):

```yaml
            - key: kubernetes.io/hostname
              operator: NotIn
              values: [uicnrp-fiona2.evl.uic.edu]
```

## Step 4 — The pattern that survives disconnects: `nohup` + log file

`kubectl exec` sessions die (laptop sleeps, network resets, timeouts). Never
run anything long in a bare exec. Launch detached, log to a file, poll the log:

```bash
# launch (returns immediately)
kubectl exec my-gpu-pod -- bash -c \
  'nohup python /workspace/train.py > /output/train.log 2>&1 & echo launched'

# check on it whenever
kubectl exec my-gpu-pod -- tail -5 /output/train.log

# wait for a completion marker (put one in your script: print("DONE"))
until kubectl exec my-gpu-pod -- grep -qE 'DONE|Traceback' /output/train.log; do sleep 60; done
```

Make your script's last line print a sentinel (`DONE`, `TRAINING_COMPLETE`) —
greppable completion beats guessing.

## Step 5 — Persistent storage (PVC) — `examples/pvc.yaml`

Anything you can't afford to lose goes on a PVC, mounted at e.g. `/ckpt`:

```bash
kubectl apply -f examples/pvc.yaml       # once, ever
```

Then in the pod spec, mount it and point caches at it:

```yaml
env:
  - name: HF_HOME
    value: /ckpt/hf_cache        # model downloads survive pod restarts
volumeMounts:
  - {name: ckpt, mountPath: /ckpt}
volumes:
  - name: ckpt
    persistentVolumeClaim: {claimName: my-lab-pvc}
```

First run downloads the model once; every later pod reuses it. Same for
datasets and checkpoints. Note: `du -sh` on cephfs is slow — don't panic.

## Step 6 — Secrets (tokens)

Never put tokens in YAML or code. Create once:

```bash
kubectl create secret generic hf-secret --from-literal=HF_TOKEN=hf_xxx
kubectl create secret generic wandb-secret --from-literal=WANDB_API_KEY=xxx
```

Reference them as env vars in the pod spec (see `examples/batch_job.yaml`).

## Step 7 — A complete worked example

This is the actual flow we used to measure GPU energy per generation
(profiling Qwen2.5-1.5B on a dedicated 2080 Ti):

```bash
# 1. launch the interactive pod (GPU list = anything that fits 1.5B)
kubectl apply -f examples/interactive_pod.yaml

# 2. wait until it's placed and ready
kubectl get pods -w
kubectl logs my-gpu-pod | grep READY_FOR_CODE

# 3. pin fragile deps INSIDE the pod (see §8 for why), copy code in
kubectl exec my-gpu-pod -- pip install -q 'transformers==4.46.3' 'accelerate==1.1.1'
kubectl cp profile.py my-gpu-pod:/workspace/profile.py

# 4. run detached with a log
kubectl exec my-gpu-pod -- bash -c \
  'cd /workspace && nohup python profile.py --out /output/run1 > /output/run1.log 2>&1 & echo ok'

# 5. poll until the sentinel
until kubectl exec my-gpu-pod -- grep -q 'SUMMARY' /output/run1.log 2>/dev/null; do sleep 60; done

# 6. collect and clean up
kubectl cp my-gpu-pod:/output/run1 ./results/run1
kubectl delete pod my-gpu-pod
```

## Step 8 — Errors you WILL hit (we hit all of them in one day)

| Symptom | Cause | Fix |
|---|---|---|
| Pod `StartError`: "NVML: Driver/library version mismatch" | broken driver **on that node** | exclude the node by hostname (§3), relaunch |
| Pod `Pending` forever | no free GPU matching your list | widen the product list, or try later |
| Setup script dies silently at `conda activate` | conda's activate breaks under `set -u` | skip activation; call `/opt/conda/envs/ea/bin/python` by absolute path |
| `conda create` fails on ToS | new conda requires accepting channel ToS | `conda create -n ea -c conda-forge --override-channels python=3.10` |
| `fast-downward` / native builds fail | runtime images lack compilers | `apt-get install -y build-essential cmake g++` |
| build says `No such file: 'python'` | image has only `python3` | `ln -s $(which python3) /usr/local/bin/python` |
| `pip install -r requirements.txt` half-installs | one package (e.g. flash-attn) fails and aborts the rest | filter it out (`grep -v flash-attn`), install the rest, use fallbacks (`VLLM_ATTENTION_BACKEND=TORCH_SDPA`) |
| Imports break after `pip install -U something` | unpinned pip upgraded torch/transformers past compatibility | **pin versions**; if torchvision breaks transformers, `pip uninstall torchvision` |
| `kubectl exec` hangs or "connection reset" | pod CPU-saturated, or API-server flakiness | it does NOT mean your job died — use `kubectl get pod` (light) + the log file |
| Results vanished with the pod | they were on emptyDir | put anything precious on the PVC (§5) |

## Step 9 — Etiquette

- **Delete your pods/jobs when done.** `kubectl get pods` before logging off.
- One GPU per pod unless you truly need more; `requests` = `limits`.
- Name things so others can tell whose they are and what they do
  (`danielar-effort-profiling`, not `test2`).
- Big downloads (models, datasets) go to the PVC once — not to emptyDir on
  every pod.
- If you take an exclusive GPU for measurement, verify it *is* exclusive
  (our probe checks `nvmlDeviceGetComputeRunningProcesses`) and release it fast.

---

## Files in `examples/`

| File | What it is |
|---|---|
| `interactive_pod.yaml` | develop-mode pod: GPU + deps + `READY_FOR_CODE` + sleep |
| `batch_job.yaml` | reproducible Job with secrets + PVC mounts |
| `pvc.yaml` | persistent 60 Gi volume (apply once) |
| `hello_gpu.py` | 20-line sanity script: GPU name, VRAM, NVML energy counter |

Questions → ask in the lab channel. Real production examples live in
`agent-energy-profiling/nautilus/` and `energy-agent-training-2026-07-29/`
in the group's repos.
