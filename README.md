# Nautilus for the Shi Lab — a practical tutorial

How to run GPU experiments on [NRP Nautilus](https://nrp.ai) using the Shi Lab
namespace — from first login to a reproducible batch job. Every command in
this guide has been used in real lab runs — and because things do break, there
is a table of the errors we actually hit (with fixes) near the end (Step 9).

**Two ways to learn it:** read this guide top to bottom, or open
[`notebooks/01_first_gpu_job.ipynb`](notebooks/01_first_gpu_job.ipynb) and do
the complete hands-on exercise cell by cell (~20 min, one small GPU). The
notebook is self-contained: **section 0 checks your setup** (kubectl, the
kubelogin plugin, your config, namespace access) and prints the exact fix for
whatever is missing, so start there if a command in this guide fails.

> **Golden rule (lab policy): verify correctness locally first.** Nautilus is
> for reproducible, compute-intensive runs — not for debugging basic syntax,
> data-path, or dependency errors. Start with the smallest useful GPU request
> and scale only after a successful smoke test.

**What Nautilus is:** a national Kubernetes cluster with free shared GPUs
(compute contributed by universities, including UCSD). You don't SSH into
machines — you describe *what to run* in a YAML file, submit it with
`kubectl`, and the cluster finds a machine for you.

**Our namespace:** `shi-group-ece-advanced`.
**What GPUs can I use?** Availability changes — check the live list on the
[NRP resources page](https://portal.nrp-nautilus.io/resources) or run
`kubectl get nodes -L nvidia.com/gpu.product` (Step 4). Note that premium GPUs
(A100/H100-class) require an explicit namespace quota grant; without one, the
free pool (consumer and workstation cards up to 48 GB) is what schedules.

---

## Step 0 — Get access (once)

1. Email **Prof. Yuanyuan Shi** (yyshi@ucsd.edu) and ask to have your UCSD
   email added to the lab Group Workspace on NRP/Nautilus.
2. Once added, log in at [portal.nrp-nautilus.io](https://portal.nrp-nautilus.io)
   (UCSD institutional login) and confirm you can see the lab workspace.
3. Install `kubectl` **and `kubelogin`** (`brew install kubectl int128/kubelogin/kubelogin`
   on macOS), then follow NRP's
   [Getting Started](https://docs.nrp.ai/userdocs/start/getting-started/) to
   download your config into `~/.kube/config`. `kubelogin` installs a binary
   called **`kubectl-oidc_login`** — that is the name to check for
   (`which kubectl-oidc_login`); `which kubelogin` may find nothing even on a
   correct install. `kubectl` locates that plugin *through your `PATH`*, so
   anything that starts with a short `PATH` (notably a Jupyter kernel launched
   from the Dock — see Step 9) cannot authenticate.
4. Set the lab namespace as your default:

```bash
kubectl config set-context nautilus --namespace=shi-group-ece-advanced
```

5. Verify (and know the fix for stale logins):

```bash
kubectl config current-context   # should print: nautilus
kubectl get pods                 # empty list is fine; an error is not
kubectl oidc-login clean         # if auth ever goes stale, then retry
```

## Step 0.5 — One environment for everyone (do this once)

Version drift ("works on my laptop") is the most expensive kind of problem on a
shared cluster, so this repo pins the environment in **two separate places** —
and only one of them is on your machine:

| Layer | Pinned by | What lives there |
|---|---|---|
| **Your laptop** — control plane | `env/requirements-notebook.txt` (just `ipykernel`) | `kubectl` + a Python kernel. The tutorial notebook imports **only the standard library**, so there is nothing here that can drift between students. |
| **The container** — where your code runs | `env/Dockerfile` + `env/requirements-lab.txt`, pinned **by digest** | torch, CUDA, transformers, your deps. This is where versions actually break, so this is where they are frozen. |

**Laptop, once:**

```bash
bash env/setup_local_env.sh     # .venv + a kernel named "Nautilus Lab (.venv)"
```

Then select **Nautilus Lab (.venv)** in VS Code's kernel picker. The script also
lists every kernel registered on your machine and marks the **dead** ones — a
kernel left behind by a deleted project makes the notebook fail *before the first
cell runs*, which looks like a broken notebook and is not one.

The notebook names every object it creates after **you** (`<you>-tutorial-pod`), so
the whole lab can do the exercise at the same time. It takes that tag from your
**git identity** (`git config user.email` → `d2rojas@ucsd.edu` → `d2rojas`) rather
than your machine's login name, which on a shared or inherited laptop is often
someone else entirely. Override it with `export LAB_USER=<your-ucsd-username>` or
by setting `USER_TAG` in the notebook's config cell.

**Container, once per lab (not per student):**

```bash
REGISTRY=ghcr.io/ucsd-pier-lab bash env/build_image.sh
```

It builds `env/Dockerfile` for `linux/amd64` (Nautilus nodes are amd64, even if you
build on an Apple Silicon Mac), pushes it, and prints the **digest**. Put the digest
— not the tag — in your YAML, because a tag can be overwritten upstream while a
digest cannot:

```yaml
image: ghcr.io/ucsd-pier-lab/shi-lab-gpu@sha256:<digest>
```

The notebook picks this up from an env var, so nobody has to edit cells:

```bash
export LAB_IMAGE=ghcr.io/ucsd-pier-lab/shi-lab-gpu@sha256:<digest>
```

**Why an image instead of `pip install` at job start** — the pattern the examples
still show as a fallback:

- one student's `pip install` resolving a different version than another's is
  exactly the drift we are trying to kill;
- a failed download mid-`pip install` leaves a half-built environment (Step 9);
- installs at job start burn GPU-minutes on every single run.

Until the lab image exists, the honest fallback is: install from
`env/requirements-lab.txt` (all pins, no ranges) and never `pip install -U`
anything on top of the CUDA base image — upgrading torch is the fastest way to
break a working pod. Every Job in this repo prints its image, Python version and
key package versions at startup, so **the log is the record of the environment
that ran**.

## Step 1 — The recommended lab workflow

| Stage | What you do | Exit criterion |
|---|---|---|
| **A. Local debug** | run on your laptop/workstation: imports, paths, data loading, forward/backward, checkpoint save/load | a small run completes correctly |
| **B. Cluster smoke test** | submit **1 GPU**, small data subset, few steps | job completes; logs/checkpoints/results are correct |
| **C. Full experiment** | scale GPUs/CPU/RAM/data only as justified | run is reproducible and recorded in the lab tracker |
| **D. Cleanup** | copy final outputs; delete finished/failed jobs | no abandoned jobs or wasted resources |

Before any substantial run: push the exact code to GitHub
(`github.com/ucsd-pier-lab/<your-project>`), record the commit, and log the
run in the **Group Computing Tracker** document. Make the command
non-interactive and self-contained (`python train.py --config configs/exp1.yaml`).

## Step 2 — The five concepts you actually need

| Concept | What it is | When you use it |
|---|---|---|
| **Job** | a pod that runs to completion (with retries) | **the default** — all real runs |
| **Pod** | one running container | short interactive debugging only |
| **emptyDir** | scratch disk that **dies with the pod** | temp files, code you `kubectl cp` in |
| **PVC** | persistent disk that **survives pods** | checkpoints, HF cache, datasets |
| **Secret** | encrypted key-value in the namespace | HF_TOKEN, WANDB_API_KEY |

Rules of the namespace (Nautilus + lab policy, both enforced):
- `requests` ≈ `limits` (Nautilus caps the ratio; when unsure set them equal).
- Bare interactive pods are reaped after ~6 h without GPU use.
- **Never reserve a GPU with `sleep infinity` and "start training later".**
  The actual training command goes in the Job YAML. (Lab rule.)
- Always delete your pods/jobs when done — GPUs are shared.

## Step 3 — Launching work: Job by default, pod only to debug

### A. Batch Job (the default) — `examples/batch_job.yaml`

The real command lives **in the YAML**; the Job retries on failure
(`backoffLimit`) and is the record of exactly what ran.

```bash
kubectl apply -f examples/batch_job.yaml
kubectl get jobs && kubectl get pods
kubectl logs -f job/my-experiment            # stream output
# when finished:
POD=$(kubectl get pods -l job-name=my-experiment -o jsonpath='{.items[0].metadata.name}')
kubectl cp $POD:/output ./results
kubectl delete job my-experiment
```

### B. Interactive pod (debug sessions ONLY) — `examples/interactive_pod.yaml`

For the cluster-specific debugging a laptop can't do (does my model fit on
this card? does the env build here?). The pod starts, prints
`READY_FOR_CODE`, and sleeps so you can `kubectl cp` / `exec` by hand.

```bash
kubectl apply -f examples/interactive_pod.yaml
kubectl logs my-gpu-pod | tail -2            # wait for READY_FOR_CODE
kubectl cp my_script.py my-gpu-pod:/workspace/my_script.py
kubectl exec my-gpu-pod -- python /workspace/my_script.py
kubectl delete pod my-gpu-pod                # SAME DAY, always
```

**The line between A and B:** the moment your script runs end-to-end, freeze
it into a Job. An interactive pod is a debugging session, not a reservation —
holding a GPU idle in one is exactly what the lab rule above prohibits.

## Step 4 — Choosing a GPU

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

Start with **one** small GPU (stage B above); scale only after it works.
To **exclude a broken node** (we hit one with a corrupted NVIDIA driver):

```yaml
            - key: kubernetes.io/hostname
              operator: NotIn
              values: [uicnrp-fiona2.evl.uic.edu]
```

## Step 5 — Monitor, diagnose, stop

```bash
kubectl logs -f <pod-name>          # stream output
kubectl describe pod <pod-name>     # THE debugging command: scheduling events, kill reasons
kubectl delete job <job-name>       # stop a bad run IMMEDIATELY — don't let it retry
```

What the statuses mean:

| Status | Meaning | First move |
|---|---|---|
| `Pending` | requested resources not schedulable right now | `describe pod`; widen GPU list or shrink request |
| `OOMKilled` | memory limit exceeded | raise memory, or reduce batch/model |
| `Error` / CrashLoop | command, image, data, or env failed | `logs` + `describe pod` |
| `StartError` (NVML mismatch) | broken driver on that node | exclude the node (Step 4), relaunch |

**For anything long, use the disconnect-proof pattern** — `kubectl exec`
sessions die (laptop sleeps, network resets). Launch detached, log to a file,
poll the log:

```bash
kubectl exec my-gpu-pod -- bash -c \
  'nohup python /workspace/train.py > /output/train.log 2>&1 & echo launched'
kubectl exec my-gpu-pod -- tail -5 /output/train.log
until kubectl exec my-gpu-pod -- grep -qE 'DONE|Traceback' /output/train.log; do sleep 60; done
```

Make your script's last line print a sentinel (`DONE`) — greppable completion
beats guessing. (A Job with the command in its YAML doesn't need this; it's
for exec-driven debug sessions.)

## Step 6 — Data, checkpoints, storage (PVC) — `examples/pvc.yaml`

Containers are stateless: files inside can disappear on restart or eviction.

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

- Checkpoint **frequently enough that a node restart doesn't erase hours** of training.
- Write to a unique experiment directory — `/ckpt/<user>/<project>/<date>-<exp>/` —
  never a shared top-level folder.
- Ephemeral/emptyDir is for temp files only; **large writes to ephemeral
  storage can get the pod evicted** unless `ephemeral-storage` is requested
  explicitly.
- `du -sh` on cephfs is slow — don't panic.

## Step 7 — Secrets (tokens)

Never put tokens in YAML or code. Create once:

```bash
kubectl create secret generic hf-secret --from-literal=HF_TOKEN=hf_xxx
kubectl create secret generic wandb-secret --from-literal=WANDB_API_KEY=xxx
```

Reference them as env vars in the pod spec (see `examples/batch_job.yaml`).

## Step 8 — A complete worked example

The actual flow we used to measure GPU energy per generation
(profiling Qwen2.5-1.5B on a dedicated 2080 Ti — an exec-driven debug/measure
session; a training run would be a Job instead):

```bash
# 1. launch the pod (GPU list = anything that fits 1.5B)
kubectl apply -f examples/interactive_pod.yaml
kubectl get pods -w
kubectl logs my-gpu-pod | grep READY_FOR_CODE

# 2. pin fragile deps INSIDE the pod (the errors table explains why), copy code in
kubectl exec my-gpu-pod -- pip install -q 'transformers==4.46.3' 'accelerate==1.1.1'
kubectl cp profile.py my-gpu-pod:/workspace/profile.py

# 3. run detached with a log
kubectl exec my-gpu-pod -- bash -c \
  'cd /workspace && nohup python profile.py --out /output/run1 > /output/run1.log 2>&1 & echo ok'

# 4. poll until the sentinel
until kubectl exec my-gpu-pod -- grep -q 'SUMMARY' /output/run1.log 2>/dev/null; do sleep 60; done

# 5. collect and clean up
kubectl cp my-gpu-pod:/output/run1 ./results/run1
kubectl delete pod my-gpu-pod
```

## Step 9 — Errors you WILL hit (we hit all of them in one day)

| Symptom | Cause | Fix |
|---|---|---|
| Pod `StartError`: "NVML: Driver/library version mismatch" | broken driver **on that node** | exclude the node by hostname (Step 4), relaunch |
| In a notebook: `command not found: kubectl`, or `error: unknown command "oidc-login" for "kubectl"` — while the same command works in your terminal | the Jupyter/VS Code kernel did **not** inherit your shell `PATH`, so `kubectl` (or its `kubectl-oidc_login` plugin) is invisible to it | run **section 0 of the notebook** — it repairs `PATH` inside the kernel. Outside the notebook: launch the editor from a terminal, or add `/opt/homebrew/bin` (and `~/.krew/bin`) to the kernel's `PATH` |
| `error: You must be logged in to the server (Unauthorized)` | the OIDC token cached in your keyring expired | `kubectl oidc-login clean`, then any `kubectl` command re-opens the browser login |
| Notebook fails *before any cell runs* ("kernel failed to start") | a stale kernel is selected — its virtualenv was deleted | `bash env/setup_local_env.sh`, then pick **Nautilus Lab (.venv)**; the script marks dead kernels |
| `pods is forbidden: User ... cannot list resource "pods"` | authenticated, but not a member of the namespace | ask Prof. Shi to add your UCSD email to the lab workspace (Step 0) |
| Job stays `Pending` and it asks for an A100/H100 | premium GPUs need an explicit quota grant; ours is `0/0` | check with `kubectl get resourcequota`; request free-pool cards instead (Step 4) |
| Pod `Pending` forever | no free GPU matching your list | widen the product list, or try later |
| Setup script dies silently at `conda activate` | conda's activate breaks under `set -u` | skip activation; call `/opt/conda/envs/ea/bin/python` by absolute path |
| `conda create` fails on ToS | new conda requires accepting channel ToS | `conda create -n ea -c conda-forge --override-channels python=3.10` |
| `fast-downward` / native builds fail | runtime images lack compilers | `apt-get install -y build-essential cmake g++` |
| build says `No such file: 'python'` | image has only `python3` | `ln -s $(which python3) /usr/local/bin/python` |
| `pip install -r requirements.txt` half-installs | one package (e.g. flash-attn) fails and aborts the rest | filter it out (`grep -v flash-attn`), install the rest, use fallbacks (`VLLM_ATTENTION_BACKEND=TORCH_SDPA`) |
| Imports break after `pip install -U something` | unpinned pip upgraded torch/transformers past compatibility | **pin versions**; if torchvision breaks transformers, `pip uninstall torchvision` |
| `kubectl exec` hangs or "connection reset" | pod CPU-saturated, or API-server flakiness | it does NOT mean your job died — use `kubectl get pod` (light) + the log file |
| Results vanished with the pod | they were on emptyDir | put anything precious on the PVC (Step 6) |

## Step 10 — Lab rules checklist (before any substantial run)

- ☐ Code verified **locally** before requesting cluster GPUs.
- ☐ Job name identifies **you + the experiment** (`danielar-effort-profiling`, not `test2`).
- ☐ Exact code pushed to GitHub (`ucsd-pier-lab/...`) and the **commit recorded**.
- ☐ Launch command + config file recorded; run logged in the **Group Computing Tracker**.
- ☐ A 1-GPU **smoke test** succeeded before the full experiment.
- ☐ Checkpoints/results go to the **PVC**, in a unique experiment directory.
- ☐ Failed or obsolete jobs **deleted promptly**.
- ☐ Final result/conclusion recorded — expensive runs are not repeated unnecessarily.
- ☐ If you took an exclusive GPU for measurement: verified exclusivity
  (`hello_gpu.py`) and released it fast.

## Command quick reference

| Task | Command |
|---|---|
| Launch | `kubectl apply -f <file>.yaml` |
| List jobs / pods | `kubectl get jobs` · `kubectl get pods` |
| Stream logs | `kubectl logs -f <pod>` |
| Diagnose scheduling/failure | `kubectl describe pod <pod>` |
| Run something in a pod | `kubectl exec <pod> -- <cmd>` |
| Copy in / out | `kubectl cp local <pod>:/path` · `kubectl cp <pod>:/path local` |
| Stop/delete a run | `kubectl delete job <job>` · `kubectl delete pod <pod>` |
| Refresh stale login | `kubectl oidc-login clean` |
| What GPUs exist | `kubectl get nodes -L nvidia.com/gpu.product` |

---

## Contributing — help keep this guide alive

The cluster changes, images change, new errors appear. This repo works as a
small lab community effort:

- **Hit an error that isn't in the Step 9 table?** [Open an issue](../../issues/new/choose)
  with the symptom, what caused it, and (if you found one) the fix — that's
  exactly how the current table was built.
- **Have a fix or an improvement?** Pull requests are welcome — to the guide,
  the templates, or the notebook.
- **Something in the guide conflicts with current NRP docs?** The official
  docs win; open an issue so the guide gets corrected.

Maintained and reviewed by **Daniela Rojas** ([@d2rojas](https://github.com/d2rojas)) —
issues and PRs are reviewed to keep the guide current.

## What's in this repo

| File | What it is |
|---|---|
| `notebooks/01_first_gpu_job.ipynb` | **start here** — the full exercise, cell by cell (self-contained: preflight, unique object names per student, cleanup) |
| `batch_job.yaml` | **the default**: reproducible Job with secrets + PVC mounts |
| `interactive_pod.yaml` | debug-session pod: GPU + deps + `READY_FOR_CODE` + sleep |
| `pvc.yaml` | persistent 60 Gi volume (apply once) |
| `hello_gpu.py` | 20-line sanity script: GPU name, VRAM, exclusivity, NVML energy counter |
| `env/setup_local_env.sh` | laptop setup: `.venv` + the **Nautilus Lab** kernel, and it flags dead kernels |
| `env/Dockerfile` · `env/requirements-lab.txt` | the pinned lab image — one environment for every student |
| `env/build_image.sh` | build + push it, and print the digest to pin in your YAML |

**Read next when you need more detail:**
[NRP User Docs](https://docs.nrp.ai) · [Running Batch Jobs](https://docs.nrp.ai/userdocs/running/jobs/) ·
[Storage](https://docs.nrp.ai/userdocs/storage/intro/) ·
[kubectl cheat sheet](https://kubernetes.io/docs/reference/kubectl/quick-reference/).

*NRP policies and resources change. If this guide conflicts with current NRP
documentation, follow the official docs and tell the lab admin so this guide
gets updated.*
