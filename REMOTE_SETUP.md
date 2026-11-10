# GPU machine setup (Windows, RTX 5080, reached with AnyDesk)

Runs are started directly on the PC and are local processes there, so nothing depends on the
AnyDesk connection staying up. All commands are PowerShell. Nothing in this file has been run on
the GPU machine; treat the first pass as a checklist and tell me where it deviates.

## 1. AnyDesk

Gotchas first:

- **Install AnyDesk, do not run the portable exe.** Only the installed version runs as a service, which is what lets you reconnect after a reboot or when nobody is at the PC.
- **Never sign out of Windows on the PC.** Signing out kills every process of that user, including training. Close the AnyDesk window (or lock with `Win+L`) instead.
- **No monitor attached or monitor powered off** can give a black or tiny remote screen. Leave a monitor connected, or use an HDMI dummy plug.
- AnyDesk file transfer is too slow for datasets (tens of GB). Use an external drive or a cloud download on the PC itself.

Steps:

1. On the PC: install AnyDesk, then Settings → Security → *Unattended Access* → enable and set a strong password; enable two-factor if offered.
2. Note the PC's AnyDesk address. On the laptop: connect to it and tick *Log in automatically*.
3. Test once: reboot the PC, reconnect from the laptop without anyone touching the PC.

## 2. Keep the machine awake and stop surprise restarts

Do this before the first long run. Run PowerShell **as Administrator**:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change disk-timeout-ac 0
powercfg /hibernate off
```

The display may still turn off; that is harmless.

Windows Update is the usual cause of a run dying overnight:

- Settings → Windows Update → **Pause updates** (up to 5 weeks) before a multi-day batch of runs. Resume and update between batches.
- Settings → Windows Update → Advanced options → Active hours: set manually to the widest window (18 h).
- Install pending updates and reboot *before* launching, so no restart is already queued.

GPU driver updates (NVIDIA app / GeForce Experience) also reset the GPU and kill a run: turn off automatic driver installation.

## 3. Python environment

Gotchas first:

- **RTX 50-series needs a PyTorch build for CUDA 12.8 (`cu128`), i.e. PyTorch 2.7 or newer.** Older builds (cu118 / cu121 / cu124) install fine and then fail at the first CUDA call with a "sm_120 is not compatible" warning or "no kernel image is available". The plain `pip install torch` on Windows gives a **CPU-only** build, so the index URL below is required.
- You do not need to install the CUDA toolkit or cuDNN; the wheel bundles them. You do need a current NVIDIA driver (570 series or newer).
- Use **Python 3.12** (3.13 also has wheels, but 3.12 is what the code was tested with).
- Clone into a short path without spaces, e.g. `C:\work\coevo`. The code handles spaces, some tools do not.

```powershell
nvidia-smi                                   # driver version and that the 5080 is visible
py -3.12 --version

mkdir C:\work; cd C:\work
git clone <your-repo-url> coevo
cd coevo

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1                 # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt              # torch is already satisfied and is left alone

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); x=torch.randn(8,8,device='cuda'); print((x@x).sum().item())"
```

Expected: `2.7.1+cu128 True NVIDIA GeForce RTX 5080` and a number, with no warning. If the version
prints `+cpu`, the second `pip install` replaced the build: uninstall torch and torchvision and
repeat the `--index-url` line.

Then run the tests and the smoke test on the PC (about a minute; see README):

```powershell
python -m pytest -q
python tools\make_synthetic.py --out data_synthetic
python train.py --config configs\base.yaml configs\smoke.yaml
python train.py --config configs\base.yaml configs\smoke.yaml --set device=cuda output_dir=runs\smoke_gpu phase1.cache_dir=runs\smoke_gpu\anchors
```

## 4. Datasets

Gotchas first:

- The MSMT17_V2 copy on the laptop is **incomplete** (no `mask_train_v2`, test set cut off at identity 2016). Extract the archive again on the PC and check the counts below.
- Three more datasets are needed before any real run: **Market-1501, CUHK02, CUHK03**.
- First run downloads model weights (CLIP ViT-B/16 ≈ 600 MB from Hugging Face, ResNet-50 ≈ 100 MB from download.pytorch.org). The PC needs internet for that once.
- Put the data on an SSD and, if reading is slow, add the data folder to Windows Security → Virus & threat protection → Exclusions. Real-time scanning of ~250k small files is a common hidden bottleneck.

Layout (`data/` inside the repo is git-ignored; any other location works with `--set data.root=D:/reid`):

```
data/
  market1501/Market-1501-v15.09.15/{bounding_box_train,bounding_box_test,query}/*.jpg
  msmt17/MSMT17_V2/{mask_train_v2,mask_test_v2}/<pid>/*.jpg
  msmt17/MSMT17_V2/{list_train,list_val,list_query,list_gallery}.txt
  cuhk02/Dataset/{P1..P5}/{cam1,cam2}/*.png
  cuhk03/cuhk03_release/cuhk-03.mat          # as downloaded
  cuhk03/cuhk03_release/{images/,meta.json,splits.json}   # produced by the next command
```

CUHK03 conversion (once; this script has not been run against the real file yet):

```powershell
python tools\prepare_cuhk03.py --mat data\cuhk03\cuhk03_release\cuhk-03.mat --out data\cuhk03\cuhk03_release
```

It must print `identities: 1467 ... splits: 20 test ids in split 0: 100`. Anything else: stop and tell me.

Check all four:

```powershell
python -c "from coevo.config import load_config; from coevo.data import load_dataset, missing_files, dataset_names; cfg=load_config(['configs/base.yaml']); [print(d.summary(), '| missing files:', len(missing_files(d.train+d.query+d.gallery))) for d in (load_dataset(n, cfg.data) for n in dataset_names())]"
```

Expected (train ids / train images; query; gallery):

| Dataset | Train | Query | Gallery |
|---|---|---|---|
| market1501 | 751 / 12,936 | 3,368 | 15,913 (junk removed) |
| msmt17 | 1,041 / 32,621 | 11,659 | 82,161 |
| cuhk02 | 1,577 ids | P5 cam1 | P5 cam2 |
| cuhk03 | 1,367 ids | 200 | rest of the 100 test ids |

and `missing files: 0` on every line. `train.py` refuses to start if any file is missing.

## 5. Launching a run that survives a disconnect

Gotchas first:

- **Do not click inside a console window that is running training.** Windows consoles enter "select" mode on a click and freeze the process until you press Enter. Launching hidden with output redirected (below) avoids this.
- A run started in a visible PowerShell window dies when that window is closed. The launch below is detached from the window.
- DataLoader workers are separate processes on Windows and each loads the CUDA DLLs. With three clients that is 3 × `data.num_workers` processes. If you see `OSError: [WinError 1455] The paging file is too small`, lower `data.num_workers` (default 4) or let Windows manage the page file (System → About → Advanced system settings → Performance → Advanced → Virtual memory → *Automatically manage*).

Launch (from `C:\work\coevo`, venv not required to be active):

```powershell
$run = "p1_market"
New-Item -ItemType Directory -Force "runs\$run" | Out-Null
$p = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
  -ArgumentList "train.py --config configs\base.yaml configs\$run.yaml" `
  -WorkingDirectory (Get-Location) -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput "runs\$run\stdout.txt" -RedirectStandardError "runs\$run\stderr.txt"
$p.Id | Set-Content "runs\$run\pid.txt"
```

You can now close PowerShell and AnyDesk.

Monitor:

```powershell
Get-Content runs\p1_market\train.log -Tail 20 -Wait        # Ctrl+C stops watching, not the run
Get-Content runs\p1_market\stderr.txt -Tail 30             # tracebacks land here
Get-Process -Id (Get-Content runs\p1_market\pid.txt)       # still alive?
nvidia-smi -l 5                                            # GPU load and memory
```

TensorBoard runs on the PC and you view it through AnyDesk in the PC's browser, so no port
forwarding is needed:

```powershell
.\.venv\Scripts\tensorboard.exe --logdir runs --port 6006  # then open http://localhost:6006 on the PC
```

Stop a run:

```powershell
Stop-Process -Id (Get-Content runs\p1_market\pid.txt)
```

## 6. Resuming after a crash or reboot

`<output_dir>\ckpt\last.pt` is rewritten after every round (written to a temp file and renamed, so a
crash mid-write cannot corrupt it). **Run the exact same launch command again**: the run logs
`resumed from ... at round N` and continues. Phase I anchors are cached in `cache\anchors\` and are
not recomputed; if a crash happens during Phase I, only the client that was in progress is redone.

To start over instead of resuming, delete `runs\<run>\ckpt\` (or use a new `output_dir`).

After a reboot nothing restarts by itself. Reconnect with AnyDesk and relaunch.
