"""Print a one-line progress summary of the SKEMPI fine-tuning run.
Called hourly by a Monitor. Reads the train-log CSV written by skempi_finetune.py.
"""
import os
import csv
import glob
import datetime
import subprocess

OUT = "/home/guoj0f/repos/StaB-ddG/output/2026-06-01"
CSVP = os.path.join(OUT, "skempi_ft_train_log.csv")
TOTAL_EPOCHS = 200


def alive():
    try:
        out = subprocess.check_output(["pgrep", "-fc", "skempi_finetune.py"]).decode().strip()
        return int(out) > 0
    except Exception:
        return False


def gpu_mem():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"]).decode().splitlines()
        return out[0].replace(" ", "") + "MiB(TITANX)"
    except Exception:
        return "n/a"


def main():
    if not os.path.exists(CSVP):
        print(f"[HEARTBEAT] train log 尚未出现; 进程存活={alive()}")
        return
    rows = list(csv.DictReader(open(CSVP)))
    done = len(rows)
    running = alive()
    if done == 0:
        print(f"[HEARTBEAT] 0 epochs 完成; 进程存活={running}")
        return
    last = rows[-1]
    # per-epoch time from last up-to-10 epochs
    if done >= 2:
        t_a = datetime.datetime.fromisoformat(rows[max(0, done - 11)]["timestamp"])
        t_b = datetime.datetime.fromisoformat(rows[-1]["timestamp"])
        n = done - max(0, done - 11)
        per = (t_b - t_a).total_seconds() / max(n, 1)
    else:
        per = float("nan")
    remain_h = per * (TOTAL_EPOCHS - done) / 3600 if per == per else float("nan")
    def _ep(p):
        b = os.path.basename(p)[len("skempi_ft_"):-len(".pt")]
        return int(b) if b.isdigit() else -1
    ckpts = sorted(glob.glob(os.path.join(OUT, "skempi_ft_*.pt")), key=_ep)
    last_ckpt = os.path.basename(ckpts[-1]) if ckpts else "(none yet)"
    flag = "  ⚠ 本epoch有复合物失败!" if last.get("n_failed_complexes", "0") not in ("0", "") else ""
    print(f"[HEARTBEAT {datetime.datetime.now():%H:%M}] "
          f"epoch {done}/{TOTAL_EPOCHS} | 进程存活={running} | "
          f"~{per:.0f}s/ep | 剩余~{remain_h:.1f}h | "
          f"loss={last['train_loss']} sp={last['train_spearman']} "
          f"n_failed={last['n_failed_complexes']} | "
          f"ckpt={last_ckpt} | GPU={gpu_mem()}{flag}")


if __name__ == "__main__":
    main()
