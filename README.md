# SO-101 Vision Pick-and-Place

Simulation, motion control and computer vision for a single SO-101 follower arm.

## Setup

```bash
git clone https://github.com/NE0NMIKE/so101-vision.git
cd so101-vision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`lerobot` pulls in PyTorch, which is a large download (several GB with CUDA).

On Linux, add yourself to the `dialout` group so the scripts can open the arm's serial port (`/dev/ttyACM0`), then log out and back in:

```bash
sudo usermod -aG dialout $USER
```

## Quick test (simulation only)

```bash
python scripts/so101_sim_move.py --scene assets/urdf/scene.xml --target 0.20 0.00 0.15
```

Add `--viewer` to open the 3D window (on macOS, run it with `mjpython` instead of `python`).
