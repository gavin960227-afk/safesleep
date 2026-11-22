Hello I just try to build a workflow for our final project!! (I asked the teacher and he says he recommend to finetune the pre-exist models ex:YOLO and we should focus on how to improve rather building a model from ground up)
I am currently using the data set from "roboflow.com" and this is the share link "https://app.roboflow.com/join/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3Jrc3BhY2VJZCI6ImtBZU5IQWZSQlpnaXpuRnFRbThYc3BtNTNZdTIiLCJyb2xlIjoib3duZXIiLCJpbnZpdGVyIjoiZ2F2aW45NjAyMjdAZ21haWwuY29tIiwiaWF0IjoxNzk1MzM2ODkxfQ.GtGgJkg1rEQIciQjD-FApAYKqTllDkO8RRPFnH2uMO0" the dataset i use have 738 images but the labels are dangerous safe and warning, the labels I want to use is back, stomach, side and  face_covered. I think we can share this link with the KEE department teammates and ask them to help do the labling(I can record a short video to teach them how to relabel the images).
I have created the workflow such that once the whole YOLO8 format output from roboflow is sent into data/raw we can run src/sort_by_label and the data will fall into data/posture/{train/val}/{back, ...} then we run src/train_posture and the model(weights) and its info will be saved to runs/.
I havent started the object detection part, please feel free to tell me what you would like to change or any suggestions 
by Gavin 許維恆:)  



# SafeSleep: Infant Sleep Safety Monitor

A camera-based system that watches a sleeping infant and raises an alarm when it detects a dangerous situation: the baby sleeping on its stomach, a covered face, or soft or loose objects on the sleep surface.

> **Not a medical device.** This is a student machine-learning project and a supplementary aid at best. It must never replace safe-sleep practice (baby on its back, bare firm crib, no loose bedding) or adult supervision. Do not rely on it to protect a real infant.

## Design

Two separate models, combined by alarm logic:

| Model | Task | Type | Classes |
|---|---|---|---|
| **A: Posture** | What position is the baby in? | YOLOv8 classification (`yolov8n-cls`) on the cropped baby | `back`, `stomach`, `side`, `face_covered` |
| **B: Objects** | What is on the bed? | YOLOv8 detection (`yolov8n`) | `pillow`, `loose_blanket`, `stuffed_toy`, `bumper` |

A baby detector finds the baby so Model A receives a clean crop. The alarm should fire only if a dangerous state lasts N seconds (temporal smoothing), to avoid false alarms from single frames. Priority is **high recall on dangerous states**.

## Status

- [x] Environment and GPU check
- [x] Data tooling (frame extraction, label sorting, train/val split)
- [x] Model A training script (tested on a 3-class workflow-test dataset)
- [ ] Real 4-class posture dataset
- [ ] Model B (object detection)
- [ ] Combined pipeline, alarm logic, live camera loop
- [ ] Night/infrared evaluation

## Repository layout

```
safesleep/
├── data/
│   ├── raw/                 # unsorted downloads, videos, crops (not in git)
│   ├── posture_test/        # workflow-test data: safe/dangerous/warning (not in git)
│   ├── posture/             # real 4-class data: train/ and val/ (not in git)
│   ├── sources.csv          # where every image came from + license
│   └── LABELING_GUIDE.md    # class definitions and edge-case rules
├── models/                  # trained weights (not in git)
├── src/
│   ├── check_env.py         # verifies PyTorch + CUDA
│   ├── smoke_test.py        # YOLO on webcam
│   ├── download_roboflow.py # download a dataset with the Roboflow API
│   ├── extract_frames.py    # video -> frames
│   ├── sort_by_label.py     # Roboflow YOLO labels -> class folders (crops)
│   ├── label_posture.py     # manual keypress labeling tool
│   ├── split_posture.py     # train/val split, grouped by source
│   ├── inspect_dataset.py   # per-class counts + contact sheets
│   ├── train_posture.py     # train Model A
│   └── predict_posture.py   # run Model A on images
├── requirements.txt
└── README.md
```

## Setup (Windows, PowerShell)

Requirements: **64-bit Python 3.10 to 3.12** and, for GPU training, an NVIDIA GPU with an up-to-date driver.

**1. Clone and enter the project**
```powershell
git clone <repo-url>
cd safesleep
```

**2. Create and activate a virtual environment**
```powershell
python -m venv .venv
.venv\Scripts\activate
```
If activation is blocked, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**3. Install PyTorch first** (choose the command for your machine at https://pytorch.org)
```powershell
# NVIDIA GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# or CPU only:
pip install torch torchvision
```

**4. Install the remaining packages**
```powershell
pip install -r requirements.txt
```

**5. Create the empty folders** (not stored in git)
```powershell
mkdir data\raw, data\posture, data\posture_test, models
```

**6. Verify the environment**
```powershell
python src/check_env.py
```
Expected: `CUDA available: True` (or `False` if you installed the CPU version; training still works, just slower). Optional webcam test: `python src/smoke_test.py` (press `q` to quit).

## Getting the data

The data is **not in the repository**: the images are large, carry their own licenses, and may show real infants.

### Workflow-test dataset (safe / dangerous / warning)
Used only to check that the pipeline works.

1. Open the Roboflow Universe project `bhone-min-thant/infant-safe-sleep-detection-f2dvm` (license **CC BY 4.0**).
2. Fork it into your own workspace, create a version (no augmentation), and **Export Dataset** in **YOLOv8** format.
3. Unzip into `data/raw/roboflow_infant/` (it should contain `train/`, `valid/`, `test/`, and `data.yaml`).
4. Crop each labeled box into class folders:
   ```powershell
   python src/sort_by_label.py
   ```
   This creates `data/posture_test/{train,val,test}/{dangerous,safe,warning}/`.

You can also download via the API: set `$env:ROBOFLOW_API_KEY = "<your key>"` and run `python src/download_roboflow.py`. Never commit an API key.

### Real 4-class posture dataset
1. Collect candidate images into `data/raw/unsorted/` (public datasets with a compatible license, frames from staged videos of **baby dolls** via `src/extract_frames.py`).
2. Sort them by hand with the keypress tool:
   ```powershell
   python src/label_posture.py
   ```
   Keys: `1` back, `2` stomach, `3` side, `4` face_covered, `s` skip, `d` delete, `q` quit.
3. Split into train/val (grouped by source so near-duplicate frames don't leak across the split):
   ```powershell
   python src/split_posture.py
   ```
4. Record every source and its license in `data/sources.csv`.

**Safety and privacy rules for data:** never photograph a real infant in a hazardous position or surrounded by pillows or loose blankets (use a doll). Only use footage of real infants with the parents' consent. Never commit images or video of infants.

## Training and testing

Train Model A (defaults train on the workflow-test data for 10 epochs):
```powershell
python src/train_posture.py
```
For the real data:
```powershell
python src/train_posture.py --data data/posture --name posture_v1 --epochs 50
```
Outputs go to `models/<name>/`: `weights/best.pt`, `confusion_matrix.png`, `results.png`.

Predict on images:
```powershell
python src/predict_posture.py path\to\image1.jpg path\to\image2.jpg
```
(`predict_posture.py` loads `models/posture_test/weights/best.pt`; edit the `weights` line to use another run.)

When evaluating, look at **per-class recall for `stomach` and `face_covered`** first. A missed dangerous position is the costly error. Always evaluate on babies, cribs, and lighting not seen in training.

## Troubleshooting

- **`WinError 193` when importing torch:** the Python in your virtual environment is 32-bit, or the PyTorch install is corrupted. Check with `python -c "import struct; print(struct.calcsize('P')*8)"` (should print `64`), then reinstall: `pip uninstall -y torch torchvision`, `pip cache purge`, and repeat setup step 3. Installing the Microsoft Visual C++ Redistributable (x64) can also help.
- **CUDA not available:** update the NVIDIA driver (`nvidia-smi` should work) and install the CUDA build of PyTorch **before** `ultralytics`.
- **Training hangs or crashes on Windows:** set `workers=0` in `src/train_posture.py`. The script already guards its entry point with `if __name__ == "__main__":`, which Windows requires.
- **Training says it can't find images:** the `--data` folder must contain `train/` and `val/`, each with one subfolder per class that contains images.

## Reproducibility notes

- The train/val split uses a fixed random seed (42). Training itself can still vary slightly between machines and GPU/CPU runs.
- Library versions are not pinned; if results differ from a teammate's, compare `pip freeze` output.

## Credits and licenses

- Workflow-test data: Infant-Safe-Sleep-Detection by `bhone-min-thant` on Roboflow Universe, licensed **CC BY 4.0** (https://universe.roboflow.com/bhone-min-thant/infant-safe-sleep-detection-f2dvm).
- Models built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0); check its license terms before distributing or commercializing this project.
- Add every additional dataset and its license to `data/sources.csv`.