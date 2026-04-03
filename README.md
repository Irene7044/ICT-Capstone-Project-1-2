# ICT-Capstone-Project-1-2

## Project Setup for YOLO in VS Code using WSL Ubuntu

This project uses **VS Code with WSL Ubuntu** and a local Python virtual environment to install and run **Ultralytics YOLO**.

We use this setup so that all dependencies stay inside the project folder instead of being installed globally on the computer. This makes the project cleaner, easier to manage, and more consistent for all team members.

---

## 1. Open the project in VS Code using WSL

Open the GitHub repository in **VS Code** and make sure it is running in **WSL Ubuntu**.

You can check this in the bottom-left corner of VS Code. It should say something like:

```bash
WSL: Ubuntu
````

This is important because all commands below are meant to be run inside the WSL Linux environment.

---

## 2. Install Python tools in WSL

Open a terminal in VS Code and run:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```

This installs the required Python tools:

* `python3`
* `pip`
* `venv`

---

## 3. Create a virtual environment inside the project

From the root of the repository, run:

```bash
python3 -m venv .venv
```

This creates a local virtual environment called `.venv` inside the project folder.

A virtual environment keeps project packages isolated, which helps avoid conflicts with other Python projects on the machine.

---

## 4. Activate the virtual environment

Run the following command:

```bash
source .venv/bin/activate
```

If the activation works, the terminal should show `(.venv)` at the start of the command line.

Example:

```bash
(.venv) yosep@LAPTOP:~/ICT-Capstone-Project-1-2$
```

This means the virtual environment is active and ready to use.

---

## 5. Upgrade pip

After activating the environment, run:

```bash
python -m pip install --upgrade pip
```

This updates `pip` to the latest version inside the virtual environment.

---

## 6. Install Ultralytics YOLO

Install Ultralytics with:

```bash
pip install ultralytics
```

This installs the Ultralytics package, which includes YOLO models and tools for prediction, training, and testing.

---

## 7. Make VS Code use the virtual environment

Even if the terminal is already using `.venv`, VS Code should also use the same Python interpreter for running scripts and providing editor support.

### Steps

1. Press `Ctrl + Shift + P`
2. Search for `Python: Select Interpreter`
3. Choose the interpreter that looks like:

```bash
.venv/bin/python
```

If the interpreter option does not appear, install the **Python extension by Microsoft** in VS Code first.

---

## 8. Git ignore rules

The repository already includes a `.gitignore` file.

This prevents local environment files and generated YOLO outputs from being tracked by Git, including:

- `.venv/`
- `runs/`
- `datasets/`
- `*.pt`
- `*.onnx`
- `__pycache__/`
- `*.pyc`

Team members only need to pull the latest version of the repository to get these rules.

---
## 9. Check Git status

Run:

```bash
git status
```

If `.venv` does **not** appear in the output, that means Git is not tracking it, which is correct.

In our current setup, `.venv` was already not being tracked, so no extra removal step was needed.



## 11. Check that Ultralytics is installed

To confirm that Ultralytics was installed correctly, run:

```bash
pip show ultralytics
```

If installation was successful, this will show package information.

If it says the package is not found, run:

```bash
pip install ultralytics
```

---

## 12. First YOLO test

Create a file called:

```bash
test_yolo.py
```

Put this code inside it:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("https://ultralytics.com/images/bus.jpg", save=True)

print("YOLO setup works")
```

Then run:

```bash
python test_yolo.py
```

If everything is working correctly, YOLO will:

* download the small pretrained model
* run detection on the test image
* save the output in a `runs/` folder
* print:

```bash
YOLO setup works
```

This confirms that the setup is working properly.

---

## 13. Suggested project structure

A simple and clean structure for the technical side of this project is shown below:

```bash
ICT-Capstone-Project-1-2/
│
├── .venv/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw_videos/
│   ├── extracted_frames/
│   └── labels/
├── scripts/
│   ├── test_yolo.py
│   ├── extract_frames.py
│   └── predict_video.py
└── runs/
```

### Notes

* `.venv/` should stay local and should not be pushed to GitHub
* `runs/` stores YOLO outputs and should also be ignored
* `data/` stores raw videos, extracted frames, and labels
* `scripts/` stores Python scripts for testing and processing

---

## 14. Recreating the setup on another machine

If another team member clones the repository, they can repeat the setup with these commands:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics
```

If a `requirements.txt` file is added later, dependencies can also be installed using:

```bash
pip install -r requirements.txt
```

---

## 15. Current setup status

The project setup currently includes:

* VS Code running in WSL Ubuntu
* a local `.venv` created inside the repository
* the `.venv` environment activated successfully
* a `.gitignore` file created
* Git confirmed that `.venv` is not being tracked

This means the basic Python and YOLO environment setup is ready.

---

## 16. Next steps

Now that the environment setup is complete, the next technical tasks are:

1. run the YOLO smoke test
2. create a frame extraction script for sample videos
3. test YOLO on extracted frames
4. decide which roadway classes to use for the first prototype
5. prepare a small annotated dataset if needed

These steps will help move the project from setup into actual prototype development.

```

If you want, I can also make this into a **more professional capstone README** with sections like **Project Overview**, **Team**, **Setup**, **How to Run**, and **Next Milestone**.
```
