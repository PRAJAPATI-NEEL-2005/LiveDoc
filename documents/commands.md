# Quick Start & Commands Reference

This is a cheat sheet to help you quickly remember how to start working on the project, especially if you step away for a few days!

---

### 1. Activating the Environment (Always do this first!)
Before writing code or running the project, you **must** activate the Conda environment so Python knows where all your libraries (like OpenCV and YOLO) are installed.

Open your terminal in the `LiveDoc` folder and type:
```bash
conda activate livedoc
```

### 2. Installing or Updating Libraries
If Neel adds a new library, or you need to re-install everything, run this (make sure your environment is activated first!):
```bash
pip install -r requirements.txt
```

### 3. Running & Testing Your Code
Here are the commands to test the code we have written so far.

**To test the Live Camera Feed (Viraj's Task):**
```bash
python src/camera_manager.py
```
*(Press 'q' on your keyboard to close the camera window when you are done).*

**To run the main application (Once `main.py` is built):**
```bash
python main.py
```

---

### 4. Git Commands Cheat Sheet
When you finish a coding session and want to save your work to GitHub:

```bash
# 1. See what files you changed
git status

# 2. Add all your changes
git add .

# 3. Save the changes with a message
git commit -m "Describe what you did here"

# 4. Upload your changes to your branch
git push origin Viraj
```
