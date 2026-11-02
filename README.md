# Talk with Your Hands: AI Sign Language Project

An interactive school-expo project with two modes:

1. **Gesture to Character** — Show an ASL hand sign to the webcam; the program predicts the letter.
2. **Character to Gesture** — Type a name; the program displays the ASL sign for each letter.

## 1. Install Python packages

Open Command Prompt (Windows) or Terminal (Mac/Linux) inside this folder and run:

```bash
pip install opencv-python mediapipe scikit-learn numpy pandas joblib
```

## 2. Add sign images

Put one ASL alphabet image for every letter in the `sign_images` folder:

```text
sign_images/A.png
sign_images/B.png
sign_images/C.png
...
sign_images/Z.png
```

These images are used by **Mode 2: Name to Sign Language**.

## 3. Run the project

```bash
python main.py
```

Then choose:

```text
1. Detect sign from hand gesture
2. Convert a name into sign language
3. Collect new training data
4. Train the model
5. Exit
```

## 4. First-time setup: train your own model

The gesture-recognition model needs examples from students.

1. Choose option `3` or run:

   ```bash
   python collect_data.py
   ```

2. Show one ASL letter sign to the webcam.
3. Press the same letter key on the keyboard, for example `A`, to save it.
4. Collect at least **30–50 examples per letter**.
5. Press `ESC` to stop.
6. Choose option `4` or run:

   ```bash
   python train_model.py
   ```

This creates:

```text
sign_model.pkl
label_map.json
```

7. Choose option `1` or run:

   ```bash
   python sign_detector.py
   ```

## 5. Controls in gesture detector

| Key | Action |
|---|---|
| `A-Z` | Save current hand sign while collecting data |
| `C` | Clear the detected word |
| `SPACE` | Add a space |
| `ESC` | Quit |

## 6. Important notes

- Use good lighting and keep one hand clearly visible.
- Collect signs from different students to improve accuracy.
- Static ASL letters work best. Letters **J** and **Z** involve motion, so they may need special handling.
- This project recognizes individual alphabet signs (fingerspelling), not full sign-language sentences.

## 7. Expo demonstration script

Say:

> “Our project uses computer vision to track 21 points on a hand. A machine-learning model learns patterns from these points and predicts the sign. In the second mode, students can type their name and learn how to sign each letter.”

## Project structure

```text
sign_language_project/
├── main.py
├── collect_data.py
├── train_model.py
├── sign_detector.py
├── name_to_sign.py
├── sign_images/
├── hand_data.csv          # created after data collection
├── sign_model.pkl         # created after training
└── label_map.json         # created after training
```
