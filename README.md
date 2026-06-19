# Air Canvas Drawing 🎨👋


https://github.com/user-attachments/assets/0848ebf9-4d0d-4a08-8361-533dd1d4dfcb


A fun, interactive computer vision application that turns your webcam into a digital canvas. By using real-time hand gesture tracking, you can paint, change colors, and erase lines directly in the air without touching your mouse or keyboard.

This project was built as a learning milestone, combining concepts from a YouTube community tutorial with custom programming logic and real-time optimization developed alongside an AI collaborator.

---

## 🚀 Features

* **Fluid In-Air Drawing:** Lift your index finger to draw continuous colorful lines perfectly tracked to your position.
* **Hover & Selection Mode:** Raise both your index and middle fingers to toggle "Selection Mode" where you can hover over the UI layout without accidentally drawing.
* **Dynamic Color Palette:** Interactive on-screen buttons to switch instantly between Green, Red, White, and Yellow.
* **Smart Eraser Tool:** A dedicated 50px thick eraser block that expands dynamically and displays a hollow target ring around your finger for precision wiping.
* **Hardware Optimizations:** Fine-tuned AI model confidence boundaries to ensure stable tracking even if you step back from the camera or shift your environment lighting.
* **Quick Controls:** Press `C` to clear the canvas instantly or `Q` to close the application safely.

---

## 🛠️ The Tech Stack

* **Language:** Python
* **Webcam & Frame Processing:** OpenCV (`cv2`)
* **Machine Learning Pipeline:** Google MediaPipe (HandLandmarker Task Model)
* **Matrix Operations:** NumPy

---

## 💡 How It Works Under the Hood

1. **Computer Vision Input:** OpenCV captures frames from the webcam, mirrors the view for natural navigation, and extracts the frame pixel dimensions dynamically.
2. **AI Landmark Extraction:** The raw frame passes locally into Google's MediaPipe model, tracking 21 structural coordinates across your hand skeleton.
3. **Gesture Logic Math:** The script analyzes the vertical coordinates of your fingertips relative to your knuckles to detect states (`Y_tip < Y_knuckle` confirms a finger is raised).
4. **Advanced Digital Masking:** To keep lines crisp, the script creates a black binary mask of your drawings, punches transparent cutout holes into the live webcam video array, and maps the solid colors perfectly over the feed.

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/FreezerTheGod/Air-Canvas-Drawing.git](https://github.com/FreezerTheGod/Air-Canvas-Drawing.git)
cd Air-Canvas-Drawing
