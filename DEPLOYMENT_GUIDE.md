# Project Meswak: Web Server Hosting & Cloud Connection Guide

This guide explains how to run the AI engine locally, connect phones on your local network without any paid servers, host the backend API completely free on cloud platforms (Render, Hugging Face, Railway), and use GitHub Pages.

---

## 1. Local Testing & Phone Connection (Zero Cloud Server Required)

You can run the models on your PC and use your mobile phone as a live frontend client over your local Wi-Fi network.

### Step 1: Start the Backend Server on your PC
```powershell
python run_server.py
```
The server will bind to `0.0.0.0:8000` (accessible from any device on your local Wi-Fi).

### Step 2: Find Your PC's Local IP Address
Open PowerShell and run:
```powershell
ipconfig
```
Look for **IPv4 Address** (e.g., `192.168.1.15` or `192.168.29.102`).

### Step 3: Open on Your Phone
On your mobile phone's browser (Chrome / Safari), open:
* **Citizen Mobile App:** `http://<YOUR_PC_IP>:8000/citizen` (e.g., `http://192.168.1.15:8000/citizen`)
* **Government Command Center:** `http://<YOUR_PC_IP>:8000/gov`

> **Tip:** In Chrome or Safari on your phone, tap **Share / Menu &rarr; "Add to Home Screen"**. This installs the Citizen App as a standalone fullscreen app on your phone with touch gestures!

---

## 2. Free Cloud Hosting Options

If you want the API to be accessible over the public internet from anywhere in the world without keeping your PC turned on:

### Option A: Render.com (Recommended - 100% Free Web Service)
1. Push your `project_meswak` folder to a GitHub repository.
2. Sign up at [Render.com](https://render.com) (Free tier).
3. Click **New + &rarr; Web Service** and connect your GitHub repository.
4. Select **Environment: Python 3** (or Docker).
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `python run_server.py`
5. Render will build and launch your service with a free public HTTPS URL:
   `https://project-meswak.onrender.com`
6. Open your Citizen App or Government Command Center, click ⚙️ **Server Settings**, and enter `https://project-meswak.onrender.com` to connect!

---

### Option B: Hugging Face Spaces (Free Cloud Docker CPU)
1. Create a free account at [huggingface.co](https://huggingface.co).
2. Create a **New Space**, choose **Docker** SDK, and name it `meswak-aqi-engine`.
3. Upload the project files (`Dockerfile`, `requirements.txt`, `app/`, `train.py`, `run_server.py`).
4. Hugging Face will automatically build the container and provide a permanent free HTTPS endpoint:
   `https://<username>-meswak-aqi-engine.hf.space`

---

### Option C: Railway.app / Fly.io
1. Install the CLI or link your GitHub repo to Railway / Fly.io.
2. Railway detects the `Dockerfile` and deploys automatically in under 2 minutes.

---

## 3. Using GitHub Pages for Frontend

If you want to host the frontend HTML/JS files on **GitHub Pages**:
1. In your GitHub repo, enable GitHub Pages from the `app/static/` folder or `docs/` folder.
2. When users open the GitHub Pages URL (e.g. `https://yourname.github.io/project_meswak/citizen/`):
3. Tap ⚙️ **Server Settings** on the top right.
4. Enter your backend cloud URL (e.g. `https://project-meswak.onrender.com`).
5. The frontend will now communicate directly with your remote cloud server!

---

## 4. Packaging into Native Android APK (Optional)

If you want to compile the Citizen App into an installable `.apk` file:
1. Install [Capacitor](https://capacitorjs.com):
   ```bash
   npm init @capacitor/app
   npx cap add android
   ```
2. Set `webDir` to `app/static/citizen`.
3. Build the Android project in Android Studio &rarr; Generate Signed APK.

