# 🐾 Smart Re-Identification System for Stray Cats Post-TNR Program

A mobile-first application built with React Native and integrated with an image-based cat re-identification system to prevent redundant medical treatments of stray cats. This system is designed to support volunteers, animal hospitals, and TNR organizations, especially in the Kansai region of Japan. This project is made as a part of a Project-Based-Learning Course which spans over 15 weeks.

---

## 📌 Project Objective

To streamline the Trap-Neuter-Return (TNR) process and reduce unnecessary hospital visits for stray cats by enabling users to:

- Identify previously captured and treated cats using AI-based image matching.
- View and manage cat profiles with medical histories.
- Coordinate efficiently between caretakers, hospitals, and organizations.
- Ensure data integrity, usability, and privacy compliance.

---

## 🔧 Features

### 🐱 Cat Re-Identification
- Upload or capture a photo of a stray cat to check for prior registration.
- AI provides a **confidence score** and match result (high, moderate, or low).
- Feedback system for users to report false matches.

### 👤 Account Management
- Role-based access for Volunteers, Animal Hospitals, and Administrators.
- Profile creation, editing, verification, and deletion supported.
- Password recovery and secure authentication mechanisms.

### 🏥 Medical Record System
- View and update cat profiles: age, gender, vaccination status, and treatment history.
- Hospitals can log surgeries and medical interventions.
- Tagging system (e.g., neutered, under treatment, released).

### 📸 Image Submission Workflow
- Supports photo capture via device camera or gallery upload.
- Validates format, size, and resolution (≥ 1280x720, ≤ 5MB).
- Mobile and offline-capable submission process.

### 📊 Administration & Analytics
- System dashboards for match statistics and cat counts.
- Access control, audit logs, and activity tracking.
- Re-ID match reviews and visualization of trends.

---

## 👥 Target Users

| Role            | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| Volunteers      | Submit cat sightings, upload images, and help reduce redundant captures.   |
| Animal Hospitals| Update medical histories, create/edit profiles, and manage treatment logs. |
| Administrators  | Oversee system users, manage content, and monitor analytics.               |

---

## 🗺️ Deployment Scope

- Initial deployment in **Kansai Region, Japan**.
- Supports **up to 1000 volunteers** and **3 animal hospitals**.
- Mobile-first design compatible with Android and iOS (React Native).
- Backend support via Flask (Python) and TensorFlow-based re-ID model.

---

## 🚀 Tech Stack

| Layer       | Technology                       |
|-------------|----------------------------------|
| Frontend    | React Native, Expo               |
| Backend     | Python (Flask)                   |
| AI Model    | TensorFlow (Cat Re-Identification)|
| Storage     | Encrypted Cloud Storage          |
| Auth        | Secure Email + Role-Based Access |
| Analytics   | Server-side statistics, charts   |

---

## ✅ Key Functional Requirements

- Account creation with verification (FR-1, FR-4)
- Photo upload and Re-ID results with confidence scores (FR-7, FR-8)
- View, add, edit, delete cat profiles (FR-9, FR-10, FR-11)
- Role-based access and logging (FR-15)
- Admin analytics and match management (FR-13, FR-14)

---

## 🔐 Non-Functional Highlights

- **Mobile-first design** with responsive layouts and offline sync.
- **Visual accessibility** (WCAG 2.2 AA) and performance optimizations.
- **Secure session management** and encryption (TLS, AES-256).
- **Data privacy** and GDPR/Japanese compliance.
- **Disaster resilience** and eco-friendly cloud architecture.

---

## 🧪 Testing & Validation

- Unit and integration tests with 80%+ code coverage.
- Secure input validation and audit logging.
- AI model validation using confidence scoring and match tiers.
- Supports 100 concurrent users and 5s sync guarantees.

---

## 📦 Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/your-org/cat-reid-app.git
   cd cat-reid-app
   ```

2. **Backend Setup (Flask)**
   - Install Python dependencies:
     ```sh
     pip install flask ultralytics opencv-python
     ```
   - Register known cats:
     ```sh
     python ai_model/register_known_cats.py
     ```
   - Start the server:
     ```sh
     python serve.py
     ```
   - The server will run on `http://<your-ip>:5000`.

3. **Mobile App Setup (Expo)**
   - Install Node.js (v18+) and Expo CLI:
     ```sh
     npm install -g expo-cli
     ```
   - Install dependencies:
     ```sh
     cd PBL3Expo
     npm install
     npx expo install expo-camera expo-image-picker
     ```
   - Start the app:
     ```sh
     npm start
     ```
   - Run on your phone:
     1. Install **Expo Go** from the App Store/Google Play.
     2. Connect your phone and computer to the same WiFi.
     3. Scan the QR code from the terminal/browser.

---

## ⚙️ Configuration

- **API URL:** Edit `PBL3Expo/config/api.ts` and set `BASE_URL` to your computer's IP (e.g., `http://192.168.1.100:5000`).
- **Known Cats:** Place images in `images/known_cats/` and run the registration script.

---

## 📱 Usage

1. **Open the mobile app** (on phone or emulator)
2. **Take a photo** or **choose from gallery**
3. **Tap 'Identify Cat'**
4. **View results** (match, no match, or error)
5. **Configure server** in the Explore tab if needed

---

## 🛠️ Troubleshooting

- **Cannot connect to server:**
  - Ensure both devices are on the same WiFi
  - Check the Flask server is running
  - Update the IP in `api.ts`
- **Camera/gallery permissions denied:**
  - Enable permissions in device settings
- **App not loading:**
  - Restart Expo, clear cache: `npx expo start --clear`

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License. 