## 🚀 Getting Started

Follow these steps to set up and run **Lip Cipher** on your local machine:

---

### 1. Clone the repository

```bash
git clone https://github.com/your-username/lip-cipher.git
cd lip-cipher
```

> Make sure you're inside the project root before running anything.

---

### 2. (Optional) Create and activate a virtual environment

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the Streamlit App

```bash
streamlit run src/streamlitapp.py
```

This will launch the web app in your browser at [http://localhost:8501](http://localhost:8501).

---

### 5. Project Structure

```
lip-cipher/
├── data/                  # Video data and alignment files
├── src/                  
│   ├── model_util.py      # Handles model loading and predictions
│   ├── utils.py           # Preprocessing utilities
│   └── streamlitapp.py    # Main Streamlit frontend   
├── requirements.txt      
├── README.md             
└── .gitignore             
```

---

### 6. Notes

- Make sure your **Python version is 3.9–3.11**.
- If `streamlit` isn't found, try running:

```bash
python -m streamlit run src/streamlitapp.py
```
