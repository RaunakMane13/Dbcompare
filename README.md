# DBCompare  
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)  
[![Node](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/)  

A web-based tool to **upload, shard, and benchmark** datasets across **MySQL**, **MongoDB**.  
Includes file previews, visualizations, query performance comparisons, and inline updates.  
Powered by **Flask + Celery** for backend processing and **React** for the frontend.  

---

## 📑 Table of Contents  
- [Installation](#installation)  
- [Usage](#usage)  
- [Contributing](#contributing)  
- [License](#license)  
- [Credits](#credits)  

---

## ⚙️ Installation  

**Prerequisites**  
- Python 3.10+  
- Node.js 18+  
- MySQL, MongoDB  
- Redis or RabbitMQ (Celery broker)  

**Setup**  
```bash
# Clone the repository
git clone https://github.com/RaunakMane13/Dbcompare.git
cd Dbcompare

# Backend
pip install -r requirements.txt
python main.py

# Celery worker
celery -A celery_app.celery worker -l INFO

# Frontend
npm install
npm run build
```

## 🚀 Usage  

- **Login/Signup** to unlock the demo page.  
- **Upload CSV/JSON** → Select databases for insertion.  
- **Preview & Explore** → View first rows of uploaded files.  
- **Visualize** → Generate correlation, feature importance, and histograms.  
- **Shard & Benchmark** → Compare query performance on sharded vs unsharded tables/collections.  
- **Edit Records** → Paginated inline editor with change tracking.  

---

## 🙏 Tech Stack 

Built with:  
- **Flask**, **Celery**, **PyMySQL**, **PyMongo**
- **React**, **Tailwind-style UI**  
- **Pandas**, **Scikit-learn**, **Matplotlib**, **Seaborn**  
