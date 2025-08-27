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

# Sharding setup:
	
Create required directories for data and configurations:
mkdir -p shard-demo/configsrv shard-demo/configsrv1 shard-demo/configsrv2 shard-demo/shardrep1 shard-demo/shardrep2 shard-demo/shardrep3 shard-demo/shard2rep1 shard-demo/shard2rep2 shard demo/shard2rep3 

Config server:
		nohup mongod --configsvr  --port 28041 --bind_ip localhost --replSet config_repl --dbpath /home/neeraj/shard-demo/configsrv &
		
		nohup mongod --configsvr  --port 28042 --bind_ip localhost --replSet config_repl --dbpath /home/neeraj/shard-demo/configsrv1 &
		 
		nohup mongod --configsvr  --port 28043 --bind_ip localhost --replSet config_repl --dbpath /home/neeraj/shard-demo/configsrv2 &
		
		mongosh --host localhost  --port 28041
		
		rsconf = {
				  _id: "config_repl",
				  members: [
					{
					 _id: 0,
					 host: "localhost:28041"
					},
					{
					 _id: 1,
					 host: "localhost:28042"
					},
					{
					 _id: 2,
					 host: "localhost:28043"
					}
				   ]
				}
		
		rs.initiate(rsconf)
		rs.status()
					
			
		Shard server:
			nohup mongod --shardsvr --port 28081 --bind_ip localhost --replSet shard_repl --dbpath /home/neeraj/shard-demo/shardrep1 &
		
			nohup mongod --shardsvr --port 28082 --bind_ip localhost --replSet shard_repl --dbpath /home/neeraj/shard-demo/shardrep2 &
			
			nohup mongod --shardsvr --port 28083 --bind_ip localhost --replSet shard_repl --dbpath /home/neeraj/shard-demo/shardrep3 &
			
			mongosh --host localhost  --port 28081
			
			rsconf = {
				  _id: "shard_repl",
				  members: [
					{
					 _id: 0,
					 host: "localhost:28081"
					},
					{
					 _id: 1,
					 host: "localhost:28082"
					},
					{
					 _id: 2,
					 host: "localhost:28083"
					}
				   ]
				}
		
			rs.initiate(rsconf)
			rs.status()
			
			
			nohup mongod --shardsvr --port 29081 --bind_ip localhost --replSet shard2_repl --dbpath /home/neeraj/shard-demo/shard2rep1 &
		
			nohup mongod --shardsvr --port 29082 --bind_ip localhost --replSet shard2_repl --dbpath /home/neeraj/shard-demo/shard2rep2 &
			
			nohup mongod --shardsvr --port 29083 --bind_ip localhost --replSet shard2_repl --dbpath /home/neeraj/shard-demo/shard2rep3 &
			
			mongosh --host localhost  --port 29081
			
			rsconf = {
				  _id: "shard2_repl",
				  members: [
					{
					 _id: 0,
					 host: "localhost:29081"
					},
					{
					 _id: 1,
					 host: "localhost:29082"
					},
					{
					 _id: 2,
					 host: "localhost:29083"
					}
				   ]
				}
		
			rs.initiate(rsconf)
			rs.status()
			
		MongoS:
			nohup mongos --configdb config_repl/localhost:28041,localhost:28042,localhost:28043 --bind_ip localhost &
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
