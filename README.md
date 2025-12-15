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

# MySql setup
sudo apt update && sudo apt install -y mysql-server
sudo systemctl enable --now mysql
mysql --version

CREATE USER 'dbcompare'@'%' IDENTIFIED BY '*password*';
GRANT ALL PRIVILEGES ON *.* TO 'dbcompare'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;

# MongoDB Sharding setup
	
# Create required directories for data and configurations:
mkdir -p shard-demo/configsrv shard-demo/configsrv1 shard-demo/configsrv2 shard-demo/shardrep1 shard-demo/shardrep2 shard-demo/shardrep3 shard-demo/shard2rep1 shard-demo/shard2rep2 shard demo/shard2rep3 

# Config server:
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
					
			
		# Shard server:
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
			
		# MongoS:
			nohup mongos --configdb config_repl/localhost:28041,localhost:28042,localhost:28043 --bind_ip localhost &

#MySQL NDB Setup

1. Environment Preparation (All Nodes)
sudo apt update
sudo apt install -y wget curl gnupg lsb-release ufw
# Firewall rules for full NDB communication
sudo ufw allow 22/tcp
sudo ufw allow from 172.16.1.0/24 to any port 3306 proto tcp
sudo ufw allow from 172.16.1.0/24 to any port 1186 proto tcp
sudo ufw allow from 172.16.1.0/24 to any port 2202 proto tcp
sudo ufw enable
sudo ufw status verbose
# Connectivity validation
ping -c 3 172.16.1.125
ping -c 3 172.16.1.217
ping -c 3 172.16.1.192

2. Install MySQL APT Repository (All Nodes)
cd /tmp
wget https://dev.mysql.com/get/mysql-apt-config.deb
sudo dpkg -i mysql-apt-config.deb
sudo apt update
# If config selector appears:
# Select: MySQL Server & Cluster
# Select: mysql-cluster-8.0

3. Install Cluster Components by Role
# Management + SQL Node (172.16.1.125)
sudo apt install -y mysql-cluster-community-management-server mysql-server
# Data Node 1 (172.16.1.217)
sudo apt install -y mysql-cluster-community-data-node
# Data Node 2 (172.16.1.192)
sudo apt install -y mysql-cluster-community-data-node

4. Management Node Configuration (172.16.1.125)
sudo mkdir -p /var/lib/mysql-cluster
sudo nano /var/lib/mysql-cluster/config.ini
# Paste the following:
[ndbd default]
NoOfReplicas=2
DataMemory=1024M
IndexMemory=512M
[tcp default]
portnumber=2202
[ndb_mgmd]
hostname=172.16.1.125
datadir=/var/lib/mysql-cluster
[ndbd]
hostname=172.16.1.217
datadir=/usr/local/mysql/data
[ndbd]
hostname=172.16.1.192
datadir=/usr/local/mysql/data
[mysqld]
hostname=172.16.1.125
# Management node connect string
sudo nano /etc/mysql/mysql.conf.d/ndb_mgmd.cnf
[mysql_cluster]
ndb-connectstring=172.16.1.125
# Start the management server
sudo ndb_mgmd -f /var/lib/mysql-cluster/config.ini --initial
# Verify
ndb_mgm -e "show"

5. Data Node Configuration (217 and 192)
# Create directories
sudo mkdir -p /usr/local/mysql/data
sudo chown -R mysql:mysql /usr/local/mysql
# Configure NDB
sudo nano /etc/mysql/mysql.conf.d/ndbd.cnf
[mysql_cluster]
ndb-connectstring=172.16.1.125
# Start data node
sudo ndbmtd --initial
# Verify from mgmt node
ndb_mgm -e "show"

6. SQL Node Configuration (172.16.1.125)
sudo nano /etc/mysql/mysql.conf.d/ndb_mysqld.cnf
[mysqld]
ndbcluster
ndb-connectstring=172.16.1.125
sudo systemctl restart mysql
mysql -u root -p -e "SHOW ENGINES LIKE 'NDB%';"
ndb_mgm -e "show"
```

## 🚀 Usage  

- **Login/Signup** → user authentication/authorization.  
- **Upload CSV/JSON** → Select databases for insertion.  
- **Preview & Explore** → Preview of uploaded files.  
- **Visualize** → Generate correlation, feature importance, and histograms.  
- **Shard & Benchmark** → Compare query performance on sharded vs unsharded tables/collections.  
- **Edit Records** → Paginated inline editor with change tracking.  

---

## 🙏 Tech Stack 

Built with:  
- **Flask**, **Celery**, **PyMySQL**, **PyMongo**
- **React**, **Tailwind-style UI**  
- **Pandas**, **Scikit-learn**, **Matplotlib**, **Seaborn**  

## 📁 Directory Structure

- **backend/** — Flask app, Celery tasks, config, and API blueprints.
- **frontend/** — React source (served by Flask).  
  - **src/components/** — Reusable UI (e.g., `Nav.jsx`).  
  - **src/pages/** — Route-level pages (e.g., `HomePage.js`, `Visualize.js`, `ProjectOverview.js`, etc.).  
- **legacy/** — **Old redundant files** kept for reference; not imported by the app.
- **requirements.txt** — Python dependencies for backend.
