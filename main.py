from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_session import Session
import os
import datetime as _dt
import csv
import json
import pymysql
from dateutil import parser as date_parser
import math
from pymysql.err import OperationalError
from pymongo import MongoClient, ASCENDING
from bson.objectid import ObjectId
from neo4j import GraphDatabase
import bcrypt
import jwt
from flask import session
from bson import ObjectId
import pandas as pd
from pandas import json_normalize
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import send_file
from celery_app import celery
from flask import jsonify
from celery.result import AsyncResult
import uuid
import shutil
import random
from werkzeug.utils import secure_filename
import time
from bson.min_key import MinKey
from bson.max_key import MaxKey
from decimal import Decimal
import re # Import for regex in query parsing
import itertools

build_dir = '/home/student/my-app/build'  # 🔧 Change path if your build folder is elsewhere
app = Flask(__name__, static_folder=build_dir, static_url_path='')
matplotlib.use('Agg')  # For headless environments

# 🧭 Set exact paths like in server.py
DATA_DIR = '/home/student/my-app/src/uploads'


# Debug prints to verify the frontend is reachable
print("Static folder:", app.static_folder)
print("index.html exists:", os.path.exists(os.path.join(app.static_folder, 'index.html')))


# --------------------
# CORS Configuration
# --------------------
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "https://dbcompare.webdev.gccis.rit.edu",
                "http://172.16.1.254:5000",
                "http://localhost:5000",
                "http://localhost:3000",
            ]
        },
        r"/file-preview/*": {
            "origins": [
                "https://dbcompare.webdev.gccis.rit.edu",
                "http://172.16.1.254:5000",
                "http://localhost:5000",
                "http://localhost:3000",
            ]
        },
    },
    supports_credentials=True,
)


# --------------------
# Session Setup
# --------------------
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = _dt.timedelta(minutes=30)  # <- change as you like
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True

Session(app)

# --------------------
# File Upload Setup
# --------------------
STATIC_DIR = '/home/student/my-app/src/static'
UPLOAD_FOLDER = '/home/student/my-app/src/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Create static dir if not present
os.makedirs(STATIC_DIR, exist_ok=True)

@celery.task(name='main.process_file_task')
def process_file_task(filepath, task_type, column=None):
    import uuid
    visualizations = {}

    filename = os.path.basename(filepath)
    if not os.path.exists(filepath):
        return {'error': 'File not found'}

    MAX_ROWS = 100_000
    FILESIZE_LIMIT = 1000 * 1024**2  # 100 MB

    if filename.lower().endswith('.csv'):
        if os.path.getsize(filepath) > FILESIZE_LIMIT:
            # read only first MAX_ROWS rows (or you could sample randomly via skiprows)
            data = pd.read_csv(filepath, nrows=MAX_ROWS)
        else:
            data = pd.read_csv(filepath)

    elif filename.lower().endswith('.json'):
        # load up to MAX_ROWS records
        with open(filepath, 'r', encoding='utf-8') as f:
            txt = f.read().strip()
        if txt.startswith('['):
            records = json.loads(txt)
        else:
            records = [json.loads(line) for line in txt.splitlines() if line]
        if len(records) > MAX_ROWS:
            records = random.sample(records, MAX_ROWS)
        data = pd.DataFrame(records)
    else:
        return {'error': 'Unsupported file type'}

    # --- Correlation Heatmap ---
    numerical_cols = data.select_dtypes(include='number').columns.tolist()
    valid_cols = [col for col in numerical_cols if 'id' not in col.lower() and 'code' not in col.lower() and 'number' not in col.lower()]
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(data[numerical_cols].corr(), annot=True, cmap='coolwarm')
        heatmap_name = f'correlation_heatmap_{uuid.uuid4()}.png'
        corr_path = os.path.join(STATIC_DIR, heatmap_name)
        plt.title('Correlation Heatmap')
        plt.savefig(corr_path)
        plt.close()
        visualizations['correlation_heatmap'] = f'static/{heatmap_name}'

    # --- Feature Importance ---
    # --- Feature Importance (only if we can infer a numeric target) ---
    if valid_cols:
        try:
            # auto-create a binary target from the first valid numeric column
            if 'target' not in data.columns:
                data['target'] = data[valid_cols[0]] > data[valid_cols[0]].mean()

            X = data.drop(columns=['target'], errors='ignore')
            y = data['target']

            # Label-encode any remaining objects
            X_encoded = X.copy()
            for col in X_encoded.select_dtypes(include='object').columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

            model = RandomForestClassifier(random_state=42)
            model.fit(X_encoded, y)

            importance_df = pd.DataFrame({
                'Feature': X_encoded.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature')
            importance_name = f'feature_importance_{uuid.uuid4()}.png'
            importance_path = os.path.join(STATIC_DIR, importance_name)
            plt.title('Feature Importance')
            plt.savefig(importance_path)
            plt.close()

            visualizations['feature_importance'] = f'static/{importance_name}'
            visualizations['importance_table'] = importance_df.to_dict(orient='records')
        except Exception as e:
            visualizations['feature_importance_error'] = str(e)
    else:
        visualizations['feature_importance'] = None
        visualizations['importance_table'] = []

    # --- Histogram for selected column ---
    if column and column in data.columns:
        series = data[column]
        plt.figure(figsize=(10, 6))

        # NUMERIC (including bool cast→int)
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            if pd.api.types.is_bool_dtype(series):
                series = series.astype(int)
            values = series.dropna()
            plt.hist(values, bins=30)

        # OTHERWISE: treat as categorical/dict—stringify & bar‐plot top 10
        else:
            # Convert unhashable objects (dicts, lists) to strings
            cat = series.dropna().map(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else str(v))
            top = cat.value_counts().nlargest(10)
            plt.barh(top.index, top.values)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        hist_filename = f"histogram_{column}_{uuid.uuid4()}.png"
        hist_path = os.path.join(STATIC_DIR, hist_filename)
        plt.savefig(hist_path)
        plt.close()
        visualizations['histogram'] = f'static/{hist_filename}'

    return {'visualizations': visualizations}

def serialize_mongo(obj):
    if isinstance(obj, list):
        return [serialize_mongo(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: serialize_mongo(v) for k, v in obj.items()}
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, _dt.datetime): # Use _dt.datetime for datetime objects
        return obj.isoformat()
    return obj

def serialize_json_safe(obj):
    """
    Recursively convert anything Flask can’t JSON-encode (datetime, date,
    Decimal, ObjectId, etc.) into plain primitives.
    """
    if isinstance(obj, list):
        return [serialize_json_safe(i) for i in obj]
    if isinstance(obj, dict):
        return {k: serialize_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, ObjectId):            # already imported earlier
        return str(obj)
    if isinstance(obj, float):
       return None if math.isnan(obj) else obj
    return obj

def mysql_insert_chunk(chunk_df, db_name, table_name):
    import pymysql

    conn = pymysql.connect(
        host='172.16.1.125',
        user='root',
        password='StrongPassword123!',
        db=db_name
    )
    cursor = conn.cursor()

    # Create table if not exists using inferred column types
    cols_and_types = []
    for col in chunk_df.columns:
        sample = chunk_df[col].dropna()
        sample_value = str(sample.iloc[0]) if not sample.empty else ''
        dtype = infer_data_type(sample_value)
        cols_and_types.append(f"`{col}` {dtype}")
    create_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({', '.join(cols_and_types)})"
    cursor.execute(create_sql)

    # Insert data
    cols = ', '.join([f"`{col}`" for col in chunk_df.columns])
    placeholders = ', '.join(['%s'] * len(chunk_df.columns))
    insert_sql = f"INSERT INTO `{table_name}` ({cols}) VALUES ({placeholders})"

    for _, row in chunk_df.iterrows():
        sanitized_row = []
        for val in row:
            # nulls → None
            if pd.isna(val):
                sanitized_row.append(None)
            # nested JSON objects/arrays → serialize to string
            elif isinstance(val, (dict, list)):
                sanitized_row.append(json.dumps(val))
            # UTC timestamps in strings → normalize to MySQL DATETIME
            elif isinstance(val, str) and 'UTC' in val:
                try:
                    dt = date_parser.parse(val)
                    sanitized_row.append(dt.strftime('%Y-%m-%d %H:%M:%S'))
                except:
                    sanitized_row.append(val)
            else:
                sanitized_row.append(val)
        cursor.execute(insert_sql, tuple(sanitized_row))


    conn.commit()
    cursor.close()
    conn.close()




def mongodb_insert_chunk(chunk_df, collection_name, db_name):
    import pymongo
    from pymongo.errors import BulkWriteError, ServerSelectionTimeoutError, AutoReconnect
    import time, json

    # ── create one long-lived client with no per-socket timeouts ───────────
    client = pymongo.MongoClient(
        "mongodb://172.16.1.126:27017/",
        serverSelectionTimeoutMS=60000,
        connectTimeoutMS=60000,
        socketTimeoutMS=0
    )
    db = client[db_name]
    collection = db[collection_name]

    # sanitize nested structures into JSON-serializable dicts
    records = []
    for rec in chunk_df.to_dict(orient='records'):
        sanitized = {}
        for k, v in rec.items():
            if isinstance(v, (dict, list)):
                sanitized[k] = json.dumps(v)
            else:
                sanitized[k] = v
        records.append(sanitized)

    # use smaller batches for huge uploads
    batch_size   = 100
    max_retries  = 5

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        for attempt in range(1, max_retries + 1):
            try:
                collection.insert_many(batch, ordered=False)
                break
            except (ServerSelectionTimeoutError, AutoReconnect) as se:
                print(f"⚠️ MongoDB timeout or reconnect error for batch {i}-{i+batch_size} (try {attempt}): {se}")
                time.sleep(5)
            except BulkWriteError as bwe:
                print(f"⚠️ Bulk write error for batch {i}-{i+batch_size}: {bwe.details}")
                break  # don’t retry on write errors
            except Exception as e:
                print(f"⚠️ Insert failed for batch {i}-{i+batch_size} (try {attempt}): {e}")
                time.sleep(3)
        else:
            print(f"❌ Giving up on batch {i}-{i+batch_size} after {max_retries} retries.")
    # close the client when done
    # client.close()



def neo4j_insert_chunk(chunk_df, label_name):
    from neo4j import GraphDatabase
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "12345"))
    with driver.session() as session:
        for record in chunk_df.to_dict(orient='records'):
            props = ', '.join([f"{k}: ${k}" for k in record])
            query = f"CREATE (n:{label_name} {{{props}}})"
            session.run(query, **record)
    driver.close()

@celery.task(name="main.process_upload_task")
def process_upload_task(filepath, databases, database_name, extension, session_id):
    print("Processing upload task for:", filepath)
    start_time = time.time()

    CHUNK_SIZE = 1000
    uploads_summary = []

    if 'mysql' in databases:
        try:
            conn = pymysql.connect(host='172.16.1.125', user='root', password='StrongPassword123!')
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database_name}`")
            conn.commit()
            conn.close()
        except Exception as e:
            return {'error': f'Failed to create MySQL database: {str(e)}'}

    # unified CSV / JSON chunking
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.csv':
        reader = pd.read_csv(filepath, chunksize=CHUNK_SIZE, dtype=str)

    elif ext == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
           text = f.read().strip()

        # detect array vs. newline-delimited JSON
        if text.startswith('['):
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line]

        # flatten nested fields into columns
        df = pd.DataFrame(records)

        # break into chunks
        reader = (
            df[i : i + CHUNK_SIZE]
            for i in range(0, len(df), CHUNK_SIZE)
        )

    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    table_name = os.path.splitext(os.path.basename(filepath))[0]

    for chunk in reader:
        for db in databases:
            if db == 'mysql':
                mysql_insert_chunk(chunk, database_name, table_name)
            elif db == 'mongodb':
                mongodb_insert_chunk(chunk, table_name, database_name)
            elif db == 'neo4j':
                neo4j_insert_chunk(chunk, table_name)

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    for db in databases:
        uploads_summary.append({
            "database": db,
            "date": _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "timeTaken": f"{duration}s"
        })

    return uploads_summary

def authenticate_token(func):
    def wrapper(*args, **kwargs):
        token = session.get('token')
        if not token:
            return jsonify({'message': 'Access token missing'}), 401
        try:
            decoded = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            request.user = decoded
            return func(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token expired'}), 403
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token'}), 403
    wrapper.__name__ = func.__name__
    return wrapper

# --------------------
# Upload Endpoint
# --------------------
@app.route('/api/upload', methods=['POST'])
def upload():
    file = request.files['file']
    databases = json.loads(request.form['databases'])
    database_name = request.form['databaseName']
    extension = request.form['fileExtension']
    session_id = session.get("user")  # or however you track sessions

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    task = process_upload_task.apply_async(args=[filepath, databases, database_name, extension, session_id])
    return jsonify({"task_id": task.id}), 202



# --------------------
# Preview Endpoint
# --------------------



# --------------------
# File List Endpoint
# --------------------
@app.route('/api/files', methods=['GET'])
def list_files():
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': f'Error reading files: {str(e)}'}), 500

# --------------------



@app.route('/file-preview/<filename>', methods=['GET'])
def file_preview(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.isfile(file_path):
            return jsonify({'error': 'File not found'}), 404

        preview = []
        # CSV → read first 5 rows via pandas
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(file_path, nrows=5)
            preview = df.to_dict(orient='records')

        # JSON → handle both array/object and newline-delimited JSON
        elif filename.lower().endswith('.json'):
            content = ""
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip() # Read content while file is open

            if content.startswith('[') or content.startswith('{'):
                try:
                    # Attempt to load as a single JSON array or object
                    full_data = json.loads(content)
                    if isinstance(full_data, list):
                        preview = full_data[:5]
                    else:
                        # If it's a single object, wrap it in a list for consistent handling
                        preview = [full_data]
                except json.JSONDecodeError:
                    # Fallback if it's not a valid single JSON structure
                    # This might happen if the file is malformed or truly NDJSON but with issues
                    pass # Will proceed to NDJSON attempt if preview is still empty
            
            # If preview is still empty (e.g., not a single JSON array/object, or initial parse failed),
            # try treating it as newline-delimited JSON.
            if not preview:
                # Reopen the file to read lines if the first attempt failed
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = list(itertools.islice(f, 5))
                    for line in lines:
                        try:
                            parsed_line = json.loads(line.strip())
                            preview.append(parsed_line)
                        except json.JSONDecodeError:
                            # If a line isn't valid JSON, treat it as a raw line
                            preview.append({'value': line.strip()})


        # anything else → just return first 5 raw lines
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = list(itertools.islice(f, 5))
                preview = [{'value': ln.rstrip('\n')} for ln in lines]

        # Ensure all data in the preview is JSON-serializable
        cleaned_preview = serialize_json_safe(preview)
        
        return jsonify({'preview': cleaned_preview})
    
    except Exception as e:
        # Log the error for debugging
        app.logger.error(f"Error in file-preview: {e}")
        return jsonify({'error': str(e)}), 500

# ... (rest of the code) ...


# --------------------
# Main Entry
# --------------------
MYSQL_HOST = '172.16.1.125'
MYSQL_USER = 'root'
MYSQL_PASSWORD = 'StrongPassword123!'

# This global variable will store the FULL sharded data in memory
global_sharded_data = {}

def infer_data_type(value):
    try:
        int_val = int(value)
        if abs(int_val) <= 2147483647:
            return 'INT'
        else:
            return 'BIGINT'
    except:
        try:
            float_val = float(value)
            return 'FLOAT'
        except:
            try:
                parsed_date = date_parser.parse(value)
                return 'DATETIME'
            except:
                if isinstance(value, str) and len(value) > 255:
                    return 'TEXT'
                return 'VARCHAR(255)'

def determine_column_types(rows):
    if not rows:
        return {}
    column_types = {}
    sample = rows[:5]
    for col in sample[0].keys():
        types = [infer_data_type(row[col]) for row in sample]
        if all(t == 'INT' for t in types):
            column_types[col] = 'INT'
        elif all(t in ['INT', 'FLOAT'] for t in types):
            column_types[col] = 'FLOAT'
        elif all(t == 'DATETIME' for t in types):
            column_types[col] = 'DATETIME'
        else:
            column_types[col] = 'VARCHAR(255)'
    return column_types

def get_connection(database=None):
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=database,
        cursorclass=pymysql.cursors.DictCursor
    )

def process_uploaded_file(filepath, databases, db_name, extension, session_id):
    results = []
    start_time = time.time()

    for db in databases:
        result = insert_data_into_database(filepath, db, db_name, extension)
        end_time = time.time()
        results.append({
            "database": db,
            "date": _dt.datetime.now().strftime("%Y-%m-%d"),
            "timeTaken": round(end_time - start_time, 2)
        })

    return results

@app.route('/api/mysql/databases', methods=['GET'])
def list_mysql_databases():
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SHOW DATABASES")
            result = [row['Database'] for row in cursor.fetchall()]
        return jsonify({'databases': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mysql/tables', methods=['GET'])
def list_mysql_tables():
    db = request.args.get('database')
    try:
        conn = get_connection(db)
        with conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            result = [list(row.values())[0] for row in cursor.fetchall()]
        return jsonify({'tables': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- REPLACE the current list_mysql_columns -------  (≈ L84-95)
@app.route('/api/mysql/columns', methods=['GET'])
def list_mysql_columns():
    db     = request.args.get('database')
    table  = request.args.get('table')
    sql = """
        SELECT COLUMN_NAME
        FROM   INFORMATION_SCHEMA.COLUMNS
        WHERE  TABLE_SCHEMA=%s AND TABLE_NAME=%s
        ORDER  BY ORDINAL_POSITION
    """
    try:
        conn = get_connection()                     # no need to USE <db>
        with conn.cursor() as cur:
            cur.execute(sql, (db, table))
            cols = [row['COLUMN_NAME'] for row in cur.fetchall()]
        return jsonify({'columns': cols})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/mysql/records', methods=['GET'])
def fetch_mysql_records():
    db = request.args.get('database')
    table = request.args.get('table')
    try:
        conn = get_connection(db)
        limit  = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))

        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM `{table}` LIMIT %s OFFSET %s", (limit, offset))
            rows = cur.fetchall()

        return jsonify({'records': rows})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mysql/update', methods=['PUT'])
def update_mysql_records():
    db = request.args.get('database')
    table = request.args.get('table')
    records = request.json.get('records', [])

    try:
        conn = get_connection(db)
        with conn.cursor() as cursor:
            try:
                cursor.execute(f"ALTER TABLE `{table}` ADD COLUMN `last_updated` DATETIME")
            except OperationalError as e:
                # MySQL error 1060 = "Duplicate column name"
                if e.args[0] != 1060:
                    raise
            cursor.execute(f"SHOW KEYS FROM `{table}` WHERE Key_name = 'PRIMARY'")
            pk = cursor.fetchone()
            pk_col = pk['Column_name'] if pk else list(records[0].keys())[0]

            for record in records:
                identifier = record[pk_col]
                # 1) fetch the current row
                cursor.execute(
                    f"SELECT * FROM `{table}` WHERE `{pk_col}`=%s",
                    (identifier,)
                )
                existing = cursor.fetchone() or {}

                # 2) build an updates dict only for fields that changed
                dirty = {}
                for col, new_val in record.items():
                    if col == pk_col:
                        continue
                    old_val = existing.get(col)
                    # DATETIME columns: parse and compare
                    if isinstance(old_val, _dt.datetime):
                        try:
                            parsed = date_parser.parse(str(new_val))
                            if parsed != old_val:
                                dirty[col] = parsed.strftime('%Y-%m-%d %H:%M:%S')
                        except Exception:
                            # skip invalid dates
                            continue
                    else:
                        if new_val != old_val:
                            dirty[col] = new_val

                if not dirty:
                    continue  # nothing changed for this row

                # 3) execute UPDATE with only changed fields + timestamp
                update_str = ', '.join([f"`{k}`=%s" for k in dirty])
                values     = list(dirty.values()) + [identifier]
                sql = (
                    f"UPDATE `{table}` "
                    f"SET {update_str}, `last_updated`=NOW() "
                    f"WHERE `{pk_col}`=%s"
                )
                cursor.execute(sql, values)
        conn.commit()
        return jsonify({'message': 'Records updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mysql/upload', methods=['POST'])
def upload_to_mysql():
    data = request.json
    database = data['database']
    filename = data['filename']
    ext = filename.split('.')[-1]
    table = os.path.splitext(filename)[0]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
        conn.commit()

        conn = get_connection(database)
        with conn.cursor() as cursor:
            with open(filepath, 'r', encoding='utf-8') as f:
                if ext == 'json':
                    rows = json.load(f)
                else:
                    rows = list(csv.DictReader(f))
            column_types = determine_column_types(rows)
            columns = ', '.join([f"`{k}` {v}" for k, v in column_types.items()])
            cursor.execute(f"CREATE TABLE IF NOT EXISTS `{table}` ({columns})")

            col_names = list(rows[0].keys())
            placeholders = ', '.join(['%s'] * len(col_names))
            insert_query = f"INSERT INTO `{table}` ({', '.join(col_names)}) VALUES ({placeholders})"
            values = [[row.get(col) for col in col_names] for row in rows]
            cursor.executemany(insert_query, values)
        conn.commit()
        return jsonify({'message': f'{len(rows)} records inserted into {table}.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

mongo_client = MongoClient('mongodb://172.16.1.126:27017')

@app.route('/api/mongodb/databases', methods=['GET'])
def list_mongo_databases():
    try:
        dbs = mongo_client.list_database_names()
        return jsonify({'databases': dbs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/tables', methods=['GET'])
def list_mongo_collections():
    dbname = request.args.get('database')
    try:
        collections = mongo_client[dbname].list_collection_names()
        return jsonify({'tables': collections})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/columns', methods=['GET'])
def list_mongo_fields():
    dbname = request.args.get('database')
    collection = request.args.get('collection')
    try:
        sample = mongo_client[dbname][collection].find_one()
        return jsonify({'columns': list(sample.keys()) if sample else []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/records', methods=['GET'])
def get_mongo_records():
    db = request.args.get('database')
    table = request.args.get('table')
    limit = request.args.get('limit', default=100, type=int)
    offset = request.args.get('offset', default=0, type=int)
    try:
        cursor = mongo_client[db][table].find().skip(offset).limit(limit)
        raw = list(cursor)
        cleaned = []
        for doc in raw:
            # stringify ObjectId
            doc['_id'] = str(doc['_id'])
            # replace NaNs, datetimes, Decimals, etc.
            cleaned.append(serialize_json_safe(doc))
        return jsonify({'records': cleaned})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/update', methods=['PUT'])
def update_mongo_records():
    db = request.args.get('database')
    table = request.args.get('table')
    records = request.json.get('records', [])

    try:
        collection = mongo_client[db][table]
        for record in records:
            _id = record.get('_id')
            if not _id:
                continue
            existing = collection.find_one({'_id': ObjectId(_id)})
            updates = {}
            for key, value in record.items():
                if key != '_id' and existing.get(key) != value:
                    updates[key] = value
            if updates:
                updates['last_updates'] = _dt.datetime.now()
                collection.update_one({'_id': ObjectId(_id)}, {'$set': updates})
        return jsonify({'message': 'Records updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mongodb/upload', methods=['POST'])
def upload_to_mongodb():
    data = request.json
    database = data['database']
    filename = data['filename']
    ext = filename.split('.')[-1]
    collection = os.path.splitext(filename)[0]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            rows = json.load(f) if ext == 'json' else list(csv.DictReader(f))
            if not isinstance(rows, list):
                rows = [rows]
        mongo_client[database][collection].insert_many(rows)
        return jsonify({'message': f'{len(rows)} records inserted into MongoDB collection {collection}.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mysql/shard-status', methods=['GET'])
def get_mysql_shard_status():
    database = request.args.get('database')
    table = request.args.get('table')
    
    if not all([database, table]):
        return jsonify({"error": "Database and table parameters are required"}), 400
    
    try:
        status = sharding_status_cache.get((database, table), {})
        partitions = global_sharded_data.get((database, table), {})
        
        return jsonify({
            "status": status.get('status', 'unknown'),
            "method": status.get('method', ''),
            "column": status.get('column', ''),
            "num_partitions": status.get('num_partitions', 0),
            "timestamp": status.get('timestamp', ''),
            "partitions": {k: len(v) for k, v in partitions.items()}
        })
    except Exception as e:
        logger.error(f"Error fetching MySQL shard status: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/mongodb/shard-status", methods=["GET"])
def get_mongodb_shard_status():
    from bson import MinKey, MaxKey

    database = request.args.get("database")
    collection = request.args.get("collection")

    if not database or not collection:
        return jsonify({"error": "Database and collection are required."}), 400

    try:
        admin_db = mongo_client["admin"]
        config_db = mongo_client["config"]
        db = mongo_client[database]
        coll = db[collection]

        full_name = f"{database}.{collection}"
        print(f"⏳ Waiting for config.chunks to reflect sharding of {full_name}")

        # Retry loop to wait for chunk propagation
        shard_chunks = []
        wait_intervals = [5] * 18  # 18 * 5s = 90s max wait
        for wait in wait_intervals:
            shard_chunks = list(config_db["chunks"].find({"ns": full_name}))
            if shard_chunks:
                break
            print(f"⚠️ No chunks yet for {full_name}... retrying in {wait}s")
            time.sleep(wait)

        if not shard_chunks:
            # Final fallback: try getting shard info via collStats
            print(f"⚠️ No chunks found in config.chunks, but checking distribution via coll.stats()")
            stats = coll.aggregate([{"$collStats": {"storageStats": {}}}])
            stats = list(stats)
            if stats and "sharded" in stats[0] and stats[0]["sharded"]:
                shard_chunks = [{"shard": k} for k in stats[0]["sharded"].keys()]
                print(f"✅ Sharding confirmed via collStats with {len(shard_chunks)} chunks.")
            else:
                return jsonify({
                    "message": f"Sharding applied but no chunks found in config.chunks for {full_name} after 90s",
                    "isSharded": False,
                    "shardChunks": [],
                    "shardData": {}
                })

        shard_status = admin_db.command({"listShards": 1})
        shard_data = {}

        for shard in shard_status["shards"]:
            shard_name = shard["_id"]
            chunks_for_shard = [chunk for chunk in shard_chunks if chunk["shard"] == shard_name]
            records = coll.find().limit(5)
            shard_data[shard_name] = {
                "chunkCount": len(chunks_for_shard),
                "records": list(records)
            }

        return jsonify({
            "message": f"Sharding status for collection '{collection}' retrieved successfully.",
            "totalChunks": len(shard_chunks),
            "shardChunks": shard_chunks,
            "shardData": shard_data,
            "isSharded": True
        })

    except Exception as e:
        print(f"❌ MongoDB Shard Status Error: {e}")
        return jsonify({"error": str(e), "isSharded": False}), 500

@app.route('/api/mongodb/query-performance', methods=['POST'])
def mongodb_query_performance():
    try:
        data = request.get_json()
        database = data['database']
        collection = data['collection']
        filter_str = data['filter']

        task = mongodb_query_performance_task.apply_async(args=[database, collection, filter_str])
        return jsonify({"task_id": task.id})
    except Exception as e:
        print("Error dispatching MongoDB query performance task:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/api/mongodb/query-performance', methods=['POST'])
def mongo_query_performance_sync(): # Renamed to avoid conflict with Celery task
    body = request.json
    database = body['database']
    collection = body['collection']
    filter_criteria = json.loads(body['filter'])

    try:
        coll = mongo_client[database][collection]
        unsharded = mongo_client[database][f"{collection}_unsharded"]

        t1 = _dt.datetime.now()
        results_sharded = list(coll.find(filter_criteria))
        t2 = _dt.datetime.now()

        t3 = _dt.datetime.now()
        results_unsharded = list(unsharded.find(filter_criteria))
        t4 = _dt.datetime.now()

        return jsonify({ # Return jsonify directly
            "message": "Query performance comparison executed successfully.",
            "sharded": {
                "timeTaken": (t2 - t1).total_seconds() * 1000,
                "results": serialize_mongo(results_sharded[:5]) # Limit and serialize
            },
            "unsharded": {
                "timeTaken": (t4 - t3).total_seconds() * 1000,
                "results": serialize_mongo(results_unsharded[:5]) # Limit and serialize
            }
        })


    except Exception as e:
        return jsonify({'error': str(e)}), 500

NEO4J_URI = 'neo4j://172.16.1.132:7687'
NEO4J_USER = 'neo4j'
NEO4J_PASSWORD = 'student1'

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

@app.route('/api/neo4j/upload', methods=['POST'])
def upload_to_neo4j():
    data = request.json
    filename = data['filename']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    label = os.path.splitext(filename)[0]
    ext = filename.split('.')[-1]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            rows = json.load(f) if ext == 'json' else list(csv.DictReader(f))

        if not isinstance(rows, list):
            rows = [rows]

        with driver.session() as session:
            for row in rows:
                props = {k: str(v).replace("'", "\\'") for k, v in row.items()}
                query = f"CREATE (n:{label} {{"
                query += ", ".join([f"{k}: '${k}'" for k in props])
                query += "})"
                session.run(query, **props)

        return jsonify({'message': f'{len(rows)} records inserted as Neo4j nodes with label {label}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


JWT_SECRET = 'secretKey'
JWT_EXPIRY_SECONDS = 1800

@app.route('/api/register', methods=['POST'])
def register_user():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    try:
        users_col = mongo_client['admin']['app_users']
        if users_col.find_one({'username': username}):
            return jsonify({'message': 'User already exists'}), 409

        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        users_col.insert_one({
            'username': username,
            'password': hashed_pw.decode('utf-8')
        })

        # Optional: also create a read-only MongoDB user
        mongo_client.admin.command("createUser", username,
            pwd=password,
            roles=[{"role": "read", "db": "yourDatabaseName"}]
        )

        return jsonify({'message': 'User registered successfully'}), 201
    except Exception as e:
        return jsonify({'message': f'Registration failed: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login_user():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    try:
        user = mongo_client['admin']['app_users'].find_one({'username': username})
        if not user:
            return jsonify({'message': 'Invalid username or password'}), 400

        if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return jsonify({'message': 'Invalid username or password'}), 400

        # include expiry in the JWT
        payload = {
            'userId': str(user['_id']),
            'username': username,
            'exp': _dt.datetime.utcnow() + _dt.timedelta(seconds=JWT_EXPIRY_SECONDS)
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')

        # make the session cookie expire according to PERMANENT_SESSION_LIFETIME
        session.permanent = True
        session['token'] = token

        return jsonify({'message': 'Login successful', 'token': token, 'expiresIn': JWT_EXPIRY_SECONDS})
    except Exception as e:
        return jsonify({'message': f'Login failed: {str(e)}'}), 500




@app.route('/api/home', methods=['GET'])
@authenticate_token
def home():
    return jsonify({'message': f"Welcome {request.user['username']}!"})

@app.route('/api/mongodb/query', methods=['POST'])
def mongo_query():
    try:
        data = request.json
        dbname = data['database']
        collection = data['collection']
        filter_data = data.get('filter', {})

        docs = list(mongo_client[dbname][collection].find(filter_data).limit(100))
        for doc in docs:
            doc['_id'] = str(doc['_id'])
        return jsonify({'results': docs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- MySQL Sharding Endpoint ---

#@app.route('/api/mysql/shard', methods=['POST'])
def mysql_shard_endpoint():
    data = request.get_json()
    required = ['database', 'table', 'column', 'method', 'numPartitions']
    if not all(k in data for k in required):
        return jsonify({"error": "Missing required parameters"}), 400

    if data['method'] == 'custom' and 'customRules' not in data:
        return jsonify({"error": "customRules required for custom sharding"}), 400

    try:
        # import task here to break circularity
        from celery_app import mysql_shard_task

        # enqueue on the worker's default queue
        async_result = mysql_shard_task.apply_async(args=[data])
        return jsonify({"task_id": async_result.id}), 202

    except Exception as e:
        app.logger.error(f"Failed to enqueue MySQL sharding task: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/folder-size', methods=['GET'])
def get_folder_size():
    folder_path = app.config['UPLOAD_FOLDER']
    total_size = 0

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)

    if total_size >= 1024 ** 3:
        size_str = f"{total_size / (1024 ** 3):.2f} GB"
    elif total_size >= 1024 ** 2:
        size_str = f"{total_size / (1024 ** 2):.2f} MB"
    elif total_size >= 1024:
        size_str = f"{total_size / 1024:.2f} KB"
    else:
        size_str = f"{total_size} Bytes"

    return jsonify(size=size_str)

@app.route('/api/user-count', methods=['GET'])
def get_user_count():
    try:
        user_count = mongo_client['admin']['system.users'].count_documents({})
        return jsonify({'count': user_count})
    except Exception as e:
        return jsonify({'message': 'Failed to fetch user count'}), 500

@app.route('/api/visualize/<filename>', methods=['POST'])
def generate_visualization(filename):
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    try:
        if filename.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif filename.endswith('.json'):
            data = pd.read_json(file_path)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400

        visualizations = {}
        numerical_cols = data.select_dtypes(include='number').columns.tolist()
        valid_numerical_cols = [col for col in numerical_cols if 'id' not in col.lower()]

        if len(numerical_cols) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(data[numerical_cols].corr(), annot=True, cmap='coolwarm')
            corr_path = os.path.join(STATIC_DIR, f'correlation_heatmap_{uuid.uuid4()}.png')
            plt.title('Correlation Heatmap')
            plt.savefig(corr_path)
            plt.close()
            visualizations['correlation_heatmap'] = f'static/correlation_heatmap.png'

        if 'target' not in data.columns:
            if valid_numerical_cols:
                data['target'] = data[valid_numerical_cols[0]] > data[valid_numerical_cols[0]].mean()
            else:
                return jsonify({'error': 'Cannot infer target column'}), 400

        X = data.drop(columns=['target'], errors='ignore')
        y = data['target']

        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        importance_path = os.path.join(STATIC_DIR, f'feature_importance_{uuid.uuid4()}.png')
        plt.title('Feature Importance')
        plt.savefig(importance_path)
        plt.close()
        visualizations['feature_importance'] = f'static/feature_importance.png'
        visualizations['importance_table'] = importance_df.to_dict(orient='records')

        column = request.json.get('column')
        if column and column in data.columns:
            plt.figure(figsize=(10, 6))
            data[column].hist(bins=30)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            hist_path = os.path.join(STATIC_DIR, f'histogram_{column}_{uuid.uuid4()}.png')
            plt.savefig(hist_path)
            plt.close()
            visualizations['histogram'] = f'static/histogram_{column}.png'

        return jsonify({'visualizations': visualizations})

    except Exception as e:
        return jsonify({'error': f'Visualization error: {str(e)}'}), 500

@app.route('/api/static/<filename>')
def serve_static_file(filename):
    return send_from_directory(STATIC_DIR, filename, mimetype='image/png')


@app.route('/columns/<filename>', methods=['GET'])
def get_columns(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    try:
        if filename.lower().endswith('.csv'):
            # Only read the header row
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, [])
            columns = header

        elif filename.lower().endswith('.json'):
            # Read the first meaningful line
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = next((l for l in f
                                   if l.strip() and l.strip() not in ('[', ']', ',')),
                                  '')
            first_line = first_line.rstrip().rstrip(',')
            # Try strict JSON parse first
            try:
                parsed = json.loads(first_line)
                # If it’s an array, grab the first element
                first_obj = parsed[0] if isinstance(parsed, list) and parsed else parsed
                columns = list(first_obj.keys() if isinstance(first_obj, dict) else {})
            except json.JSONDecodeError:
                # Fallback: regex‐extract keys like foo: or "foo":
                import re
                columns = re.findall(r'"?([A-Za-z0-9_]+)"?\s*:', first_line)

        else:
            return jsonify({'error': 'Unsupported file type'}), 400

        return jsonify({'columns': columns})

    except Exception as e:
        app.logger.error(f"get_columns error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/api/start-task", methods=["POST"])
def start_task():
    request_data = request.get_json()
    filepath = request_data.get("filepath")
    task_type = request_data.get("task_type")
    column = request_data.get("column", None)
    task = process_file_task.delay(filepath, task_type, column)


    if not filepath or not task_type:
        return jsonify({"error": "Missing filepath or task_type"}), 400

    column = request_data.get("column")
    task = process_file_task.apply_async(args=[filepath, task_type, column])

    return jsonify({"task_id": task.id}), 202

@app.route('/api/task-status/<task_id>')
def task_status(task_id):
    result = process_file_task.AsyncResult(task_id)
    if result.state == 'PENDING':
        return jsonify({'status': 'pending'})
    elif result.state == 'FAILURE':
        return jsonify({'status': 'failed', 'error': str(result.info)})
    elif result.state == 'SUCCESS':
        return jsonify({
            'status': 'done',
            'result': result.result  # ✅ this should contain {"visualizations": ...}
        })
    else:
        return jsonify({'status': result.state})

@app.route('/api/upload-status/<task_id>')
def upload_status(task_id):
    result = process_upload_task.AsyncResult(task_id)
    if result.state == 'PENDING':
        return jsonify({'status': 'pending'})
    elif result.state == 'FAILURE':
        return jsonify({'status': 'failed', 'error': str(result.info)})
    elif result.state == 'SUCCESS':
        return jsonify({
            'status': 'done',
            'result': result.result  # This is the list of summaries like [{'database': ..., 'timeTaken': ...}]
        })
    else:
        return jsonify({'status': result.state})


# Removed mysql_shard_task as its logic is now synchronous in mysql_shard endpoint

@celery.task(name="main.mongodb_shard_task")
def mongodb_shard_task(data):
    from pymongo import MongoClient
    from bson import ObjectId
    import time

    database = data['database']
    collection = data['collection']
    shardKey = data['shardKey']
    method = data['method']

    try:
        client = MongoClient("mongodb://172.16.1.126:27017",
        serverSelectionTimeoutMS=60000,  # give 60 s to find a mongos
        socketTimeoutMS=0              )  # MUST be mongos
        admin_db = client["admin"]
        db = client[database]
        coll = db[collection]

        clone_name = f"{collection}_unsharded"
        if clone_name not in db.list_collection_names():
            # run the aggregation as a direct command to avoid cursor timeouts
            db.command({
                "aggregate": collection,
                "pipeline": [
                    { "$match": {} },
                    { "$out": clone_name }
                ],
                "allowDiskUse": True,
                # no maxTimeMS limit
                "cursor": {}
            })
            print(f"✅ Server‐side clone created via db.command: {clone_name}")


        full_name = f"{database}.{collection}"
        config_db = client["config"]

        # Check via collStats whether it’s actually sharded
        stats = db.command({"collStats": collection})
        if not stats.get("sharded", False):
            # 1) allow sharding on this DB
            admin_db.command({"enableSharding": database})

            # 2) create the index on the shard key
            index_type = "hashed" if method == "hash" else 1
            coll.create_index([(shardKey, index_type)])

            # 3) shard the collection
            admin_db.command({
                "shardCollection": full_name,
                "key": {shardKey: index_type}
            })

            # 4) wait for at least one chunk to appear
            for _ in range(20):       # up to 20×2s = 40s
                if config_db["chunks"].find_one({"ns": full_name}):
                    break
                time.sleep(2)
            else:
                print(f"⚠️ No chunks found for {full_name} after waiting; check balancer/logs")

        # ✅ Fetch shards
        shards = admin_db.command("listShards")["shards"]
        sample_docs = list(coll.find().limit(5))
        for doc in sample_docs:
            doc["_id"] = str(doc["_id"])

        # ✅ Same records copied into each shard's result
        shard_data = {shard["_id"]: {"records": sample_docs} for shard in shards}

        return {
            "status": "done",
            "message": f"✅ Displaying 5 records from '{collection}' on all shards",
            "shardData": shard_data
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}

# --- NEW TASK: MySQL sharding in the background --------------------------
@celery.task(name="main.mysql_shard_task")
def mysql_shard_task(data):
    import re, math
    from decimal import Decimal

    database       = data.get("database")
    table          = data.get("table")
    column         = data.get("column")
    method         = data.get("method")
    num_partitions = int(data.get("numPartitions", 1))
    custom_rules   = data.get("customRules", {})

    # Validate
    if not all([database, table, column, method]):
        return {"error": "Missing required parameters", "isSharded": False}

    # Sanitize identifiers
    safe_table  = re.sub(r'[^0-9a-zA-Z_]', '_', table)
    safe_column = re.sub(r'[^0-9a-zA-Z_]', '_', column)

    # 1) Fetch every row (just like server.js)
    conn = get_connection(database)
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(f"SELECT * FROM `{safe_table}`")
        results = cur.fetchall()

    # 2) Initialize empty partitions
    parts = {f"p{i}": [] for i in range(num_partitions)} if method != "custom" else {p: [] for p in custom_rules}

    # 3) Custom‐rule buckets
    if method == "custom" and custom_rules:
        for row in results:
            val = row.get(column)
            for p, vals in custom_rules.items():
                if val in vals:
                    parts[p].append(row)
        # done

    else:
        # detect numeric
        is_numeric = all(
            isinstance(r.get(column), (int, float, Decimal)) or
            re.fullmatch(r'-?\d+(\.\d+)?', str(r.get(column)))
            for r in results if r.get(column) is not None
        )

        if is_numeric:
            # RANGE logic
            if method == "range":
                vals = [float(r[column]) for r in results]
                min_v, max_v = min(vals), max(vals)
                size = math.ceil((max_v - min_v) / num_partitions) or 1
                for row in results:
                    idx = int((float(row[column]) - min_v) / size)
                    idx = max(0, min(idx, num_partitions - 1))
                    parts[f"p{idx}"].append(row)

            # HASH logic
            else:  # method == "hash"
                for row in results:
                    h = row[column]
                    idx = abs(hash(h)) % num_partitions
                    parts[f"p{idx}"].append(row)

        # Categorical fallback
        else:
            cat_map = {}
            for row in results:
                val = row.get(column)
                if val not in cat_map:
                    cat_map[val] = len(cat_map) % num_partitions
                parts[f"p{cat_map[val]}"].append(row)

    # 4) Store & return preview
    global global_sharded_data
    global_sharded_data[(database, table)] = parts
    return {
        "message": f"✅ Sharding applied using {method} on `{table}`",
        "partitions": {k: v[:5] for k, v in parts.items()},
        "isSharded": True
    }

@celery.task(name="main.mysql_query_performance_task")
def mysql_query_performance_task(data):
    """
    Runs the user SQL against both the full table and the in‐memory partition,
    in batches of 50k rows, so we can time each without blocking Flask.
    """
    import time, re
    from decimal import Decimal

    database          = data.get("database")
    table             = data.get("table")
    user_query        = data.get("userQuery")
    selected_partition= data.get("selectedPartition")

    # 1) Execute on full (unsharded) table in batches
    conn = get_connection(database)
    with conn.cursor(pymysql.cursors.SSDictCursor) as cur:
        
        start_full = time.time()
        cur.execute(user_query)
        full_rows = []
        while chunk := cur.fetchmany(50_000):
            full_rows.extend(chunk)
        time_full = int((time.time() - start_full) * 1000)

    # 2) Execute on sharded data in batches
    key = (database, table)
    parts = global_sharded_data.get(key, {})
    shard_rows = parts.get(selected_partition, [])
    start_shard = time.time()
    # simple WHERE filtering if query had a WHERE clause
    m = re.search(r"WHERE\s+(.+)", user_query, re.IGNORECASE)
    conds = [c.strip() for c in m.group(1).split("AND")] if m else []
    def matches(row):
        for c in conds:
            fld, val = [p.strip().strip("';") for p in c.split("=",1)]
            if str(row.get(fld)) != val:
                return False
        return True

    sharded_rows = []
    for i in range(0, len(shard_rows), 50_000):
        batch = shard_rows[i:i+50_000]
        if conds:
            batch = [r for r in batch if matches(r)]
        sharded_rows.extend(batch)
    time_shard = int((time.time() - start_shard) * 1000)

    return {
        "message": "MySQL query performance measured.",
        "timeTakenUnsharded": f"{time_full} ",
        "timeTakenSharded": f"{time_shard} ",
        "queryResult": full_rows[:5]
    }

@celery.task(name='main.mongodb_query_performance_task')
def mongodb_query_performance_task(database, collection, filter_str):
    client = MongoClient("mongodb://172.16.1.126:27017")
    db = client[database]
    coll = db[collection]
    unsharded_coll = db[f"{collection}_unsharded"]

    try:
        parsed_filter = json.loads(filter_str)
        print(f"✅ Parsed MongoDB Filter: {parsed_filter}")

        start_sharded = time.time()
        sharded_results = list(coll.find(parsed_filter).limit(5))
        time_sharded = round((time.time() - start_sharded) * 1000, 2)

        start_unsharded = time.time()
        unsharded_results = list(unsharded_coll.find(parsed_filter).limit(5))
        time_unsharded = round((time.time() - start_unsharded) * 1000, 2)

        return {
            "message": "Query performance comparison executed successfully.",
            "sharded": {
                "timeTaken": time_sharded,
                "results": serialize_mongo(sharded_results) # Fixed: Apply serialize_mongo
            },
            "unsharded": {
                "timeTaken": time_unsharded,
                "results": serialize_mongo(unsharded_results) # Fixed: Apply serialize_mongo
            }
        }
    except Exception as e:
        print(f"❌ Error in query comparison: {str(e)}")
        return {"error": str(e)}

@celery.task(bind=True)
def custom_shard_task(self, data):
    # Assuming custom_shard_handler is defined elsewhere or logic is inline
    # from your_module import custom_shard_handler
    # return custom_shard_handler(data)
    return {"error": "Custom shard handler not implemented in this context"}

# Removed mysql_query_perf_task as its logic is now synchronous in mysql_query_performance endpoint

@celery.task(bind=True)
def mongodb_query_perf_task_celery(self, data): # Renamed to avoid conflict
    # Assuming mongodb_query_performance is defined elsewhere or logic is inline
    # from your_module import mongodb_query_performance
    # return mongodb_query_performance(data)
    return {"error": "MongoDB query performance task not implemented in this context"}

@app.route('/api/<db_type>/shard', methods=['POST'])
def trigger_shard(db_type):
    try:
        data = request.get_json()
        if db_type == 'mysql':
            task = mysql_shard_task.apply_async(args=[data])
            return jsonify({"task_id": task.id}), 202
        elif db_type == 'mongodb':
            task = mongodb_shard_task.apply_async(args=[data])
            return jsonify({"task_id": task.id}), 202
        else:
            return jsonify({"error": "Invalid database type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/custom/shard', methods=['POST'])
def trigger_custom_shard():
    data = request.get_json()
    task = custom_shard_task.apply_async(args=[data])
    return jsonify({"task_id": task.id})

@app.route('/api/<db_type>/query-performance', methods=['POST'])
def trigger_query_perf(db_type):
    data = request.get_json()
    if db_type == 'mysql':
        task = mysql_query_performance_task.apply_async(args=[data])
        return jsonify({"task_id": task.id}), 202
    elif db_type == 'mongodb':
        task = mongodb_query_performance_task.apply_async(args=[data]) # Use the correct Celery task
        return jsonify({"task_id": task.id}), 202
    else:
        return jsonify({"error": "Invalid database type"}), 400

# Removed mysql_query_perf_task as its logic is now synchronous in mysql_query_performance endpoint

# Removed mongodb_query_perf_task as its logic is now synchronous in mongo_query_performance_sync endpoint

@app.route('/api/mysql/shard/partitions', methods=['GET'])
def get_mysql_partitions():
    db    = request.args.get('database')
    table = request.args.get('table')
    if not db or not table:
        return jsonify({"error": "database & table required"}), 400

    parts = global_sharded_data.get((db, table), {})
    # send back only previews
    display = {p: rows[:5] for p, rows in parts.items()}
    return jsonify({'partitions': display})

@app.route('/api/task-result/<task_id>', methods=['GET'])
def task_result(task_id):
    # This endpoint is primarily for Celery tasks.
    # For synchronous MySQL sharding, the frontend won't poll this.
    # However, if you use Celery for other tasks, this remains relevant.
    task = celery.AsyncResult(task_id)
    if task.state == 'PENDING':
        return jsonify({'status': 'pending'})
    elif task.state == 'FAILURE':
        return jsonify({'status': 'failed', 'error': str(task.info)})
    elif task.state == 'SUCCESS':
        return jsonify({'status': 'done', 'result': serialize_json_safe(task.result)})
    return jsonify({'status': task.state})

# @app.route('/login')
# def login_page():
#     # allow unauthenticated users to hit your React login route
#     return send_from_directory(app.static_folder, 'index.html')

@app.route('/register')
def register_page():
    # allow unauthenticated users to hit your React register route
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    # if a static file exists, serve it (e.g. /static/js/main.js)
    file_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    # otherwise serve index.html and let React Router take over
    return send_from_directory(app.static_folder, 'index.html')

# @app.route('/<path:path>')
# @authenticate_token
# def serve_static(path):
#     # now any other React route (including /demopage) requires a valid token
#     full = os.path.join(app.static_folder, path)
#     if os.path.isfile(full):
#         return send_from_directory(app.static_folder, path)
#     return send_from_directory(app.static_folder, 'index.html')

# @app.route('/')
# @authenticate_token
# def serve_root():
#     return send_from_directory(app.static_folder, 'index.html')

@app.errorhandler(404)

def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

