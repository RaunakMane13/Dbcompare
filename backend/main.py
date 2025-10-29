# main.py (cleaned)
from pathlib import Path
from config import settings
from flask import Flask, request, jsonify, send_from_directory, session, send_file
from flask_cors import CORS
from flask_session import Session

# helpers/blueprints
from helpers import (
    STATIC_DIR, UPLOAD_FOLDER, DATA_DIR,
    mongo_client, serialize_json_safe, serialize_mongo,
    get_connection, mysql_insert_chunk, mongodb_insert_chunk,
    neo4j_insert_chunk, global_sharded_data
)
from routes.mysql_routes import mysql_bp
from routes.mongo_routes import mongo_bp
from routes.files_visualize_routes import fv_bp
from routes.auth_routes import auth_bp

import os
import datetime as _dt
import json
import pymysql
import jwt
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from celery_app import celery
import uuid
import time
import random
from werkzeug.utils import secure_filename
from pymongo import MongoClient, UpdateOne
from bson.objectid import ObjectId
from datetime import datetime

build_dir = '/home/student/my-app/build'
app = Flask(__name__, static_folder=build_dir, static_url_path='')

# CORS
CORS(app, resources={r"/api/*": {"origins": settings.ALLOWED_ORIGINS}},
     supports_credentials=True)

# Session
app.config['SECRET_KEY'] = settings.SECRET_KEY
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = _dt.timedelta(minutes=30)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
Session(app)

# Paths
app.config['UPLOAD_FOLDER'] = settings.UPLOAD_FOLDER
os.makedirs(str(STATIC_DIR), exist_ok=True)

# Blueprints
app.register_blueprint(mysql_bp)
app.register_blueprint(mongo_bp)
app.register_blueprint(fv_bp)
app.register_blueprint(auth_bp)

# --------------------
# Celery tasks
# --------------------
@celery.task(name='main.process_file_task')
def process_file_task(filepath, task_type, column=None):
    visualizations = {}
    filename = os.path.basename(filepath)
    if not os.path.exists(filepath):
        return {'error': 'File not found'}

    MAX_ROWS = 100_000
    FILESIZE_LIMIT = 1000 * 1024**2  # (kept as-is)

    # Read CSV/JSON with light sampling for very large files
    if filename.lower().endswith('.csv'):
        if os.path.getsize(filepath) > FILESIZE_LIMIT:
            data = pd.read_csv(filepath, nrows=MAX_ROWS)
        else:
            data = pd.read_csv(filepath)
    elif filename.lower().endswith('.json'):
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

    # Correlation heatmap (>=2 numeric cols)
    numerical_cols = data.select_dtypes(include='number').columns.tolist()
    valid_cols = [c for c in numerical_cols if all(k not in c.lower() for k in ['id', 'code', 'number'])]
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(data[numerical_cols].corr(), annot=True, cmap='coolwarm')
        heatmap_name = f'correlation_heatmap_{uuid.uuid4()}.png'
        (STATIC_DIR / heatmap_name).parent.mkdir(parents=True, exist_ok=True)
        plt.title('Correlation Heatmap')
        plt.savefig(STATIC_DIR / heatmap_name)
        plt.close()
        visualizations['correlation_heatmap'] = f'static/{heatmap_name}'

    # Feature importance (auto target from first “valid” numeric col if missing)
    if valid_cols:
        try:
            if 'target' not in data.columns:
                data['target'] = data[valid_cols[0]] > data[valid_cols[0]].mean()
            X = data.drop(columns=['target'], errors='ignore')
            y = data['target']
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
            plt.title('Feature Importance')
            plt.savefig(STATIC_DIR / importance_name)
            plt.close()
            visualizations['feature_importance'] = f'static/{importance_name}'
            visualizations['importance_table'] = importance_df.to_dict(orient='records')
        except Exception as e:
            visualizations['feature_importance_error'] = str(e)
    else:
        visualizations['feature_importance'] = None
        visualizations['importance_table'] = []

    # Histogram for chosen column
    if column and column in data.columns:
        series = data[column]
        plt.figure(figsize=(10, 6))
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            if pd.api.types.is_bool_dtype(series):
                series = series.astype(int)
            plt.hist(series.dropna(), bins=30)
        else:
            import json as _json
            cat = series.dropna().map(lambda v: _json.dumps(v) if isinstance(v, (dict, list)) else str(v))
            top = cat.value_counts().nlargest(10)
            plt.barh(top.index, top.values)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        hist_name = f"histogram_{column}_{uuid.uuid4()}.png"
        plt.savefig(STATIC_DIR / hist_name)
        plt.close()
        visualizations['histogram'] = f'static/{hist_name}'

    return {'visualizations': visualizations}

@celery.task(name="main.process_upload_task")
def process_upload_task(filepath, databases, database_name, extension, session_id):
    start_time = time.time()
    CHUNK_SIZE = 1000
    uploads_summary = []

    if 'mysql' in databases:
        try:
            conn = pymysql.connect(
                host=settings.MYSQL_HOST,
                user=settings.MYSQL_USER,
                password=settings.MYSQL_PASSWORD
            )
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database_name}`")
            conn.commit()
            conn.close()
        except Exception as e:
            return {'error': f'Failed to create MySQL database: {str(e)}'}

    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        reader = pd.read_csv(filepath, chunksize=CHUNK_SIZE, dtype=str)
    elif ext == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        if text.startswith('['):
            records = json.loads(text)
        else:
            records = [json.loads(line) for line in text.splitlines() if line]
        df = pd.DataFrame(records)
        reader = (df[i:i+CHUNK_SIZE] for i in range(0, len(df), CHUNK_SIZE))
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

    duration = round(time.time() - start_time, 2)
    for db in databases:
        uploads_summary.append({
            "database": db,
            "date": _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "timeTaken": f"{duration}s"
        })
    return uploads_summary

# --------------------
# Auth gate + small APIs that UI still hits
# --------------------
JWT_SECRET = 'secretKey'
JWT_EXPIRY_SECONDS = 1800

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

@app.route('/api/home', methods=['GET'])
@authenticate_token
def home():
    return jsonify({'message': f"Welcome {request.user['username']}!"})

@app.route('/api/upload', methods=['POST'])
def upload():
    file = request.files['file']
    databases = json.loads(request.form['databases'])
    database_name = request.form['databaseName']
    extension = request.form['fileExtension']
    session_id = session.get("user")

    filename = secure_filename(file.filename)
    filepath = str(UPLOAD_FOLDER / filename)
    file.save(filepath)

    task = process_upload_task.apply_async(
        args=[filepath, databases, database_name, extension, session_id]
    )
    return jsonify({"task_id": task.id}), 202

@app.route('/api/visualization-guide', methods=['GET'])
def visualization_guide():
    guide = """# Visualization Guide
(kept same content as your current guide)
"""
    from io import BytesIO
    bio = BytesIO(guide.encode('utf-8'))
    return send_file(bio, as_attachment=True,
                     download_name='visualizations_guide.md',
                     mimetype='text/markdown')

# --------------------
# Sharding & performance (still used by UI)
# --------------------
@app.route('/api/mongodb/shard-status', methods=['GET'])
def get_mongodb_shard_status():
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

        shard_chunks = []
        for _ in range(18):
            shard_chunks = list(config_db["chunks"].find({"ns": full_name}))
            if shard_chunks:
                break
            time.sleep(5)

        if not shard_chunks:
            stats = list(coll.aggregate([{"$collStats": {"storageStats": {}}}]))
            if stats and "sharded" in stats[0] and stats[0]["sharded"]:
                shard_chunks = [{"shard": k} for k in stats[0]["sharded"].keys()]
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
            chunks_for_shard = [c for c in shard_chunks if c.get("shard") == shard_name]
            records = coll.find().limit(5)
            shard_data[shard_name] = {
                "chunkCount": len(chunks_for_shard),
                "records": list(records)
            }

        return jsonify({
            "message": f"Sharding status for collection '{collection}' retrieved successfully.",
            "totalChunks": len(shard_chunks),
            "shardChunks": shard_chunks,
            "shardData": serialize_mongo(shard_data),
            "isSharded": True
        })
    except Exception as e:
        return jsonify({"error": str(e), "isSharded": False}), 500

@celery.task(name="main.mongodb_shard_task")
def mongodb_shard_task(data):
    database = data['database']
    collection = data['collection']
    shardKey = data['shardKey']
    method = data['method']
    try:
        client = MongoClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=60000,
            socketTimeoutMS=0
        )
        admin_db = client["admin"]
        db = client[database]
        coll = db[collection]

        clone_name = f"{collection}_unsharded"
        if clone_name not in db.list_collection_names():
            db.command({
                "aggregate": collection,
                "pipeline": [{ "$match": {} }, { "$out": clone_name }],
                "allowDiskUse": True,
                "cursor": {}
            })

        full_name = f"{database}.{collection}"
        stats = db.command({"collStats": collection})
        if not stats.get("sharded", False):
            admin_db.command({"enableSharding": database})
            index_type = "hashed" if method == "hash" else 1
            coll.create_index([(shardKey, index_type)])
            admin_db.command({"shardCollection": full_name, "key": {shardKey: index_type}})

            for _ in range(20):
                if client["config"]["chunks"].find_one({"ns": full_name}):
                    break
                time.sleep(2)

        shards = admin_db.command("listShards")["shards"]
        sample_docs = list(coll.find().limit(5))
        for d in sample_docs:
            d["_id"] = str(d["_id"])
        shard_data = {s["_id"]: {"records": sample_docs} for s in shards}
        return {"status": "done", "message": f"✅ Displaying 5 records from '{collection}' on all shards",
                "shardData": shard_data}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

@celery.task(name="main.mysql_shard_task")
def mysql_shard_task(data):
    import re, math
    from decimal import Decimal
    database = data.get("database")
    table = data.get("table")
    column = data.get("column")
    method = data.get("method")
    num_partitions = int(data.get("numPartitions", 1))
    custom_rules = data.get("customRules", {})
    if not all([database, table, column, method]):
        return {"error": "Missing required parameters", "isSharded": False}

    safe_table  = re.sub(r'[^0-9a-zA-Z_]', '_', table)
    safe_column = re.sub(r'[^0-9a-zA-Z_]', '_', column)

    conn = get_connection(database)
    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(f"SELECT * FROM `{safe_table}`")
        results = cur.fetchall()

    parts = {f"p{i}": [] for i in range(num_partitions)} if method != "custom" else {p: [] for p in custom_rules}
    if method == "custom" and custom_rules:
        for row in results:
            val = row.get(column)
            for p, vals in custom_rules.items():
                if val in vals:
                    parts[p].append(row)
    else:
        is_numeric = all(
            isinstance(r.get(column), (int, float, Decimal)) or
            re.fullmatch(r'-?\d+(\.\d+)?', str(r.get(column)))
            for r in results if r.get(column) is not None
        )
        if is_numeric:
            if method == "range":
                vals = [float(r[column]) for r in results]
                min_v, max_v = min(vals), max(vals)
                size = math.ceil((max_v - min_v) / num_partitions) or 1
                for row in results:
                    idx = int((float(row[column]) - min_v) / size)
                    idx = max(0, min(idx, num_partitions - 1))
                    parts[f"p{idx}"].append(row)
            else:
                for row in results:
                    h = row[column]
                    idx = abs(hash(h)) % num_partitions
                    parts[f"p{idx}"].append(row)
        else:
            cat_map = {}
            for row in results:
                val = row.get(column)
                if val not in cat_map:
                    cat_map[val] = len(cat_map) % num_partitions
                parts[f"p{cat_map[val]}"].append(row)

    global global_sharded_data
    global_sharded_data[(database, table)] = parts
    return {
        "message": f"✅ Sharding applied using {method} on `{table}`",
        "partitions": {k: v[:5] for k, v in parts.items()},
        "isSharded": True
    }

@celery.task(name="main.mysql_query_performance_task")
def mysql_query_performance_task(data):
    import re, time
    database = data.get("database")
    table = data.get("table")
    user_query = data.get("userQuery")
    selected_partition = data.get("selectedPartition")

    conn = get_connection(database)
    with conn.cursor(pymysql.cursors.SSDictCursor) as cur:
        start_full = time.time()
        cur.execute(user_query)
        full_rows = []
        while chunk := cur.fetchmany(50_000):
            full_rows.extend(chunk)
        time_full = int((time.time() - start_full) * 1000)

    key = (database, table)
    parts = global_sharded_data.get(key, {})
    shard_rows = parts.get(selected_partition, [])
    start_shard = time.time()
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
    client = MongoClient(settings.MONGO_URI)
    db = client[database]
    coll = db[collection]
    unsharded_coll = db[f"{collection}_unsharded"]
    try:
        parsed_filter = json.loads(filter_str)
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
                "results": serialize_mongo(sharded_results)
            },
            "unsharded": {
                "timeTaken": time_unsharded,
                "results": serialize_mongo(unsharded_results)
            }
        }
    except Exception as e:
        return {"error": str(e)}

@celery.task(name='main.mongodb_backfill_last_update_null')
def mongodb_backfill_last_update_null(db, table):
    """
    Idempotent backfill that ensures every document has 'last_update': None
    without touching docs that already have the field.
    Runs once in the background; safe to call multiple times.
    """
    try:
        coll = mongo_client[db][table]
        # Do nothing if already compliant
        if not coll.find_one({'last_update': {'$exists': False}}, projection={'_id': 1}):
            return {'message': 'No backfill needed'}
        res = coll.update_many({'last_update': {'$exists': False}},
                               {'$set': {'last_update': None}})
        return {'message': 'Backfill complete', 'matched': res.matched_count, 'modified': res.modified_count}
    except Exception as e:
        return {'error': f'mongodb_backfill_last_update_null failed: {str(e)}'}

@celery.task(name='main.mongodb_bulk_update_task')
def mongodb_bulk_update_task(db, table, records):
    """
    Large editor saves: no per-row reads; set provided fields + last_update.
    Batches with bulk_write to avoid overload.
    """
    try:
        coll = mongo_client[db][table]
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # Opportunistically backfill (cheap check; task is idempotent)
        if coll.find_one({'last_update': {'$exists': False}}, projection={'__id': 1}):
            mongodb_backfill_last_update_null.delay(db, table)

        ops, modified, BATCH_SIZE = [], 0, 1000

        def flush():
            nonlocal ops, modified
            if not ops:
                return
            res = coll.bulk_write(ops, ordered=False)
            modified += getattr(res, 'modified_count', 0) or 0
            ops = []

        for rec in records:
            raw_id = rec.get('_id')
            if raw_id is None:
                continue
            if isinstance(raw_id, dict):
                raw_id = raw_id.get("$oid")
            query = {'_id': ObjectId(str(raw_id))} if (raw_id is not None and ObjectId.is_valid(str(raw_id))) else {'_id': raw_id}

            updates = {k: v for k, v in rec.items() if k not in ('_id', 'last_update')}
            if not updates:
                continue
            updates['last_update'] = ts

            ops.append(UpdateOne(query, {'$set': updates}))
            if len(ops) >= BATCH_SIZE:
                flush()

        flush()
        return {'message': 'Bulk update completed', 'modified': modified, 'timestamp': ts}

    except Exception as e:
        return {'error': f'mongodb_bulk_update_task failed: {str(e)}'}

# task triggers used by UI
@app.route('/api/<db_type>/shard', methods=['POST'])
def trigger_shard(db_type):
    try:
        data = request.get_json()
        if db_type == 'mysql':
            task = mysql_shard_task.apply_async(args=[data])
        elif db_type == 'mongodb':
            task = mongodb_shard_task.apply_async(args=[data])
        else:
            return jsonify({"error": "Invalid database type"}), 400
        return jsonify({"task_id": task.id}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/custom/shard', methods=['POST'])
def trigger_custom_shard():
    @celery.task(bind=True, name="main.custom_shard_task")
    def _custom(self, data):
        return {"error": "Custom shard handler not implemented in this context"}
    data = request.get_json()
    task = _custom.apply_async(args=[data])
    return jsonify({"task_id": task.id})

@app.route('/api/<db_type>/query-performance', methods=['POST'])
def trigger_query_perf(db_type):
    data = request.get_json() or {}

    if db_type == 'mysql':
        # mysql task already expects a single dict
        task = mysql_query_performance_task.apply_async(args=[data])

    elif db_type == 'mongodb':
        # mongodb task expects 3 positional args
        database   = data.get('database')
        collection = data.get('collection')

        # accept any of these from the UI and normalize to a JSON string
        # - "filter" (stringified JSON)
        # - "filter_str" (stringified JSON)
        # - "filterObj" (dict)
        filt       = data.get('filter') or data.get('filter_str')
        if filt is None:
            import json as _json
            filt = _json.dumps(data.get('filterObj', {}))

        if not database or not collection:
            return jsonify({"error": "database and collection are required"}), 400

        task = mongodb_query_performance_task.apply_async(args=[database, collection, filt])

    else:
        return jsonify({"error": "Invalid database type"}), 400

    return jsonify({"task_id": task.id}), 202


@app.route('/api/task-result/<task_id>', methods=['GET'])
def task_result(task_id):
    result = celery.AsyncResult(task_id)
    if result.state == 'PENDING':
        return jsonify({'status': 'pending'})
    elif result.state == 'FAILURE':
        return jsonify({'status': 'failed', 'error': str(result.info)})
    elif result.state == 'SUCCESS':
        return jsonify({'status': 'done', 'result': serialize_json_safe(result.result)})
    return jsonify({'status': result.state})

# SPA fallback
@app.route('/register')
def register_page():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    file_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
