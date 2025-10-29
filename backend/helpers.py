from pathlib import Path
import os, json, math, datetime as _dt
import pymysql
from dateutil import parser as date_parser
from bson.objectid import ObjectId
from decimal import Decimal
from pymongo import MongoClient
import pandas as pd
from config import settings

# ---------- Paths (same as main.py) ----------
STATIC_DIR = Path(settings.STATIC_DIR)
UPLOAD_FOLDER = Path(settings.UPLOAD_FOLDER)
DATA_DIR = UPLOAD_FOLDER  # you already set DATA_DIR to UPLOAD_FOLDER in main

# ---------- Globals used by sharding ----------
global_sharded_data = {}

# ---------- Mongo client (keep EXACT URI as in your current main.py) ----------
# Your main.py currently has:
#   mongo_client = MongoClient(settings.MONGO_URI)
# To keep behavior identical, use the same here:
mongo_client = MongoClient(settings.MONGO_URI)

# ---------- Serializers ----------
def serialize_mongo(obj):
    if isinstance(obj, list):
        return [serialize_mongo(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: serialize_mongo(v) for k, v in obj.items()}
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, _dt.datetime):
        return obj.isoformat()
    return obj

def serialize_json_safe(obj):
    if isinstance(obj, list):
        return [serialize_json_safe(i) for i in obj]
    if isinstance(obj, dict):
        return {k: serialize_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    return obj

# ---------- MySQL helpers ----------
def get_connection(database=None):
    return pymysql.connect(
        host=settings.MYSQL_HOST,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        database=database,
        cursorclass=pymysql.cursors.DictCursor
    )

def infer_data_type(value):
    try:
        int_val = int(value)
        return 'INT' if abs(int_val) <= 2147483647 else 'BIGINT'
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

# ---------- Upload helpers (unchanged logic) ----------
def mysql_insert_chunk(chunk_df, db_name, table_name):
    conn = pymysql.connect(
        host=settings.MYSQL_HOST,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        db=db_name
    )
    cursor = conn.cursor()

    cols_and_types = []
    for col in chunk_df.columns:
        sample = chunk_df[col].dropna()
        sample_value = str(sample.iloc[0]) if not sample.empty else ''
        dtype = infer_data_type(sample_value)
        cols_and_types.append(f"`{col}` {dtype}")
    create_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({', '.join(cols_and_types)})"
    cursor.execute(create_sql)

    cols = ', '.join([f"`{col}`" for col in chunk_df.columns])
    placeholders = ', '.join(['%s'] * len(chunk_df.columns))
    insert_sql = f"INSERT INTO `{table_name}` ({cols}) VALUES ({placeholders})"

    for _, row in chunk_df.iterrows():
        sanitized_row = []
        for val in row:
            if pd.isna(val):
                sanitized_row.append(None)
            elif isinstance(val, (dict, list)):
                sanitized_row.append(json.dumps(val))
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
    import time
    client = pymongo.MongoClient(
        settings.MONGO_URI,
        serverSelectionTimeoutMS=60000,
        connectTimeoutMS=60000,
        socketTimeoutMS=0
    )
    db = client[db_name]
    collection = db[collection_name]
    records = []
    for rec in chunk_df.to_dict(orient='records'):
        sanitized = {}
        for k, v in rec.items():
            sanitized[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
        records.append(sanitized)

    batch_size, max_retries = 100, 5
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        for attempt in range(1, max_retries + 1):
            try:
                collection.insert_many(batch, ordered=False)
                break
            except (ServerSelectionTimeoutError, AutoReconnect):
                time.sleep(5)
            except BulkWriteError:
                break
            except Exception:
                time.sleep(3)

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
