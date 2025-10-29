from flask import Blueprint, request, jsonify
from pymongo import UpdateOne
from bson.objectid import ObjectId
from datetime import datetime
from helpers import mongo_client, serialize_json_safe

mongo_bp = Blueprint("mongodb", __name__, url_prefix="/api/mongodb")

@mongo_bp.get("/databases")
def list_mongo_databases():
    try:
        dbs = mongo_client.list_database_names()
        return jsonify({'databases': dbs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mongo_bp.get("/tables")
def list_mongo_collections():
    dbname = request.args.get('database')
    try:
        collections = mongo_client[dbname].list_collection_names()
        return jsonify({'tables': collections})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mongo_bp.get("/columns")
def list_mongo_fields():
    dbname = request.args.get('database')
    collection = request.args.get('collection')
    try:
        sample = mongo_client[dbname][collection].find_one()
        return jsonify({'columns': list(sample.keys()) if sample else []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mongo_bp.get("/records")
def get_mongo_records():
    db = request.args.get('database')
    table = request.args.get('table')
    limit = request.args.get('limit', default=100, type=int)
    offset = request.args.get('offset', default=0, type=int)
    try:
        cursor = mongo_client[db][table].find().skip(offset).limit(limit)
        raw = list(cursor)
        for doc in raw:
            doc['_id'] = str(doc['_id'])
        return jsonify({'records': serialize_json_safe(raw)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mongo_bp.put("/update")
def update_mongo_records():
    db = request.args.get('database')
    table = request.args.get('table')
    payload = request.get_json(silent=True) or {}
    records = payload.get('records', [])

    if not db or not table:
        return jsonify({'error': 'database and table are required'}), 400
    if not isinstance(records, list):
        return jsonify({'error': 'records must be a list'}), 400
    if not records:
        return jsonify({'message': 'No records to update', 'modified': 0}), 200

    coll = mongo_client[db][table]

    # If any doc is missing last_update, enqueue a ONE-TIME async backfill.
    try:
        missing_exists = coll.find_one({'last_update': {'$exists': False}}, projection={'_id': 1})
        if missing_exists:
            from celery_app import celery
            # fire-and-forget (idempotent) — task defined in main.py below
            celery.send_task('main.mongodb_backfill_last_update_null', args=[db, table])
    except Exception as e:
        # Non-fatal; continue with the per-record updates
        pass

    # Small vs big payloads
    BIG_BATCH_THRESHOLD = 500
    if len(records) >= BIG_BATCH_THRESHOLD:
        from celery_app import celery
        task = celery.send_task('main.mongodb_bulk_update_task', args=[db, table, records])
        return jsonify({'task_id': task.id}), 202

    # ---------- synchronous minimal-diff path ----------
    try:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        ops, modified = [], 0
        BATCH_SIZE = 200

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
            if isinstance(raw_id, dict):  # supports {"$oid": "..."}
                raw_id = raw_id.get("$oid")
            query = {'_id': ObjectId(str(raw_id))} if (raw_id is not None and ObjectId.is_valid(str(raw_id))) else {'_id': raw_id}

            existing = coll.find_one(query) or {}
            updates = {}
            for k, v in rec.items():
                if k in ('_id', 'last_update'):
                    continue
                if existing.get(k) != v:
                    updates[k] = v
            if not updates:
                continue

            updates['last_update'] = ts
            ops.append(UpdateOne(query, {'$set': updates}))
            if len(ops) >= BATCH_SIZE:
                flush()

        flush()
        return jsonify({'message': 'Records updated successfully', 'modified': modified, 'timestamp': ts}), 200

    except Exception as e:
        # surface the error so we can see the true cause if anything remains
        return jsonify({'error': f'update_mongo_records failed: {str(e)}'}), 500

def _import_dt():
    import datetime as _dt
    return _dt
