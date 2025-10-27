from flask import Blueprint, request, jsonify
from helpers import get_connection, determine_column_types, serialize_json_safe, global_sharded_data

mysql_bp = Blueprint("mysql", __name__, url_prefix="/api/mysql")

@mysql_bp.get("/databases")
def list_mysql_databases():
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SHOW DATABASES")
            result = [row['Database'] for row in cursor.fetchall()]
        return jsonify({'databases': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mysql_bp.get("/tables")
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

@mysql_bp.get("/columns")
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
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(sql, (db, table))
            cols = [row['COLUMN_NAME'] for row in cur.fetchall()]
        return jsonify({'columns': cols})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mysql_bp.get("/records")
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

@mysql_bp.put("/update")
def update_mysql_records():
    import datetime as _dt
    from dateutil import parser as date_parser
    from pymysql.err import OperationalError

    db = request.args.get('database')
    table = request.args.get('table')
    records = request.json.get('records', [])

    try:
        conn = get_connection(db)
        with conn.cursor() as cursor:
            try:
                cursor.execute(f"ALTER TABLE `{table}` ADD COLUMN `last_updated` DATETIME")
            except OperationalError as e:
                if e.args[0] != 1060:
                    raise
            cursor.execute(f"SHOW KEYS FROM `{table}` WHERE Key_name = 'PRIMARY'")
            pk = cursor.fetchone()
            pk_col = pk['Column_name'] if pk else list(records[0].keys())[0]

            for record in records:
                identifier = record[pk_col]
                cursor.execute(f"SELECT * FROM `{table}` WHERE `{pk_col}`=%s", (identifier,))
                existing = cursor.fetchone() or {}

                dirty = {}
                for col, new_val in record.items():
                    if col == pk_col:
                        continue
                    old_val = existing.get(col)
                    if isinstance(old_val, _dt.datetime):
                        try:
                            parsed = date_parser.parse(str(new_val))
                            if parsed != old_val:
                                dirty[col] = parsed.strftime('%Y-%m-%d %H:%M:%S')
                        except Exception:
                            continue
                    else:
                        if new_val != old_val:
                            dirty[col] = new_val

                if not dirty:
                    continue
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

@mysql_bp.post("/upload")
def upload_to_mysql():
    import os, csv, json
    data = request.json
    database = data['database']
    filename = data['filename']
    ext = filename.split('.')[-1]
    table = os.path.splitext(filename)[0]
    filepath = os.path.join(request.app.config['UPLOAD_FOLDER'] if hasattr(request, "app") else "", filename)

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

@mysql_bp.get("/shard-status")
def get_mysql_shard_status():
    from flask import current_app as app
    database = request.args.get('database')
    table = request.args.get('table')

    if not all([database, table]):
        return jsonify({"error": "Database and table parameters are required"}), 400

    try:
        status = {}  # original code references a cache; keeping empty preserves behavior
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
        app.logger.error(f"Error fetching MySQL shard status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@mysql_bp.get("/shard/partitions")
def get_mysql_partitions():
    db    = request.args.get('database')
    table = request.args.get('table')
    if not db or not table:
        return jsonify({"error": "database & table required"}), 400

    parts = global_sharded_data.get((db, table), {})
    display = {p: rows[:5] for p, rows in parts.items()}
    return jsonify({'partitions': display})
