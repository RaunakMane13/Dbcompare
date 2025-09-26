import os, csv, json, itertools, uuid
import pandas as pd
from flask import Blueprint, request, jsonify, send_from_directory
from helpers import UPLOAD_FOLDER, STATIC_DIR, DATA_DIR, serialize_json_safe
from celery_app import celery
from celery.result import AsyncResult

fv_bp = Blueprint("files_visualize", __name__, url_prefix="/api")

@fv_bp.get("/files")
def list_files():
    try:
        files = os.listdir(str(UPLOAD_FOLDER))
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': f'Error reading files: {str(e)}'}), 500

@fv_bp.get("/file-preview/<filename>")
def file_preview(filename):
    try:
        file_path = UPLOAD_FOLDER / filename
        if not file_path.is_file():
            return jsonify({'error': 'File not found'}), 404

        preview = []
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(file_path, nrows=5)
            preview = df.to_dict(orient='records')
        elif filename.lower().endswith('.json'):
            content = ""
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if content.startswith('[') or content.startswith('{'):
                try:
                    full_data = json.loads(content)
                    preview = full_data[:5] if isinstance(full_data, list) else [full_data]
                except json.JSONDecodeError:
                    pass
            if not preview:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = list(itertools.islice(f, 5))
                    for line in lines:
                        try:
                            parsed_line = json.loads(line.strip())
                            preview.append(parsed_line)
                        except json.JSONDecodeError:
                            preview.append({'value': line.strip()})
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = list(itertools.islice(f, 5))
                preview = [{'value': ln.rstrip('\n')} for ln in lines]

        cleaned_preview = serialize_json_safe(preview)
        resp = jsonify({'preview': cleaned_preview})
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@fv_bp.get("/columns/<filename>")
def get_columns(filename):
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404

    try:
        if filename.lower().endswith('.csv'):
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, [])
            columns = header
        elif filename.lower().endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = next((l for l in f if l.strip() and l.strip() not in ('[', ']', ',')), '')
            first_line = first_line.rstrip().rstrip(',')
            try:
                parsed = json.loads(first_line)
                first_obj = parsed[0] if isinstance(parsed, list) and parsed else parsed
                columns = list(first_obj.keys() if isinstance(first_obj, dict) else {})
            except json.JSONDecodeError:
                import re
                columns = re.findall(r'"?([A-Za-z0-9_]+)"?\s*:', first_line)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400
        return jsonify({'columns': columns})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@fv_bp.get("/folder-size")
def get_folder_size():
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(str(UPLOAD_FOLDER)):
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

@fv_bp.get("/user-count")
def get_user_count():
    from helpers import mongo_client
    try:
        user_count = mongo_client['admin']['system.users'].count_documents({})
        return jsonify({'count': user_count})
    except Exception:
        return jsonify({'message': 'Failed to fetch user count'}), 500

@fv_bp.get("/static/<filename>")
def serve_static_file(filename):
    return send_from_directory(str(STATIC_DIR), filename, mimetype='image/png')

# visualize + celery status
@fv_bp.post("/start-task")
def start_task():
    data = request.get_json()
    filepath = data.get("filepath")
    task_type = data.get("task_type")
    column = data.get("column")
    if not filepath or not task_type:
        return jsonify({"error": "Missing filepath or task_type"}), 400
    task = celery.send_task('main.process_file_task', args=[filepath, task_type, column])
    return jsonify({"task_id": task.id}), 202

@fv_bp.get("/task-status/<task_id>")
def task_status(task_id):
    result = AsyncResult(task_id, app=celery)
    if result.state == 'PENDING':
        return jsonify({'status': 'pending'})
    elif result.state == 'FAILURE':
        return jsonify({'status': 'failed', 'error': str(result.info)})
    elif result.state == 'SUCCESS':
        return jsonify({'status': 'done', 'result': result.result})
    else:
        return jsonify({'status': result.state})

@fv_bp.get("/upload-status/<task_id>")
def upload_status(task_id):
    result = AsyncResult(task_id, app=celery)
    if result.state == 'PENDING':
        return jsonify({'status': 'pending'})
    elif result.state == 'FAILURE':
        return jsonify({'status': 'failed', 'error': str(result.info)})
    elif result.state == 'SUCCESS':
        return jsonify({'status': 'done', 'result': result.result})
    else:
        return jsonify({'status': result.state})
