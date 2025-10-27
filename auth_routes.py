import bcrypt, jwt, datetime as _dt
from flask import Blueprint, request, jsonify, session
from helpers import mongo_client

auth_bp = Blueprint("auth", __name__, url_prefix="/api")

JWT_SECRET = 'secretKey'
JWT_EXPIRY_SECONDS = 1800

@auth_bp.post("/register")
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

        mongo_client.admin.command("createUser", username,
            pwd=password,
            roles=[{"role": "read", "db": "yourDatabaseName"}]
        )

        return jsonify({'message': 'User registered successfully'}), 201
    except Exception as e:
        return jsonify({'message': f'Registration failed: {str(e)}'}), 500

@auth_bp.post("/login")
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

        payload = {
            'userId': str(user['_id']),
            'username': username,
            'exp': _dt.datetime.utcnow() + _dt.timedelta(seconds=JWT_EXPIRY_SECONDS)
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
        session.permanent = True
        session['token'] = token

        return jsonify({'message': 'Login successful', 'token': token, 'expiresIn': JWT_EXPIRY_SECONDS})
    except Exception as e:
        return jsonify({'message': f'Login failed: {str(e)}'}), 500
