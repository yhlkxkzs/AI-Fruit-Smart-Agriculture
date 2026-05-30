#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNet 分类模型 — 简单 HTTP API，供手机 App 调用。

- 上传图片后自动用 MobileNet 识别，结果直接返回给手机端。
- 会保存每张上传的图片，并写入上传记录（时间、保存路径、预测结果等）。

依赖：pip install flask pillow  （torch/torchvision 训练时已有）

用法（在服务器上）：
  python api_mobilenet.py --weights runs/mobilenet_v3/weights/best.pt --port 8000

接口：
  POST /predict  表单 file=图片 或 JSON {"image_base64": "..."}  -> 识别结果
  POST /annotation/upload  表单 image=图片, annotation=JSON字符串  -> 校准标注落盘到 datatemp/<水果英文名>/
  POST /recognition_signal  JSON {"received": true/false, ...}  -> 手机上报识别信号
  GET  /upload_log?date=...  -> 按日期查询记录
  GET  /upload_log/export?date=...&format=csv  -> 导出记录
  GET  /health  -> 健康检查
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import secrets
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_file
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0, ShuffleNet_V2_X0_5_Weights, ShuffleNet_V2_X1_0_Weights
from PIL import Image

# 项目根：train_detection 的上一级
SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR.parent / "runs"
PROJECT_ROOT = SCRIPT_DIR.parent.parent

app = Flask(__name__)

# =========================
# Auth: lightweight in-process token auth
# =========================
# 说明：
# - 为满足 APP 联调（/api/v1/auth/login + Bearer）提供最小可用实现
# - 当前为内存态 access token（重启服务会失效）
# - 生产建议切换为 JWT + 刷新令牌 + 持久化会话
ACCESS_TOKEN_EXPIRES_IN = int(os.getenv("AFSA_ACCESS_TOKEN_EXPIRES", "7200"))
REQUIRE_AUTH_FOR_API = os.getenv("AFSA_REQUIRE_AUTH", "1").strip().lower() in ("1", "true", "yes", "on")
ALLOW_SELF_REGISTER = os.getenv("AFSA_ALLOW_SELF_REGISTER", "0").strip().lower() in ("1", "true", "yes", "on")
ADMIN_SECRET = os.getenv("AFSA_ADMIN_SECRET", "").strip()
DEV_CLIENT_CHANNELS = {
    s.strip().lower()
    for s in os.getenv("AFSA_DEV_CLIENT_CHANNELS", "dev,development,internal").split(",")
    if s.strip()
}

# 可通过环境变量覆盖：AFSA_AUTH_USERS='[{"id":"usr_001","login":"zhangsan","password":"secret","display_name":"张三","email":"zhangsan@example.com"}]'
_DEFAULT_AUTH_USERS = [
    {
        "id": "usr_001",
        "login": "zhangsan",
        "password": "secret",
        "display_name": "张三",
        "email": "zhangsan@example.com",
        "username": "zhangsan",
    },
    {
        "id": "usr_002",
        "login": "admin",
        "password": "admin123",
        "display_name": "管理员",
        "email": "admin@example.com",
        "username": "admin",
    },
]

try:
    _env_users = json.loads(os.getenv("AFSA_AUTH_USERS", "[]"))
    AUTH_USERS = _env_users if isinstance(_env_users, list) and _env_users else _DEFAULT_AUTH_USERS
except Exception:
    AUTH_USERS = _DEFAULT_AUTH_USERS

# access_token -> {"exp": unix_ts, "user": {...}}
_ACCESS_TOKENS: dict[str, dict] = {}
_AUTH_LOCK = threading.Lock()


def _sanitize_user_profile(user: dict) -> dict:
    return {
        "id": user.get("id"),
        "username": user.get("username") or user.get("login"),
        "display_name": user.get("display_name") or user.get("name") or user.get("username") or user.get("login"),
        "email": user.get("email"),
    }


def _find_user(login: str) -> dict | None:
    key = (login or "").strip().lower()
    for u in AUTH_USERS:
        l = str(u.get("login", "")).strip().lower()
        n = str(u.get("username", "")).strip().lower()
        e = str(u.get("email", "")).strip().lower()
        if key and (key == l or key == n or key == e):
            return u
    return None


def _issue_access_token(user: dict) -> tuple[str, int]:
    now = int(time.time())
    exp = now + ACCESS_TOKEN_EXPIRES_IN
    token = secrets.token_urlsafe(32)
    with _AUTH_LOCK:
        _ACCESS_TOKENS[token] = {"exp": exp, "user": _sanitize_user_profile(user)}
    return token, ACCESS_TOKEN_EXPIRES_IN


def _build_auth_success_payload(token: str, expires_in: int, user_profile: dict) -> dict:
    """
    登录/注册成功响应。
    使用 data 信封为主，同时补充 token 兼容别名，满足前端解析优先级。
    """
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "access_token": token,
            "accessToken": token,
            "token": token,
            "id_token": token,
            "token_type": "Bearer",
            "expires_in": expires_in,
            "expiresIn": expires_in,
            "user": user_profile,
            "profile": user_profile,
        },
    }


def _extract_bearer_token() -> str | None:
    auth = request.headers.get("Authorization", "").strip()
    if not auth.lower().startswith("bearer "):
        return None
    token = auth[7:].strip()
    return token or None


def _verify_access_token(token: str) -> dict | None:
    now = int(time.time())
    with _AUTH_LOCK:
        item = _ACCESS_TOKENS.get(token)
        if not item:
            return None
        if int(item.get("exp", 0)) <= now:
            _ACCESS_TOKENS.pop(token, None)
            return None
        return item


def _is_admin_request() -> bool:
    if not ADMIN_SECRET:
        return False
    got = (request.headers.get("X-Admin-Secret", "") or "").strip()
    return got == ADMIN_SECRET


def _client_channel() -> str:
    """客户端通道：建议前端固定传 X-App-Channel=user/dev。"""
    return (request.headers.get("X-App-Channel", "") or "").strip().lower()


def _is_dev_client() -> bool:
    channel = _client_channel()
    return bool(channel and channel in DEV_CLIENT_CHANNELS)


@app.before_request
def _enforce_auth_for_dev():
    """
    开发版默认强制鉴权：
    - 放行根路径、健康检查、登录/注册/鉴权自检
    - 其余接口必须带有效 Bearer token
    """
    if request.method == "OPTIONS" or not REQUIRE_AUTH_FOR_API:
        return None

    open_paths = {
        "/",
        "/health",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/me",
        "/api/v1/auth/sessions",
    }
    if request.path in open_paths:
        if request.path == "/api/v1/auth/sessions":
            if _is_admin_request():
                return None
            return jsonify({"message": "forbidden", "error": "admin_secret_required"}), 403
        if request.path == "/api/v1/auth/register" and _is_dev_client() and not _is_admin_request():
            return jsonify({"message": "开发版注册需管理员授权", "error": "dev_register_forbidden"}), 403
        if request.path == "/api/v1/auth/login" and _is_dev_client() and not _is_admin_request():
            return jsonify({"message": "开发版登录需管理员授权", "error": "dev_login_forbidden"}), 403
        return None

    token = _extract_bearer_token()
    if not token:
        return jsonify({"message": "未授权，请先登录", "error": "unauthorized"}), 401
    if not _verify_access_token(token):
        return jsonify({"message": "token 无效或已过期", "error": "invalid_token"}), 401
    return None


@app.after_request
def _add_cors(resp):
    """允许跨域：浏览器或手机 App 从别的主机访问时，POST /predict 不会被浏览器拦截。"""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp


@app.route("/predict", methods=["OPTIONS"])
def predict_options():
    """CORS 预检：浏览器在发 POST 前会先发 OPTIONS，需返回 200。"""
    return "", 204


@app.route("/annotation/upload", methods=["OPTIONS"])
def annotation_upload_options():
    return "", 204


@app.route("/api/v1/auth/register", methods=["OPTIONS"])
def auth_register_options():
    return "", 204


@app.route("/api/v1/auth/login", methods=["POST", "OPTIONS"])
def auth_login():
    """
    登录接口（与前端约定）:
      POST /api/v1/auth/login
      body: {"login":"zhangsan","password":"secret","remember_me":true}
    """
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True) or {}
    login = str(payload.get("login", "")).strip()
    password = str(payload.get("password", ""))
    if not login or not password:
        return jsonify({"message": "缺少 login 或 password", "detail": "login/password required"}), 422

    user = _find_user(login)
    if not user:
        return jsonify({"message": "用户名或密码错误", "error": "invalid_credentials"}), 401

    # 轻量联调实现：明文比对；生产环境应改为哈希校验
    if str(user.get("password", "")) != password:
        return jsonify({"message": "用户名或密码错误", "error": "invalid_credentials"}), 401

    token, expires_in = _issue_access_token(user)
    profile = _sanitize_user_profile(user)
    return jsonify(_build_auth_success_payload(token, expires_in, profile)), 200


@app.route("/api/v1/auth/register", methods=["POST"])
def auth_register():
    """
    注册接口（与前端约定）:
      POST /api/v1/auth/register
      body: {"username":"zhangsan","email":"zhangsan@example.com","password":"secret"}
    """
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    email = str(payload.get("email", "")).strip()
    password = str(payload.get("password", ""))

    if not username or not email or not password:
        return jsonify({"message": "缺少 username/email/password", "detail": "username/email/password required"}), 422
    if len(username) < 3:
        return jsonify({"message": "用户名至少 3 位", "detail": "username too short"}), 422
    if "@" not in email or "." not in email:
        return jsonify({"message": "邮箱格式不正确", "detail": "invalid email"}), 422
    if len(password) < 6:
        return jsonify({"message": "密码至少 6 位", "detail": "password too short"}), 422
    if _is_dev_client() and not _is_admin_request():
        return jsonify({"message": "开发版注册需管理员授权", "error": "dev_register_forbidden"}), 403
    if not ALLOW_SELF_REGISTER and not _is_admin_request():
        return jsonify({"message": "当前环境已关闭自助注册，请联系管理员开通", "error": "registration_disabled"}), 403

    ul = username.lower()
    el = email.lower()
    with _AUTH_LOCK:
        for u in AUTH_USERS:
            if str(u.get("username", "")).strip().lower() == ul or str(u.get("login", "")).strip().lower() == ul:
                return jsonify({"message": "用户名已存在", "error": "username_exists"}), 409
            if str(u.get("email", "")).strip().lower() == el:
                return jsonify({"message": "邮箱已存在", "error": "email_exists"}), 409

        # 简易用户ID生成（联调用途）
        user_id = f"usr_{len(AUTH_USERS) + 1:03d}"
        new_user = {
            "id": user_id,
            "login": username,
            "username": username,
            "email": email,
            "password": password,  # 生产环境必须存哈希
            "display_name": username,
        }
        AUTH_USERS.append(new_user)

    token, expires_in = _issue_access_token(new_user)
    profile = _sanitize_user_profile(new_user)
    return jsonify(_build_auth_success_payload(token, expires_in, profile)), 201


@app.route("/api/v1/auth/me", methods=["GET"])
def auth_me():
    """鉴权联调接口：验证 Bearer token 是否有效。"""
    token = _extract_bearer_token()
    if not token:
        return jsonify({"message": "未登录或 token 缺失"}), 401
    item = _verify_access_token(token)
    if not item:
        return jsonify({"message": "token 无效或已过期"}), 401
    return jsonify({"code": 0, "message": "ok", "data": {"user": item["user"]}})


@app.route("/api/v1/auth/logout", methods=["POST"])
def auth_logout():
    token = _extract_bearer_token()
    if not token:
        return jsonify({"message": "未登录或 token 缺失"}), 401
    with _AUTH_LOCK:
        _ACCESS_TOKENS.pop(token, None)
    return jsonify({"code": 0, "message": "ok"})


@app.route("/api/v1/auth/sessions", methods=["GET"])
def auth_sessions():
    """
    管理接口：查看当前有效会话（谁已登录）。
    需在请求头带 X-Admin-Secret，且与 AFSA_ADMIN_SECRET 一致。
    """
    if not _is_admin_request():
        return jsonify({"message": "forbidden", "error": "admin_secret_required"}), 403
    now = int(time.time())
    sessions = []
    expired = []
    with _AUTH_LOCK:
        for token, item in _ACCESS_TOKENS.items():
            exp = int(item.get("exp", 0))
            if exp <= now:
                expired.append(token)
                continue
            user = dict(item.get("user") or {})
            sessions.append(
                {
                    "token_preview": f"{token[:8]}...",
                    "expires_at": exp,
                    "expires_in": max(0, exp - now),
                    "user": user,
                }
            )
        for token in expired:
            _ACCESS_TOKENS.pop(token, None)
    return jsonify({"code": 0, "message": "ok", "data": {"count": len(sessions), "sessions": sessions}})


def _save_annotation_upload(image: Image.Image, annotation: dict, unique_stem: str, ext: str) -> tuple[Path, Path, str]:
    """将图片与标注写入 datatemp/<水果英文名>/ 和 json/，返回 (图片路径, json路径, annotation_id)。"""
    global datatemp_dir, CATEGORY_IDS, _annotation_upload_lock
    fruit_type = (annotation.get("fruit_type") or "").strip() or "其他"
    en_name = _fruit_type_to_en(fruit_type)
    category_id = CATEGORY_IDS.get(en_name, CATEGORY_IDS["other"])
    category_name = en_name

    with _annotation_upload_lock:
        root = datatemp_dir / en_name
        root.mkdir(parents=True, exist_ok=True)
        json_dir = root / "json"
        json_dir.mkdir(exist_ok=True)

    img_path = root / f"{unique_stem}{ext}"
    image.save(img_path, "JPEG", quality=95) if ext.lower() in (".jpg", ".jpeg") else image.save(img_path)
    w, h = image.size
    image_id = abs(hash(unique_stem)) % (10 ** 9) + 10 ** 9
    ann_id = image_id + 10 ** 9

    # 与现阶段 dataset 的 json 格式一致（单图单标注）
    image_type = (annotation.get("image_type") or "真实照片").strip()
    multi = bool(annotation.get("multiple_fruits", False))
    payload = {
        "info": {
            "description": "Annotation upload (calibration)",
            "version": "1.0",
            "year": 2025,
            "contributor": "task1_fruit_classification",
            "source": "annotation_upload",
            "license": {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
        },
        "images": [
            {
                "id": image_id,
                "width": w,
                "height": h,
                "file_name": img_path.name,
                "size": img_path.stat().st_size,
                "format": "JPEG" if ext.lower() in (".jpg", ".jpeg") else "PNG",
                "url": "",
                "hash": "",
                "status": "success",
                "original_filename": img_path.name,
                "is_multi_fruit": multi,
                "image_type": image_type,
                "source_dataset": "annotation_upload",
                "fruit_type": annotation.get("fruit_type"),
                "leaf_condition": annotation.get("leaf_condition"),
                "maturity": annotation.get("maturity"),
                "has_disease": annotation.get("has_disease"),
                "disease_name": annotation.get("disease_name"),
                "disease_severity": annotation.get("disease_severity"),
                "notes": annotation.get("notes"),
                "client_time": annotation.get("client_time"),
                "detection_id": annotation.get("detection_id"),
            }
        ],
        "annotations": [
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [],
                "area": w * h,
                "bbox": [0, 0, w, h],
                "is_multi_fruit": multi,
            }
        ],
        "categories": [
            {
                "id": category_id,
                "name": category_name,
                "supercategory": "Fruit",
                "is_multi_fruit": multi,
            }
        ],
    }
    json_path = json_dir / f"{unique_stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    annotation_id = f"{en_name}_{unique_stem}"
    return img_path, json_path, annotation_id


@app.route("/annotation/upload", methods=["POST"])
def annotation_upload():
    """接收 App 校准上传：multipart/form-data 中 image=图片文件，annotation=JSON 字符串。落盘到 datatemp/<水果英文名>/。"""
    global datatemp_dir
    if not datatemp_dir:
        return jsonify({"success": False, "error": "datatemp dir not configured"}), 500
    if "image" not in request.files and "file" not in request.files:
        return jsonify({"success": False, "error": "missing image: use form field 'image' or 'file'"}), 400
    raw_annotation = request.form.get("annotation")
    if not raw_annotation:
        return jsonify({"success": False, "error": "missing annotation: form field 'annotation' (JSON string)"}), 400
    try:
        annotation = json.loads(raw_annotation)
    except json.JSONDecodeError as e:
        return jsonify({"success": False, "error": f"invalid annotation JSON: {e}"}), 400
    fruit_type = (annotation.get("fruit_type") or "").strip()
    if not fruit_type:
        return jsonify({"success": False, "error": "annotation.fruit_type is required"}), 400
    en_name = _fruit_type_to_en(fruit_type)

    f = request.files.get("image") or request.files.get("file")
    if not f or not f.filename:
        return jsonify({"success": False, "error": "no image file"}), 400
    try:
        image = Image.open(f.stream).convert("RGB")
    except Exception as e:
        return jsonify({"success": False, "error": f"invalid image: {e}"}), 400

    ext = Path(f.filename).suffix or ".jpg"
    if ext.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
        ext = ".jpg"
    unique_stem = f"annotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    try:
        img_path, json_path, annotation_id = _save_annotation_upload(image, annotation, unique_stem, ext)
    except Exception as e:
        print(f"[annotation/upload] save failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": True, "annotation_id": annotation_id})


# 全局：模型、类别列表、device
model = None
class_names = None
device = None
transform = None

# 多模型模式：三模型同时推理、结果对比；为空则单模型模式
models_list = []  # [{"name": str, "model": nn.Module, "class_names": list}, ...]

# 上传图片保存目录与记录文件（在 main 里初始化）
upload_dir = None
upload_log_path = None
_log_lock = threading.Lock()

# 手机识别信号记录目录（在 main 里初始化）
connectioninfo_dir = None
_recognition_signal_lock = threading.Lock()

# 标注上传落盘目录 datatemp（在 main 里初始化）
datatemp_dir = None
_annotation_upload_lock = threading.Lock()

# 果实种类（校准用）中文 -> 英文文件夹名；App 可能传中文或英文
FRUIT_TYPE_TO_EN = {
    "苹果": "apple", "梨": "pear", "桃": "peach", "葡萄": "grape",
    "柑橘": "citrus", "香蕉": "banana", "草莓": "strawberry", "西瓜": "watermelon",
    "其他": "other",
}
VALID_EN_NAMES = set(FRUIT_TYPE_TO_EN.values())  # 若 App 传英文（如 "apple"），直接用作文件夹名
CATEGORY_IDS = {en: 2000000100 + i for i, en in enumerate(sorted(FRUIT_TYPE_TO_EN.values()), 1)}


def _fruit_type_to_en(fruit_type: str) -> str:
    """将 fruit_type（中文或英文）转为英文文件夹名。"""
    s = (fruit_type or "").strip()
    if not s:
        return "other"
    if s in VALID_EN_NAMES:
        return s
    return FRUIT_TYPE_TO_EN.get(s, "other")


def _build_model_from_ckpt(ckpt: dict):
    """根据 ckpt 中的 model_type 构建对应模型并加载权重。支持 mobilenet_v3 / efficientnet_lite0 / shufflenet_v2。"""
    model_type = (ckpt.get("model_type") or "mobilenet_v3").lower()
    class_names = ckpt.get("classes")
    num_classes = ckpt.get("num_classes")
    if class_names is None:
        class_names = [str(i) for i in range(num_classes or 0)]
    if num_classes is None:
        num_classes = len(class_names)
    state = ckpt["model_state_dict"]

    if model_type == "efficientnet_lite0":
        try:
            import timm
        except ImportError:
            raise ImportError("EfficientNet-Lite0 权重需要 timm: pip install timm")
        model = timm.create_model("tf_efficientnet_lite0", pretrained=False, num_classes=num_classes)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    if model_type == "shufflenet_v2":
        variant = ckpt.get("variant", "x1_0")
        if variant == "x0_5":
            model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
        else:
            model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    # mobilenet_v3 或旧版无 model_type 的权重
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_model(weights_path: str):
    """加载 best.pt，根据 model_type 自动选择 MobileNetV3 / EfficientNet-Lite0 / ShuffleNet V2。"""
    global model, class_names, device, transform
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    class_names = ckpt.get("classes")
    num_classes = ckpt.get("num_classes")
    if class_names is None:
        class_names = [str(i) for i in range(num_classes or 0)]
    if num_classes is None:
        num_classes = len(class_names)

    model = _build_model_from_ckpt(ckpt)
    model_type = (ckpt.get("model_type") or "mobilenet_v3").lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print(f"Loaded model: {model_type}, {num_classes} classes, device={device}, classes={class_names}")


def load_models(weights_paths: list[str]):
    """加载多个权重，用于三模型同时推理、结果对比。与 load_model 二选一。"""
    global device, transform, model, class_names, models_list
    models_list = []
    for path in weights_paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"权重不存在: {path}")
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        model_type = (ckpt.get("model_type") or "mobilenet_v3").lower()
        # 从路径推断名称，如 runs/mobilenet_v3/weights/best.pt -> mobilenet_v3
        name = model_type
        for part in ("mobilenet_v3", "efficientnet_lite0", "shufflenet_v2"):
            if part in str(p):
                name = part
                break
        mdl = _build_model_from_ckpt(ckpt)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        mdl = mdl.to(device)
        cls_names = ckpt.get("classes")
        if cls_names is None:
            cls_names = [str(i) for i in range(ckpt.get("num_classes") or 0)]
        models_list.append({"name": name, "model": mdl, "class_names": cls_names})
        print(f"Loaded [{name}]: {len(cls_names)} classes")
    if models_list:
        model = models_list[0]["model"]
        class_names = models_list[0]["class_names"]


def _predict_image_single(image: Image.Image, mdl, cls_names: list, top_k: int = 3) -> dict:
    """单模型推理，返回与 predict_image 相同结构。"""
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = mdl(img_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    conf = float(probs[idx])
    low_confidence = conf < 0.6
    k = min(top_k, len(cls_names))
    order = probs.argsort()[::-1][:k]
    top_list = [{"class": cls_names[i], "confidence": float(probs[i])} for i in order]
    return {
        "class": cls_names[idx],
        "confidence": conf,
        "low_confidence": low_confidence,
        "top_k": top_list,
        "all_scores": {cls_names[i]: float(probs[i]) for i in range(len(cls_names))},
    }


def predict_image(image: Image.Image, top_k: int = 3) -> dict:
    """对 PIL Image 做分类，返回 class, confidence, all_scores, top_k, low_confidence。"""
    global model, class_names, device, transform
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    conf = float(probs[idx])
    # 置信度低于此认为「不确定」，前端可提示用户重拍或换图
    low_confidence = conf < 0.6
    # 返回概率最高的 top_k 个，便于前端展示「可能是 A / B / C」
    k = min(top_k, len(class_names))
    order = probs.argsort()[::-1][:k]
    top_list = [{"class": class_names[i], "confidence": float(probs[i])} for i in order]
    return {
        "class": class_names[idx],
        "confidence": conf,
        "low_confidence": low_confidence,
        "top_k": top_list,
        "all_scores": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


def save_upload_and_log(image: Image.Image, result: dict, remote_addr: str | None) -> str | None:
    """保存上传图片到 upload_dir，并追加一条记录到 upload_log；同时在图片同目录保存一份模型识别结果 JSON。"""
    global upload_dir, upload_log_path, _log_lock
    if not upload_dir or not upload_log_path:
        return None
    try:
        date_str = datetime.now().strftime("%Y-%m-%d")
        subdir = upload_dir / date_str
        subdir.mkdir(parents=True, exist_ok=True)
        ext = ".jpg"
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
        saved_path = subdir / name
        image.save(saved_path, "JPEG", quality=95)
        rel_path = f"{date_str}/{name}"

        # 记录里保留完整模型识别结果（单模型含 all_scores；多模型含 model_results、consistent）
        record = {
            "time": datetime.now().isoformat(),
            "saved_path": rel_path,
            "class": result.get("class"),
            "confidence": result.get("confidence"),
            "all_scores": result.get("all_scores"),
            "remote_addr": remote_addr,
        }
        if result.get("model_results") is not None:
            record["consistent"] = result.get("consistent")
            record["model_results"] = result.get("model_results")
        with _log_lock:
            with open(upload_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 在图片同目录保留一份识别结果 JSON（与图片同主文件名）
        result_path = saved_path.with_suffix(".json")
        json_content = {"time": record["time"], "class": result.get("class"), "confidence": result.get("confidence"), "all_scores": result.get("all_scores")}
        if result.get("model_results") is not None:
            json_content["consistent"] = result.get("consistent")
            json_content["model_results"] = result.get("model_results")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(json_content, f, ensure_ascii=False, indent=2)
        return rel_path
    except Exception as e:
        print(f"[upload log] save/log failed: {e}")
        return None


def _save_per_model_upload_and_history(
    image: Image.Image,
    rel_path: str,
    model_results: list[dict],
    remote_addr: str | None,
) -> None:
    """多模型时，将图片与本模型识别结果写入 efficientnet_lite0、shufflenet_v2 的 uploads/，并追加到各模型 uploads/history.jsonl。"""
    if not rel_path or not model_results:
        return
    parts = rel_path.split("/", 1)
    if len(parts) != 2:
        return
    date_str, filename = parts[0], parts[1]
    time_iso = datetime.now().isoformat()
    for one in model_results:
        model_name = one.get("model")
        if not model_name or model_name == "mobilenet_v3":
            continue
        model_upload_dir = RUNS_DIR / model_name / "uploads"
        if not model_upload_dir.is_dir():
            continue
        try:
            subdir = model_upload_dir / date_str
            subdir.mkdir(parents=True, exist_ok=True)
            saved_path = subdir / filename
            image.save(saved_path, "JPEG", quality=95)
            result_path = saved_path.with_suffix(".json")
            json_content = {
                "time": time_iso,
                "class": one.get("class"),
                "confidence": one.get("confidence"),
                "low_confidence": one.get("low_confidence"),
                "top_k": one.get("top_k"),
            }
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(json_content, f, ensure_ascii=False, indent=2)
            history_path = model_upload_dir / "history.jsonl"
            record = {
                "time": time_iso,
                "saved_path": rel_path,
                "class": one.get("class"),
                "confidence": one.get("confidence"),
                "top_k": one.get("top_k"),
                "remote_addr": remote_addr,
            }
            with _log_lock:
                with open(history_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[per-model save] {model_name}: {e}")


def _read_log_by_date(date_start: str | None, date_end: str | None) -> list[dict]:
    """读取 upload_log.jsonl，按日期过滤。date_start/date_end 为 YYYY-MM-DD，含边界。"""
    global upload_log_path, _log_lock
    if not upload_log_path or not upload_log_path.exists():
        return []
    records = []
    with _log_lock:
        with open(upload_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    t = r.get("time", "")[:10]  # 取 YYYY-MM-DD
                    if date_start and t < date_start:
                        continue
                    if date_end and t > date_end:
                        continue
                    records.append(r)
                except Exception:
                    pass
    return records


@app.route("/upload_log", methods=["GET"])
def upload_log_query():
    """按日期查询上传记录。参数：date=YYYY-MM-DD（单日）或 start=YYYY-MM-DD&end=YYYY-MM-DD（区间）。"""
    date = request.args.get("date")
    start = request.args.get("start")
    end = request.args.get("end")
    if date:
        date_start = date_end = date
    elif start and end:
        date_start, date_end = start, end
    else:
        date_start = date_end = None  # 不传则返回全部
    records = _read_log_by_date(date_start, date_end)
    return jsonify({"count": len(records), "records": records})


@app.route("/upload_log/export", methods=["GET"])
def upload_log_export():
    """导出上传记录。参数：date= 或 start=&end= 同 /upload_log；format=json（默认）或 csv。"""
    date = request.args.get("date")
    start = request.args.get("start")
    end = request.args.get("end")
    fmt = (request.args.get("format") or "json").lower()
    if date:
        date_start = date_end = date
    elif start and end:
        date_start, date_end = start, end
    else:
        date_start = date_end = None
    records = _read_log_by_date(date_start, date_end)
    if fmt == "csv":
        buf = io.StringIO()
        if records:
            fieldnames = ["time", "saved_path", "class", "confidence", "all_scores", "remote_addr"]
            w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in records:
                row = dict(r)
                if "all_scores" in row and isinstance(row["all_scores"], dict):
                    row["all_scores"] = json.dumps(row["all_scores"], ensure_ascii=False)
                w.writerow(row)
        buf.seek(0)
        return send_file(
            io.BytesIO(buf.getvalue().encode("utf-8-sig")),
            mimetype="text/csv",
            as_attachment=True,
            download_name="upload_log.csv",
        )
    return jsonify({"count": len(records), "records": records})


@app.route("/recognition_signal", methods=["POST"])
def recognition_signal():
    """手机端上报识别信号（不论是否接到结果都发）。记录写入 connectioninfo 目录。"""
    global connectioninfo_dir, _recognition_signal_lock
    if not connectioninfo_dir:
        return jsonify({"error": "connectioninfo dir not configured"}), 500
    try:
        data = request.get_json(silent=True) or {}
        received = data.get("received")  # True=接到结果, False=未接到
        record = {
            "server_time": datetime.now().isoformat(),
            "received": received,
            "remote_addr": request.remote_addr or (request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or None),
            "payload": data,
        }
        log_file = connectioninfo_dir / "recognition_signals.jsonl"
        with _recognition_signal_lock:
            connectioninfo_dir.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return jsonify({"status": "ok", "recorded": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    """根路径说明，避免浏览器访问 / 时出现 Not Found。"""
    info = {
        "message": "统一 API：知识库 + 水果识别",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "auth_login": "POST /api/v1/auth/login",
            "auth_me": "GET /api/v1/auth/me",
            "auth_logout": "POST /api/v1/auth/logout",
            "auth_sessions": "GET /api/v1/auth/sessions (admin)",
            "knowledge_filters": "GET /knowledge/filters",
            "knowledge_items": "GET /knowledge/items",
            "knowledge_item": "GET /knowledge/items/<id>",
        },
    }
    if models_list:
        info["multi_model"] = True
        info["models"] = [m["name"] for m in models_list]
    return jsonify(info)


@app.route("/health", methods=["GET"])
def health():
    """健康检查，便于确认服务是否起来。"""
    loaded = (model is not None) or len(models_list) > 0
    return jsonify({
        "status": "ok",
        "model_loaded": loaded,
        "multi_model": len(models_list) > 0,
        "models": [m["name"] for m in models_list] if models_list else None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """接收图片：表单 file= 或 JSON {"image_base64": "..."}。单模型返回原格式；多模型返回三模型结果及是否一致。"""
    if model is None and not models_list:
        return jsonify({"error": "model not loaded"}), 500

    image = None
    if request.files:
        f = request.files.get("file") or request.files.get("image")
        if f and f.filename:
            try:
                image = Image.open(f.stream).convert("RGB")
            except Exception as e:
                print(f"[predict] 解析上传文件失败: {e}")
                return jsonify({"error": f"invalid image file: {e}"}), 400
    elif request.is_json:
        data = request.get_json() or {}
        b64 = data.get("image_base64") or data.get("image")
        if b64:
            raw = base64.b64decode(b64)
            image = Image.open(io.BytesIO(raw)).convert("RGB")

    if image is None:
        return jsonify({"error": "no image: send form 'file' or JSON 'image_base64'"}), 400

    try:
        remote = request.remote_addr or (request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or None)
        sent_at = datetime.now().isoformat()

        if models_list:
            # 三模型同时推理，结果对比
            model_results = []
            for m in models_list:
                one = _predict_image_single(image, m["model"], m["class_names"])
                model_results.append({
                    "model": m["name"],
                    "class": one["class"],
                    "confidence": one["confidence"],
                    "low_confidence": one["low_confidence"],
                    "top_k": one["top_k"],
                })
            classes = [r["class"] for r in model_results]
            consistent = len(set(classes)) == 1
            if consistent:
                # 一致：取平均置信度作为整体置信度
                avg_conf = sum(r["confidence"] for r in model_results) / len(model_results)
                out = {
                    "consistent": True,
                    "class": classes[0],
                    "confidence": round(avg_conf, 4),
                    "low_confidence": avg_conf < 0.6,
                    "model_results": model_results,
                    "sent_to_client": True,
                    "sent_at": sent_at,
                }
            else:
                # 不一致：把所有结果给手机，并给一个综合 class（多数票或最高置信度）
                from collections import Counter
                votes = Counter(classes)
                majority_class = votes.most_common(1)[0][0]
                best = max(model_results, key=lambda r: r["confidence"])
                out = {
                    "consistent": False,
                    "class": majority_class,
                    "confidence": best["confidence"],
                    "low_confidence": True,
                    "model_results": model_results,
                    "sent_to_client": True,
                    "sent_at": sent_at,
                }
            rel_path = save_upload_and_log(
                image,
                {"class": out["class"], "confidence": out["confidence"], "top_k": model_results[0]["top_k"] if model_results else [], "consistent": out["consistent"], "model_results": model_results},
                remote,
            )
            if rel_path and model_results:
                _save_per_model_upload_and_history(image, rel_path, model_results, remote)
        else:
            out = predict_image(image)
            out["sent_to_client"] = True
            out["sent_at"] = sent_at
            save_upload_and_log(image, out, remote)

        resp = jsonify(out)
        resp.headers["X-Recognition-Sent"] = "true"
        resp.headers["X-Sent-At"] = sent_at
        return resp
    except Exception as e:
        print(f"[predict] 处理异常: {e}")
        return jsonify({"error": str(e)}), 500


def main():
    global upload_dir, upload_log_path, connectioninfo_dir, datatemp_dir
    parser = argparse.ArgumentParser(description="MobileNet 分类 API")
    parser.add_argument("--weights", type=str, default=str(RUNS_DIR / "mobilenet_v3" / "weights" / "best.pt"), help="best.pt 路径")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="0.0.0.0 允许外网访问")
    parser.add_argument("--upload-dir", type=str, default=None, help="上传图片保存目录，默认 runs/mobilenet_v3/uploads")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    if not weights_path.exists():
        raise FileNotFoundError(f"权重不存在: {weights_path}")

    upload_dir = Path(args.upload_dir) if args.upload_dir else (RUNS_DIR / "mobilenet_v3" / "uploads")
    upload_dir = upload_dir.resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_log_path = upload_dir / "upload_log.jsonl"

    connectioninfo_dir = upload_dir.parent / "connectioninfo"
    connectioninfo_dir.mkdir(parents=True, exist_ok=True)

    datatemp_dir = (PROJECT_ROOT / "datatemp").resolve()
    datatemp_dir.mkdir(parents=True, exist_ok=True)

    load_model(str(weights_path))
    print(f"API: http://{args.host}:{args.port}  (predict: POST /predict, health: GET /health)")
    print(f"上传图片保存: {upload_dir}  记录: {upload_log_path}")
    print(f"识别信号记录: {connectioninfo_dir}")
    print(f"标注上传落盘: {datatemp_dir}  (POST /annotation/upload)")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
