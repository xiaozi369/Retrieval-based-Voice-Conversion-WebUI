"""
RVC-xiao 实时变声控制服务 — FastAPI + WebSocket。

职责:作为"总机",让网页控制面板 / Tauri 壳通过 HTTP/WS 指挥 RealtimeEngine。
不处理音频,只转发命令、查询信息、推送状态。
"""

import os
import re
import sys
import socket
import json
import base64
import asyncio
import threading
from collections import deque

# ---- 启动顺序关键:环境变量必须在 import configs.config / realtime_engine 之前设置 ----
# 本文件位于 xiaozi/ 子目录,项目根目录需加入 sys.path(configs 包在项目根)
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("weight_root", "assets/weights")
os.environ.setdefault("index_root", "logs")
os.environ.setdefault("outside_index_root", "assets/indices")
os.environ.setdefault("rmvpe_root", "assets/rmvpe")
# 限制 OpenMP 线程数,避免 librosa/numpy 操作抢占所有 CPU 核心导致音频卡顿
# (与 BAT GUI realtime_gui.py 行为一致)
os.environ.setdefault("OMP_NUM_THREADS", "4")

# Windows 控制台 UTF-8 输出(避免中文乱码 / Tauri 读取编码问题)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from configs.config import Config
from realtime_engine import RealtimeEngine

# ---------- 工具 ----------

def find_available_port(start_port, host="127.0.0.1"):
    """返回第一个可绑定的 TCP 端口(从 start_port 起)。"""
    if not 1 <= start_port <= 65535:
        raise ValueError(f"Port must be between 1 and 65535, got {start_port}.")
    for port in range(start_port, 65536):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, port))
            return port
        except OSError:
            continue
    raise OSError(f"No available TCP port from {start_port} through 65535.")


def get_index_path_from_model(sid):
    """按模型名自动配对 .index(逻辑同 infer/vc/utils.py,纯 os/re,避免重 import)。"""
    model_stem = os.path.splitext(os.path.basename(str(sid or "")))[0]
    experiment_name = re.sub(r"_e\d+_s\d+$", "", model_stem, flags=re.IGNORECASE)
    if not experiment_name:
        return ""

    candidates = []
    roots = [os.getenv("outside_index_root"), os.getenv("index_root")]
    for index_root in roots:
        if not index_root or not os.path.isdir(index_root):
            continue
        for root, _, files in os.walk(index_root, topdown=False):
            for name in files:
                if not name.lower().endswith(".index") or "trained" in name.lower():
                    continue
                index_stem = os.path.splitext(name)[0]
                lower_index = index_stem.lower()
                lower_experiment = experiment_name.lower()
                standard_match = (
                    lower_index.startswith(lower_experiment + "_added_")
                    or ("_" + lower_experiment + "_v1") in lower_index
                    or ("_" + lower_experiment + "_v2") in lower_index
                )
                exact_model_match = model_stem.lower() in lower_index
                if standard_match or exact_model_match:
                    path = os.path.abspath(os.path.join(root, name))
                    score = (
                        0 if standard_match else 1,
                        0
                        if os.path.abspath(index_root) == os.path.abspath(roots[0])
                        else 1,
                        -os.path.getmtime(path),
                        path.lower(),
                    )
                    candidates.append((score, path))
    return min(candidates, default=(None, ""), key=lambda item: item[0])[1]


def list_models():
    """扫描 weight_root 下的 .pth 模型,自动配对 .index。"""
    weight_root = os.getenv("weight_root", "assets/weights")
    models = []
    if not os.path.isdir(weight_root):
        return models
    for name in sorted(os.listdir(weight_root)):
        if not name.lower().endswith(".pth"):
            continue
        pth_path = os.path.abspath(os.path.join(weight_root, name))
        models.append(
            {
                "name": name,
                "pth_path": pth_path,
                "index_path": get_index_path_from_model(pth_path),
            }
        )
    return models


def _save_config(cfg):
    """将引擎配置持久化到 configs/config.json(与 BAT GUI set_values 保存逻辑一致)。"""
    import json as _json

    realtime_config_path = os.path.join("configs", "config.json")
    data = {
        "pth_path": cfg.pth_path,
        "index_path": cfg.index_path,
        "sg_hostapi": cfg.sg_hostapi,
        "sg_wasapi_exclusive": cfg.sg_wasapi_exclusive,
        "sg_input_device": cfg.sg_input_device,
        "sg_output_device": cfg.sg_output_device,
        "sr_type": cfg.sr_type,
        "threhold": cfg.threhold,
        "pitch": cfg.pitch,
        "rms_mix_rate": cfg.rms_mix_rate,
        "index_rate": cfg.index_rate,
        "block_time": cfg.block_time,
        "crossfade_length": cfg.crossfade_length,
        "extra_time": cfg.extra_time,
        "f0method": cfg.f0method,
    }
    try:
        with open(realtime_config_path, "w", encoding="utf-8") as f:
            _json.dump(data, f)
    except Exception:
        pass  # 写入失败不影响启动流程


def validate_start(params):
    """校验启动参数(与老 GUI set_values 规则一致)。"""
    if len((params.pth_path or "").strip()) == 0:
        raise ValueError("请选择pth文件")
    if len((params.index_path or "").strip()) == 0:
        raise ValueError("请选择index文件")
    pattern = re.compile(r"[^\x00-\x7F]+")
    if pattern.findall(params.pth_path):
        raise ValueError("pth文件路径不可包含中文")
    if pattern.findall(params.index_path):
        raise ValueError("index文件路径不可包含中文")


# ---------- 日志捕获(/ws/logs 广播) ----------

# 日志环形缓冲:最近 500 条,供新客户端连上时补发
LOG_BUFFER: deque = deque(maxlen=500)
_LOG_LOCK = threading.Lock()


def _colorize(line: str) -> str:
    """按级别上 ANSI 色;已有 ANSI 的(如引擎自带彩色输出)原样透传。"""
    if "\x1b[" in line:
        return line
    if any(k in line for k in ("ERROR", "CRITICAL", "Traceback")):
        return f"\x1b[31m{line}\x1b[0m"
    if "WARN" in line:
        return f"\x1b[33m{line}\x1b[0m"
    return line


class _Tee:
    """把 stdout/stderr 同时写入原始流并送入日志环形缓冲(捕获 print 与 uvicorn 日志)。"""

    def __init__(self, real):
        self._real = real

    def write(self, s: str):
        self._real.write(s)
        if s and s.strip():
            for line in s.splitlines():
                with _LOG_LOCK:
                    LOG_BUFFER.append(_colorize(line))
        return len(s)

    def flush(self):
        self._real.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self._real.fileno()


# 必须在引擎初始化之前安装,否则启动期日志会漏
sys.stdout = _Tee(sys.stdout)
sys.stderr = _Tee(sys.stderr)


# ---------- 引擎初始化 ----------

config = Config()
engine = RealtimeEngine(config)
# 从 configs/config.json 加载用户保存的参数(与 BAT GUI 行为一致),
# 否则 engine_config 只有硬编码默认值,可能导致音频参数不匹配产生电音。
engine.apply_saved_config()

# ---------- FastAPI 应用 ----------

app = FastAPI(title="RVC-xiao Realtime Server")

# 开发期跨域放开(前端可能从 WebView / 其它端口访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 请求模型 ----------

class StartParams(BaseModel):
    pth_path: str = ""
    index_path: str = ""
    sg_hostapi: str = ""
    sg_wasapi_exclusive: bool = False
    sg_input_device: str = ""
    sg_output_device: str = ""
    sr_type: str = "sr_model"
    threhold: float = -60
    pitch: int = 0
    formant: float = 0.0
    block_time: float = 0.25
    crossfade_length: float = 0.05
    extra_time: float = 2.5
    I_noise_reduce: bool = False
    O_noise_reduce: bool = False
    rms_mix_rate: float = 0.0
    index_rate: float = 0.0
    f0method: str = "rmvpe"


class ParamsUpdate(BaseModel):
    threhold: float | None = None
    pitch: int | None = None
    formant: float | None = None
    block_time: float | None = None
    crossfade_length: float | None = None
    extra_time: float | None = None
    index_rate: float | None = None
    rms_mix_rate: float | None = None
    f0method: str | None = None
    function: str | None = None
    I_noise_reduce: bool | None = None
    O_noise_reduce: bool | None = None


class CoverUpload(BaseModel):
    image: str = ""  # data:image/<mime>;base64,....(前端 FileReader 读出的 data URL)


class ModelMetaUpdate(BaseModel):
    display_name: str | None = None  # 中文名,None 表示不修改;空串表示清除
    image: str | None = None  # base64 data URL,None 表示不修改;空串表示清除封面


# ---------- REST 接口 ----------

@app.get("/api/health")
async def api_health():
    return {"ok": True, "running": engine.is_running()}


@app.get("/api/devices")
async def api_devices(hostapi: str = "", reload: bool = False):
    """获取音频设备列表。
    - hostapi: 指定设备类型(如 MME / Windows WASAPI),空则用当前配置
    - reload: 是否重新枚举 PortAudio 设备(引擎运行时拒绝,需先停止)
    与 BAT GUI 的 reload_devices / sg_hostapi 事件行为一致。"""
    if reload or hostapi:
        target = hostapi or engine.engine_config.sg_hostapi
        if target and target in engine.hostapis:
            try:
                engine.update_devices(hostapi_name=target)
                engine.engine_config.sg_hostapi = target
            except RuntimeError:
                # 引擎运行时拒绝重载设备
                pass
        elif target:
            return {"ok": False, "error": f"未知的设备类型: {target}"}
    return {
        "hostapis": engine.hostapis or [],
        "input_devices": engine.input_devices or [],
        "output_devices": engine.output_devices or [],
        "current": {
            "sg_hostapi": engine.engine_config.sg_hostapi,
            "sg_input_device": engine.engine_config.sg_input_device,
            "sg_output_device": engine.engine_config.sg_output_device,
        },
    }


@app.get("/api/models")
async def api_models():
    return {"models": list_models()}


# ---------- 封面图(存模型同目录,如 assets/weights/xxx.png) ----------

# 支持的图片扩展名(与前端 accept="image/*" 一致)
_COVER_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def _find_model(name: str):
    for m in list_models():
        if m["name"] == name:
            return m
    return None


def _cover_candidates(model) -> list[str]:
    """按模型名生成候选封面路径(stem + 各扩展名),只落在模型同目录,杜绝路径穿越。"""
    stem, _ = os.path.splitext(model["pth_path"])
    return [stem + ext for ext in _COVER_EXTS]


@app.get("/api/models/{name}/cover")
async def api_get_cover(name: str):
    # 1. 已安装模型:查模型同目录(stem + 各扩展名)
    model = _find_model(name)
    if model:
        for path in _cover_candidates(model):
            if os.path.isfile(path):
                return FileResponse(path)
    # 2. 未安装模型(或同目录无封面)回退:查统一封面目录 assets/covers/{stem}.{ext}
    stem = os.path.basename(os.path.splitext(name)[0])
    for ext in _COVER_EXTS:
        path = os.path.join("assets", "covers", stem + ext)
        if os.path.isfile(path):
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="未设置封面图")


# ---------- 本地模型元数据(用户编辑的中文名/图片,存 assets/covers/model_meta.json,与封面同目录) ----------

_MODEL_META_PATH = os.path.join("assets", "covers", "model_meta.json")


def _load_model_meta() -> dict:
    """读取本地模型元数据 JSON(不存在/损坏时返回空)。"""
    try:
        with open(_MODEL_META_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_model_meta(meta: dict):
    os.makedirs(os.path.dirname(_MODEL_META_PATH), exist_ok=True)
    with open(_MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


@app.get("/api/models/meta")
async def api_models_meta():
    """全量返回本地模型元数据:{模型名: {displayName, cover}}"""
    return {"meta": _load_model_meta()}


@app.post("/api/models/{name}/meta")
async def api_save_model_meta(name: str, body: ModelMetaUpdate):
    """保存/清除单个模型的本地元数据(中文名 + 封面图)。"""
    meta = _load_model_meta()
    entry = meta.get(name, {})

    if body.display_name is not None:
        entry["displayName"] = body.display_name

    if body.image is not None:
        if body.image == "":
            # 清除封面:删统一目录里的封面文件
            stem = os.path.basename(os.path.splitext(name)[0])
            for ext in _COVER_EXTS:
                p = os.path.join("assets", "covers", stem + ext)
                if os.path.isfile(p):
                    os.remove(p)
            entry.pop("cover", None)
        else:
            # 保存新封面到统一目录 assets/covers/{stem}.{ext}
            if not body.image.startswith("data:image/"):
                raise HTTPException(status_code=400, detail="无效的图片数据")
            _, _, b64 = body.image.partition(",")
            mime = body.image[5:].partition(";")[0]
            ext_map = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}
            ext = ext_map.get(mime, ".png")
            stem = os.path.basename(os.path.splitext(name)[0])
            cover_dir = os.path.join("assets", "covers")
            os.makedirs(cover_dir, exist_ok=True)
            cover_path = os.path.join(cover_dir, stem + ext)
            try:
                raw = base64.b64decode(b64)
            except Exception:
                raise HTTPException(status_code=400, detail="base64 解码失败")
            with open(cover_path, "wb") as f:
                f.write(raw)
            entry["cover"] = stem + ext

    # 条目有内容则保存,全空则删除该模型条目
    if entry:
        meta[name] = entry
    else:
        meta.pop(name, None)
    _save_model_meta(meta)
    return {"ok": True, "meta": meta.get(name, {})}


@app.post("/api/models/{name}/use")
async def api_use_model(name: str):
    """将模型设为当前使用:pth/index 写入引擎配置并持久化 configs/config.json(重启保留)。"""
    model = _find_model(name)
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")
    cfg = engine.engine_config
    cfg.pth_path = model["pth_path"]
    cfg.index_path = model["index_path"]
    _save_config(cfg)
    return {"ok": True, "pth_path": cfg.pth_path, "index_path": cfg.index_path}


@app.post("/api/models/{name}/cover")
async def api_upload_cover(name: str, body: CoverUpload):
    model = _find_model(name)
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")
    image = body.image or ""
    if not image.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="无效的图片数据")
    _, _, b64 = image.partition(",")
    mime = image[5:].partition(";")[0]
    ext_map = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}
    ext = ext_map.get(mime, ".png")
    stem, _ = os.path.splitext(model["pth_path"])
    cover_path = stem + ext
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="base64 解码失败")
    with open(cover_path, "wb") as f:
        f.write(raw)
    return {"ok": True, "cover": os.path.basename(cover_path)}


@app.delete("/api/models/{name}/cover")
async def api_delete_cover(name: str):
    model = _find_model(name)
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")
    for path in _cover_candidates(model):
        if os.path.isfile(path):
            os.remove(path)
            return {"ok": True}
    return {"ok": True}  # 原本就没有封面


@app.delete("/api/models/{name}")
async def api_delete_model(name: str):
    """删除模型: .pth + 配对 index + 封面(同目录/统一目录) + meta;若为当前使用模型则清空引擎配置。"""
    model = _find_model(name)
    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")

    removed = []

    # 1. 删 .pth 文件
    if os.path.isfile(model["pth_path"]):
        os.remove(model["pth_path"])
        removed.append("pth")

    # 2. 删配对 index 文件
    if model.get("index_path") and os.path.isfile(model["index_path"]):
        os.remove(model["index_path"])
        removed.append("index")

    # 3. 删封面(模型同目录 + 统一目录 assets/covers)
    stem = os.path.basename(os.path.splitext(name)[0])
    for ext in _COVER_EXTS:
        for base in (os.path.dirname(model["pth_path"]), os.path.join("assets", "covers")):
            p = os.path.join(base, stem + ext)
            if os.path.isfile(p):
                os.remove(p)
                removed.append(f"cover:{stem}{ext}")

    # 4. 删 meta 条目(中文名/封面记录)
    meta = _load_model_meta()
    if name in meta:
        meta.pop(name, None)
        _save_model_meta(meta)
        removed.append("meta")

    # 5. 若为当前使用模型,清空引擎配置并持久化
    cfg = engine.engine_config
    if cfg.pth_path and os.path.normpath(cfg.pth_path) == os.path.normpath(model["pth_path"]):
        cfg.pth_path = ""
        cfg.index_path = ""
        _save_config(cfg)
        removed.append("use")

    return {"ok": True, "removed": removed}


@app.get("/api/config")
async def api_config():
    cfg = engine.engine_config
    status = engine.get_status()
    return {
        "config": {
            "pth_path": cfg.pth_path,
            "index_path": cfg.index_path,
            "sg_hostapi": cfg.sg_hostapi,
            "sg_wasapi_exclusive": cfg.sg_wasapi_exclusive,
            "sg_input_device": cfg.sg_input_device,
            "sg_output_device": cfg.sg_output_device,
            "sr_type": cfg.sr_type,
            "threhold": cfg.threhold,
            "pitch": cfg.pitch,
            "formant": cfg.formant,
            "block_time": cfg.block_time,
            "crossfade_length": cfg.crossfade_length,
            "extra_time": cfg.extra_time,
            "I_noise_reduce": cfg.I_noise_reduce,
            "O_noise_reduce": cfg.O_noise_reduce,
            "rms_mix_rate": cfg.rms_mix_rate,
            "index_rate": cfg.index_rate,
            "f0method": cfg.f0method,
        },
        "status": status,
    }


@app.post("/api/start")
async def api_start(params: StartParams):
    if engine.is_running():
        return {"ok": False, "error": "已在运行中，请先停止"}
    try:
        validate_start(params)
        # 设备选择
        engine.set_devices(params.sg_input_device, params.sg_output_device)
        # 写入引擎配置
        cfg = engine.engine_config
        cfg.pth_path = params.pth_path
        cfg.index_path = params.index_path
        cfg.sg_hostapi = params.sg_hostapi
        cfg.sg_wasapi_exclusive = params.sg_wasapi_exclusive
        cfg.sg_input_device = params.sg_input_device
        cfg.sg_output_device = params.sg_output_device
        cfg.sr_type = "sr_model" if params.sr_type == "sr_model" else "sr_device"
        cfg.threhold = params.threhold
        cfg.pitch = params.pitch
        cfg.formant = params.formant
        cfg.block_time = params.block_time
        cfg.crossfade_length = params.crossfade_length
        cfg.extra_time = params.extra_time
        cfg.I_noise_reduce = params.I_noise_reduce
        cfg.O_noise_reduce = params.O_noise_reduce
        cfg.rms_mix_rate = params.rms_mix_rate
        cfg.index_rate = params.index_rate
        if params.f0method in ("pm", "rmvpe", "fcpe"):
            cfg.f0method = params.f0method
        # 持久化到 configs/config.json(与 BAT GUI set_values 保存逻辑一致)
        _save_config(cfg)
        # 启动(模型加载耗时,放线程池避免阻塞事件循环)
        await __start_in_thread()
        return {"ok": True}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"启动失败: {e}"}


async def __start_in_thread():
    import asyncio

    await asyncio.to_thread(engine.start_vc)


@app.post("/api/stop")
async def api_stop():
    await __stop_in_thread()
    return {"ok": True}


async def __stop_in_thread():
    import asyncio

    await asyncio.to_thread(engine.stop_stream)


@app.post("/api/params")
async def api_params(params: ParamsUpdate):
    try:
        cfg = engine.engine_config
        if params.threhold is not None:
            cfg.threhold = params.threhold
        if params.pitch is not None:
            engine.set_pitch(params.pitch)
        if params.formant is not None:
            engine.set_formant(params.formant)
        delay_changed = False
        if params.block_time is not None:
            cfg.block_time = params.block_time
            delay_changed = True
        if params.crossfade_length is not None:
            cfg.crossfade_length = params.crossfade_length
            delay_changed = True
        if params.extra_time is not None:
            cfg.extra_time = params.extra_time
        if params.index_rate is not None:
            engine.set_index_rate(params.index_rate)
        if params.rms_mix_rate is not None:
            engine.set_rms_mix_rate(params.rms_mix_rate)
        if params.f0method is not None:
            engine.set_f0method(params.f0method)
        if params.function is not None:
            engine.set_function(params.function)
        if params.I_noise_reduce is not None or params.O_noise_reduce is not None:
            engine.set_noise_reduce(
                (
                    params.I_noise_reduce
                    if params.I_noise_reduce is not None
                    else cfg.I_noise_reduce
                ),
                (
                    params.O_noise_reduce
                    if params.O_noise_reduce is not None
                    else cfg.O_noise_reduce
                ),
            )
            if params.I_noise_reduce is not None:
                delay_changed = True
        if delay_changed:
            engine._update_delay_time()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------- WebSocket 状态推送 ----------

@app.websocket("/ws")
async def ws_status(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            status = engine.get_status()
            await websocket.send_json(status)
            # 非阻塞探测客户端断开(客户端 close 时这里会抛异常退出循环)
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
            # 0.5s 推一次,客户端可据此刷新状态栏
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    # 连上先补发环形缓冲的历史日志
    with _LOG_LOCK:
        backlog = list(LOG_BUFFER)
        cursor = len(backlog)
    for line in backlog:
        await websocket.send_text(line)
    try:
        while True:
            # 增量推送新日志:先查长度,有新增才拷贝缓冲,避免每 0.2s 全量拷贝
            with _LOG_LOCK:
                if len(LOG_BUFFER) > cursor:
                    buf = list(LOG_BUFFER)
                else:
                    buf = None
            if buf:
                for line in buf[cursor:]:
                    await websocket.send_text(line)
                cursor = len(buf)
            # 非阻塞探测客户端断开(客户端 close 时抛异常退出循环)
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ---------- 入口 ----------

if __name__ == "__main__":
    port = find_available_port(7850)
    print(f"RVC-xiao server: http://127.0.0.1:{port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
