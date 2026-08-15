"""
实时变声引擎 — 纯音频逻辑,无任何 GUI 依赖。

从 realtime_gui.py 中拆出,可被 FreeSimpleGUI 界面 / FastAPI 控制服务 /
任意前端复用。音频回调线程只写状态槽(带锁),UI 线程读取状态槽,互不阻塞。
"""

import os
import sys
import threading
import time
import traceback

# 本文件位于 xiaozi/ 子目录,项目根目录需加入 sys.path(configs/infer/i18n/tools 包在项目根)
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat

from configs.config import Config
from infer import rtrvc as rvc_for_realtime
from i18n.i18n import I18nAuto
from tools.cuda_graph import cuda_graph_enabled, run_cuda_graph
from tools.torchgate import TorchGate

i18n = I18nAuto()


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


class EngineConfig:
    """实时变声引擎参数(原 GUIConfig)。"""

    def __init__(self):
        self.pth_path = ""
        self.index_path = ""
        self.pitch = 0
        self.formant = 0.0
        self.sr_type = "sr_model"
        self.block_time = 0.25  # s
        self.threhold = -60
        self.crossfade_length = 0.05
        self.extra_time = 2.5
        self.I_noise_reduce = False
        self.O_noise_reduce = False
        self.rms_mix_rate = 0.0
        self.index_rate = 0.0
        self.f0method = "rmvpe"
        self.sg_hostapi = ""
        self.sg_wasapi_exclusive = False
        self.sg_input_device = ""
        self.sg_output_device = ""
        self.samplerate = 0
        self.channels = 0


class RealtimeEngine:
    """实时变声引擎:设备枚举、模型加载、音频流处理与推理。"""

    def __init__(self, config: Config):
        self.config = config
        self.engine_config = EngineConfig()
        self.function = "vc"  # "vc" 输出变声 / "im" 输入监听
        self.rvc = None
        self.stream = None
        self.hostapis = None
        self.input_devices = None
        self.output_devices = None
        self.input_devices_indices = None
        self.output_devices_indices = None
        self._running = False

        # 线程安全状态槽:音频回调线程写入,UI/服务线程读取
        self._status_lock = threading.Lock()
        self._status = {
            "running": False,
            "infer_time_ms": 0,
            "samplerate": 0,
            "delay_time_ms": 0,
        }

        self.update_devices()

    # ---------- 状态槽 ----------

    def get_status(self):
        """返回状态快照(线程安全)。"""
        with self._status_lock:
            return dict(self._status)

    def is_running(self):
        return self._running

    # ---------- 配置持久化(原 GUI.load 逻辑) ----------

    def apply_saved_config(self):
        """从 configs/config.json 读取并应用已保存的设置到 engine_config。
        BAT GUI 通过 load_settings() + set_values() 完成同样的操作;
        服务器启动时需要调用此方法,否则 engine_config 只有硬编码默认值。"""
        data = self.load_settings()
        cfg = self.engine_config
        cfg.pth_path = data.get("pth_path", cfg.pth_path)
        cfg.index_path = data.get("index_path", cfg.index_path)
        cfg.sg_hostapi = data.get("sg_hostapi", cfg.sg_hostapi)
        cfg.sg_wasapi_exclusive = data.get("sg_wasapi_exclusive", cfg.sg_wasapi_exclusive)
        cfg.sg_input_device = data.get("sg_input_device", cfg.sg_input_device)
        cfg.sg_output_device = data.get("sg_output_device", cfg.sg_output_device)
        cfg.sr_type = data.get("sr_type", cfg.sr_type)
        cfg.threhold = data.get("threhold", cfg.threhold)
        cfg.pitch = data.get("pitch", cfg.pitch)
        cfg.formant = data.get("formant", cfg.formant)
        cfg.block_time = data.get("block_time", cfg.block_time)
        cfg.crossfade_length = data.get("crossfade_length", cfg.crossfade_length)
        cfg.extra_time = data.get("extra_time", cfg.extra_time)
        cfg.I_noise_reduce = data.get("I_noise_reduce", cfg.I_noise_reduce)
        cfg.O_noise_reduce = data.get("O_noise_reduce", cfg.O_noise_reduce)
        cfg.rms_mix_rate = data.get("rms_mix_rate", cfg.rms_mix_rate)
        cfg.index_rate = data.get("index_rate", cfg.index_rate)
        f0 = data.get("f0method", cfg.f0method)
        if f0 in ("pm", "rmvpe", "fcpe"):
            cfg.f0method = f0
        # 应用设备路由(确保音频流使用正确的输入/输出设备)
        try:
            self.set_devices(cfg.sg_input_device, cfg.sg_output_device)
        except Exception:
            pass  # 设备不存在时跳过,启动时前端会提示用户选择

    def load_settings(self):
        """从 configs/config.json 读取设置,返回展开后的 dict。"""
        realtime_config_path = os.path.join("configs", "config.json")
        data = None
        try:
            import json

            from tools.file_io import read_text

            data = json.loads(read_text(realtime_config_path))
            data["sr_model"] = data["sr_type"] == "sr_model"
            data["sr_device"] = data["sr_type"] == "sr_device"
            if data.get("f0method") not in ("pm", "rmvpe", "fcpe"):
                data["f0method"] = "rmvpe"
            data["pm"] = data["f0method"] == "pm"
            data["rmvpe"] = data["f0method"] == "rmvpe"
            data["fcpe"] = data["f0method"] == "fcpe"
            if data["sg_hostapi"] in self.hostapis:
                self.update_devices(hostapi_name=data["sg_hostapi"])
                if (
                    data["sg_input_device"] not in self.input_devices
                    or data["sg_output_device"] not in self.output_devices
                ):
                    self.update_devices()
                    data["sg_hostapi"] = self.hostapis[0]
                    data["sg_input_device"] = self.input_devices[
                        self.input_devices_indices.index(sd.default.device[0])
                    ]
                    data["sg_output_device"] = self.output_devices[
                        self.output_devices_indices.index(sd.default.device[1])
                    ]
            else:
                data["sg_hostapi"] = self.hostapis[0]
                data["sg_input_device"] = self.input_devices[
                    self.input_devices_indices.index(sd.default.device[0])
                ]
                data["sg_output_device"] = self.output_devices[
                    self.output_devices_indices.index(sd.default.device[1])
                ]
        except Exception:
            data = self.default_settings()
        return data

    def default_settings(self):
        """生成默认设置 dict(设备列表已就绪)。"""
        import json

        realtime_config_path = os.path.join("configs", "config.json")
        try:
            with open(realtime_config_path, "w", encoding="utf8") as j:
                data = {
                    "pth_path": "",
                    "index_path": "",
                    "sg_hostapi": self.hostapis[0],
                    "sg_wasapi_exclusive": False,
                    "sg_input_device": self.input_devices[
                        self.input_devices_indices.index(sd.default.device[0])
                    ],
                    "sg_output_device": self.output_devices[
                        self.output_devices_indices.index(sd.default.device[1])
                    ],
                    "sr_type": "sr_model",
                    "threhold": -60,
                    "pitch": 0,
                    "formant": 0.0,
                    "index_rate": 0,
                    "rms_mix_rate": 0,
                    "block_time": 0.25,
                    "crossfade_length": 0.05,
                    "extra_time": 2.5,
                    "f0method": "rmvpe",
                }
                data["sr_model"] = data["sr_type"] == "sr_model"
                data["sr_device"] = data["sr_type"] == "sr_device"
                data["pm"] = data["f0method"] == "pm"
                data["rmvpe"] = data["f0method"] == "rmvpe"
                data["fcpe"] = data["f0method"] == "fcpe"
                json.dump(data, j)
        except Exception:
            data = {}
        return data

    # ---------- 设备管理 ----------

    def update_devices(self, hostapi_name=None):
        """获取设备列表。引擎运行时不允许重载(会破坏音频流),需先停止。"""
        if self._running:
            raise RuntimeError(
                "引擎正在运行,请先停止再刷新设备列表"
            )
        sd._terminate()
        sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        self.hostapis = [hostapi["name"] for hostapi in hostapis]
        if hostapi_name not in self.hostapis:
            hostapi_name = self.hostapis[0]
        self.input_devices = [
            d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices = [
            d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.input_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]
        self.output_devices_indices = [
            d["index"] if "index" in d else d["name"]
            for d in devices
            if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name
        ]

    def set_devices(self, input_device, output_device):
        """设置输入/输出设备。"""
        sd.default.device[0] = self.input_devices_indices[
            self.input_devices.index(input_device)
        ]
        sd.default.device[1] = self.output_devices_indices[
            self.output_devices.index(output_device)
        ]
        printt(i18n("输入设备：%s:%s"), str(sd.default.device[0]), input_device)
        printt(i18n("输出设备：%s:%s"), str(sd.default.device[1]), output_device)

    def get_device_samplerate(self):
        return int(
            sd.query_devices(device=sd.default.device[0])["default_samplerate"]
        )

    def get_device_channels(self):
        max_input_channels = sd.query_devices(device=sd.default.device[0])[
            "max_input_channels"
        ]
        max_output_channels = sd.query_devices(device=sd.default.device[1])[
            "max_output_channels"
        ]
        return min(max_input_channels, max_output_channels, 2)

    # ---------- 启动 / 停止 ----------

    def start_vc(self):
        """加载模型并启动音频流(需先配置 engine_config)。"""
        torch.cuda.empty_cache()
        self.rvc = rvc_for_realtime.RVC(
            self.engine_config.pitch,
            self.engine_config.formant,
            self.engine_config.pth_path,
            self.engine_config.index_path,
            self.engine_config.index_rate,
            self.config,
            self.rvc,
        )
        self.engine_config.samplerate = (
            self.rvc.tgt_sr
            if self.engine_config.sr_type == "sr_model"
            else self.get_device_samplerate()
        )
        with self._status_lock:
            self._status["samplerate"] = self.engine_config.samplerate
        self.engine_config.channels = self.get_device_channels()
        self.zc = self.engine_config.samplerate // 100
        self.block_frame = (
            int(
                np.round(
                    self.engine_config.block_time
                    * self.engine_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(
                np.round(
                    self.engine_config.crossfade_length
                    * self.engine_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(
                np.round(
                    self.engine_config.extra_time
                    * self.engine_config.samplerate
                    / self.zc
                )
            )
            * self.zc
        )
        self.input_wav = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
        )
        self.sola_den_kernel = torch.ones(
            1,
            1,
            self.sola_buffer_frame,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.config.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.engine_config.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.config.device)
        if self.rvc.tgt_sr != self.engine_config.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.engine_config.samplerate,
                dtype=torch.float32,
            ).to(self.config.device)
        else:
            self.resampler2 = None
        # Bundled torch.istft is not CUDA Graph-capturable, so TorchGate
        # stays eager while resampling and RVC inference still use graphs.
        self.tg = TorchGate(
            sr=self.engine_config.samplerate,
            n_fft=4 * self.zc,
            prop_decrease=0.9,
        ).to(self.config.device)
        self.prewarm_cuda_graph()
        self.start_stream()

    def prewarm_cuda_graph(self):
        if not cuda_graph_enabled(self.config.device):
            return
        try:
            printt(i18n("正在预热CUDA Graph"))
            samples = self.input_wav_res.shape[0]
            phase = torch.arange(
                samples, device=self.config.device, dtype=torch.float32
            )
            probe = 0.05 * torch.sin(2 * np.pi * 220.0 * phase / 16000.0)
            self.input_wav_res.copy_(probe)

            if self.engine_config.I_noise_reduce:
                short = self.input_wav[
                    -self.sola_buffer_frame - self.block_frame :
                ].unsqueeze(0)
                self.tg(short, self.input_wav.unsqueeze(0))

            resample_input = self.input_wav[-self.block_frame - 2 * self.zc :]
            run_cuda_graph(
                self.resampler,
                "realtime-input-resample",
                lambda audio: self.resampler(audio),
                resample_input,
            )

            inferred = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.engine_config.f0method,
            )
            if self.resampler2 is not None:
                inferred = run_cuda_graph(
                    self.resampler2,
                    "realtime-output-resample",
                    lambda audio: self.resampler2(audio),
                    inferred,
                )
            if self.engine_config.O_noise_reduce:
                self.tg(inferred.unsqueeze(0), self.output_buffer.unsqueeze(0))
            torch.cuda.synchronize(self.config.device)
            printt(i18n("CUDA Graph预热完成"))
        except Exception:
            printt(traceback.format_exc())
        finally:
            self.input_wav.zero_()
            self.input_wav_denoise.zero_()
            self.input_wav_res.zero_()
            self.output_buffer.zero_()
            self.sola_buffer.zero_()
            self.nr_buffer.zero_()
            self.rvc.cache_pitch.zero_()
            self.rvc.cache_pitchf.zero_()

    def start_stream(self):
        if not self._running:
            self._running = True
            with self._status_lock:
                self._status["running"] = True
            try:
                if (
                    "WASAPI" in self.engine_config.sg_hostapi
                    and self.engine_config.sg_wasapi_exclusive
                ):
                    extra_settings = sd.WasapiSettings(exclusive=True)
                else:
                    extra_settings = None
                self.stream = sd.Stream(
                    callback=self.audio_callback,
                    blocksize=self.block_frame,
                    samplerate=self.engine_config.samplerate,
                    channels=self.engine_config.channels,
                    dtype="float32",
                    extra_settings=extra_settings,
                )
                self.stream.start()
                self._update_delay_time()
            except Exception:
                # 流打开失败时回滚状态,否则 _running 残留导致后续无法操作
                self._running = False
                with self._status_lock:
                    self._status["running"] = False
                if self.stream is not None:
                    try:
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None
                raise

    def _update_delay_time(self):
        """更新算法延迟估算(与 BAT GUI 公式一致)。"""
        if self.stream is None:
            return
        cfg = self.engine_config
        delay = (
            self.stream.latency[-1]
            + cfg.block_time
            + cfg.crossfade_length
            + 0.01
        )
        if cfg.I_noise_reduce:
            delay += min(cfg.crossfade_length, 0.04)
        with self._status_lock:
            self._status["delay_time_ms"] = int(delay * 1000)

    def stop_stream(self):
        if self._running:
            self._running = False
            with self._status_lock:
                self._status["running"] = False
                self._status["delay_time_ms"] = 0
            if self.stream is not None:
                self.stream.abort()
                self.stream.close()
                self.stream = None

    # ---------- 参数热更新 ----------

    def set_pitch(self, pitch):
        self.engine_config.pitch = pitch
        if self.rvc is not None:
            self.rvc.change_key(pitch)

    def set_formant(self, formant):
        self.engine_config.formant = formant
        if self.rvc is not None:
            self.rvc.change_formant(formant)

    def set_index_rate(self, index_rate):
        self.engine_config.index_rate = index_rate
        if self.rvc is not None:
            self.rvc.change_index_rate(index_rate)

    def set_rms_mix_rate(self, rms_mix_rate):
        self.engine_config.rms_mix_rate = rms_mix_rate

    def set_f0method(self, f0method):
        if f0method in ("pm", "rmvpe", "fcpe"):
            self.engine_config.f0method = f0method

    def set_function(self, function):
        if function in ("vc", "im"):
            self.function = function

    def set_noise_reduce(self, input_noise_reduce, output_noise_reduce):
        self.engine_config.I_noise_reduce = input_noise_reduce
        self.engine_config.O_noise_reduce = output_noise_reduce

    # ---------- 音频处理 ----------

    def audio_callback(self, indata, outdata, frames, times, status):
        """
        音频处理(实时回调,运行在 sounddevice 音频线程)。
        """
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)
        if self.engine_config.threhold > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(
                y=indata, frame_length=4 * self.zc, hop_length=self.zc
            )[:, 2:]
            self.rms_buffer[:] = indata[-4 * self.zc :]
            indata = indata[2 * self.zc - self.zc // 2 :]
            db_threhold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0]
                < self.engine_config.threhold
            )
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc : (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2 :]
        self.input_wav[: -self.block_frame] = self.input_wav[
            self.block_frame :
        ].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(
            self.config.device
        )
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()
        # input noise reduction and resampling
        if self.engine_config.I_noise_reduce:
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[
                self.block_frame :
            ].clone()
            input_wav = self.input_wav[
                -self.sola_buffer_frame - self.block_frame :
            ]
            input_wav = self.tg(
                input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)
            ).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += (
                self.nr_buffer * self.fade_out_window
            )
            self.input_wav_denoise[-self.block_frame :] = input_wav[
                : self.block_frame
            ]
            self.nr_buffer[:] = input_wav[self.block_frame :]
            resample_input = self.input_wav_denoise[
                -self.block_frame - 2 * self.zc :
            ]
            self.input_wav_res[-self.block_frame_16k - 160 :] = run_cuda_graph(
                self.resampler,
                "realtime-input-resample",
                lambda audio: self.resampler(audio),
                resample_input,
            )[160:]
        else:
            resample_input = self.input_wav[-indata.shape[0] - 2 * self.zc :]
            self.input_wav_res[
                -160 * (indata.shape[0] // self.zc + 1) :
            ] = run_cuda_graph(
                self.resampler,
                "realtime-input-resample",
                lambda audio: self.resampler(audio),
                resample_input,
            )[160:]
        # infer
        if self.function == "vc":
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.engine_config.f0method,
            )
            if self.resampler2 is not None:
                infer_wav = run_cuda_graph(
                    self.resampler2,
                    "realtime-output-resample",
                    lambda audio: self.resampler2(audio),
                    infer_wav,
                )
        elif self.engine_config.I_noise_reduce:
            infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
        else:
            infer_wav = self.input_wav[self.extra_frame :].clone()
        # output noise reduction
        if self.engine_config.O_noise_reduce and self.function == "vc":
            self.output_buffer[: -self.block_frame] = self.output_buffer[
                self.block_frame :
            ].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[
                -self.block_frame :
            ]
            infer_wav = self.tg(
                infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
            ).squeeze(0)
        # volume envelop mixing
        if self.engine_config.rms_mix_rate < 1 and self.function == "vc":
            if self.engine_config.I_noise_reduce:
                input_wav = self.input_wav_denoise[self.extra_frame :]
            else:
                input_wav = self.input_wav[self.extra_frame :]
            rms1 = librosa.feature.rms(
                y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms1 = torch.from_numpy(rms1).to(self.config.device)
            rms1 = F.interpolate(
                rms1.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms2 = torch.from_numpy(rms2).to(self.config.device)
            rms2 = F.interpolate(
                rms2.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(rms1 / rms2, 1.0 - self.engine_config.rms_mix_rate)
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        conv_input = infer_wav[
            None, None, : self.sola_buffer_frame + self.sola_search_frame
        ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                self.sola_den_kernel,
            )
            + 1e-8
        )
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        printt(i18n("SOLA偏移：%d"), int(sola_offset))
        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += (
            self.sola_buffer * self.fade_out_window
        )
        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]
        outdata[:] = (
            infer_wav[: self.block_frame]
            .repeat(self.engine_config.channels, 1)
            .t()
            .cpu()
            .numpy()
        )
        total_time = time.perf_counter() - start_time
        if self._running:
            with self._status_lock:
                self._status["infer_time_ms"] = int(total_time * 1000)
        printt(i18n("推理耗时：%.2f秒"), total_time)
