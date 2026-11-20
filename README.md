# 反相修复台 · Phase-Fix

一个小型 FastAPI + NumPy 网页工具，用来修复电台转写素材里被反相左右声道抵消的台词。

上传 ≤ 60 秒、16 kHz、16 位 PCM、双声道 WAV 后，服务端：

1. 估计两声道之间的**固定时差**（假定至多 2 ms）；
2. 用**对齐后的短时相关性** + **直接下混的能量损失** 标出**疑似反相区间**；
3. 在网页上给出波形、疑似区间和可勾选/拖动的选区；
4. 输出一份与输入**等长**的单声道 WAV，**试听与下载是同一份字节**。

---

## 快速启动

### Docker Compose（推荐）

```bash
docker compose up --build
# 打开 http://localhost:8000
```

`docker-compose.yml` 内含健康检查（`GET /api/health`）、只读根文件系统、
`no-new-privileges`，仅暴露 8000 端口。停止：`docker compose down`。

### 本地（Python 3.11）

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 网页操作样例

1. 生成示例音频（可选，需要 NumPy）：

   ```bash
   python examples/make_samples.py examples/out
   ```

   会得到 `speech_inverted.wav`（1.5 ms 右声道延迟 + 两段反相）、
   `clean.wav`、`one_sided.wav`、`decorrelated.wav`、`no_delay_inverted.wav`。

2. 打开 <http://localhost:8000>，把 `speech_inverted.wav` 拖进上传框，点 **分析**。
   结果卡会显示：`固定时差 +1.5 ms`、`疑似反相区间 2 段`，以及原始立体声 A/B 播放器。

3. 在 **选区与修复** 里点 **全选疑似区间**（或在时间轴上点击橙色区域）。
   每个选区有两行数字框，可把边界精确到采样点。

4. 拖动区间中部可平移，拖动两端可改边界。让两个选区重叠或首尾相接，
   点 **应用修复并试听**：服务端会把它们合并成一个区间处理，
   状态栏会显示合并后的采样范围。

5. 试听结果，点 **下载单声道 WAV**。播放器的 `src` 与下载链接指向同一个
   `Blob`，两者内容完全相同。

6. 点 **清除选区** 再生成，得到的就是纯左右平均的直接下混；
   如果时差未能确认，界面会禁用选区编辑，只允许导出直接下混。

---

## 处理规则

### 上传校验

只接受：WAV (RIFF/WAVE)、双声道、16 位 PCM、16 000 Hz、1 帧以上、≤ 60 秒。
任何一项不符合都会返回 422 并给出中文原因（文件体上限 16 MB）。

### 时差估计（`dsp.estimate_delay`）

- 在 ±4 ms 内做**分块归一化互相关**，块长 250 ms、半块重叠，加 Hann 窗。
  对每块取 `|r|` 再加权平均，因此某一段被反相不会把整段时差证据抵消掉，
  且某一声道静音时不会把结果拉向 0。
- "确认"必须同时满足：
  - 峰值 `|r| ≥ 0.80`；
  - 峰值是内部极大值，且**比搜索边界高 0.10 以上**（否则真实最优可能在范围外）；
  - 比零时差和 ±1 ms 外的次峰都高 0.10 以上；
  - 两声道共同活跃时长 ≥ 300 ms；
  - `|lag| ≤ 2 ms`（假定范围），超过即视为无法确认。
- 无法确认时：`delay_confirmed=false`，界面禁用选区并只输出直接下混。

> `delay` 的符号约定：**正值表示右声道比左声道晚**（`right[n] ≈ left[n-d]`）。

### 疑似反相检测（`dsp.detect_regions`）

只在**两声道同时活跃**的 20 ms 帧（10 ms 步进，静音阈值 −50 dBFS）上计算，
且只有同时满足以下两条才标记：

```
loss_db = 10·log10( (|L|²+|R|²)/4 / |(L+R)/2|² ) ≥ 5 dB
corr    = <L,R> / (|L||R|)                     ≤ −0.45
```

其中 `L`/`R` 已按时差对齐。随后闭合 ≤ 20 ms 的检测空洞、丢弃 < 80 ms 的孤立片段、
合并重叠区间。**静音、单边有声、时差未确认时都不会产生标记。**

### 修复（`dsp.repair`）

对每个合并后的选区 `[a, b)`（重叠或相邻的选区先合并）：

1. **只在区间内部**取原始采样：把早到的一路延迟 `|d|` 个采样，露出的
   `|d|` 个位置补零，绝不从区间外借用采样；
2. 翻转晚到一路的极性，与另一路取平均：`0.5·(L_aligned − R_aligned)`；
3. 与直接下混 `0.5·(L+R)` 做 5 ms smoothstep 交叉淡化，因此选区首尾采样
   与直接下混**完全相等**；
4. 若区间短到容不下对齐填充（`length < 2|d|+2`），保持该区间为直接下混，
   避免补零造成咔嗒声。

**区间外的每一个采样都严格等于 `0.5·(L+R)`**，与原位下混逐位一致
（测试用 `np.array_equal` 断言）。

### 输出

- 单声道、16 位 PCM、与输入同采样率、**帧数与输入完全相同**（含"0 选区"和"时差未确认"两种降级路径）。
- 响应头 `X-Result-Mode: repaired | downmix`、`X-Total-Samples`、
  `X-Merged-Ranges` 供前端与测试核对。
- 同一输入 + 同一选区两次渲染字节相同，所以试听和下载天然一致。

---

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 网页 |
| GET | `/api/health` | 健康检查 |
| POST | `/api/analyze` | `multipart/form-data` 字段 `file`；返回 `job_id`、`analysis`、`waveform`、`downmix_url`、`original_url` |
| GET | `/api/jobs/{id}/downmix.wav` | 直接下混 |
| GET | `/api/jobs/{id}/original.wav` | 原始立体声（A/B 参考） |
| POST | `/api/jobs/{id}/render` | JSON `{"spans":[{"start_sample":0,"end_sample":1600}]}`；返回等长单声道 WAV |

会话保存在内存中，30 分钟过期，最多 16 个；服务重启即清空。
`/api/docs` 提供交互式 OpenAPI 文档。

---

## 测试

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. python -m pytest tests -q          # DSP + API，40 项
node tests/frontend_smoke.js                    # 前端，31 项（需 jsdom）
```

前端冒烟测试用 jsdom 驱动真实 `app.js`，覆盖上传、分析、勾选疑似区间、
数字框编辑、渲染、下载链接与播放器同源、清空后降级为 downmix、
画布拖拽新建/移动/改边界、点击疑似区间、重置等交互。

---

## 已知限制 / 未完成

- **仅含 UI 的降级提示，无批量任务队列**：分析/渲染在进程内线程池执行，
  会话状态在内存里；多副本部署需要外部存储与任务队列。
- **固定时差假设**：只估计单一常量时差（±2 ms）。时变延迟、漂移或超过
  2 ms 的偏移会被判为"无法确认"并降级为直接下混，不做动态对齐。
- **检测阈值是启发式**：5 dB 损失 / −0.45 相关 / 80 ms 最短区间按广播语音
  调过；低声压或强混响素材可能漏检或把正常立体声效果误标（可在
  `app/dsp.py` 顶部常量调整）。
- **修复采用单采样点极性翻转**：不是频域去相关，也不是自适应滤波；
  若反相并非严格逐采样镜像，修复后仍会有残留。
- **前端未做真实浏览器自动化**：仅用 jsdom 做了 DOM/交互冒烟测试，
  没有在真实 Chromium/Firefox 中跑视觉与音频播放回归。
- **Docker 镜像未在本环境实际构建**：环境中没有 Docker 守护进程，
  已验证 Dockerfile/Compose 的依赖解析、隔离目录导入与 YAML 结构，
  但 `docker compose up --build` 需在有 Docker 的机器上跑一遍。
- 上传文件与结果不落盘，也不写入任何日志内容；只在内存保留 30 分钟。
