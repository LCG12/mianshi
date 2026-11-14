# 本地模型部署

意图、向量和重排模型共用一个FastAPI进程，按请求顺序推理；该进程还通过HTTP网关转发规划、回答与复核请求到独立LLM服务。

| 用途 | 模型 | HTTP接口 |
| --- | --- | --- |
| 意图 | Qwen3-1.7B，关闭思考 | `POST /intent` |
| 向量 | Qwen3-Embedding-0.6B，1024维 | `POST /embeddings` |
| 重排 | Qwen3-Reranker-0.6B | `POST /rerank` |
| 规划、回答与复核 | qwen3.5-27b | `POST /llm/v1/chat/completions` |
| token计数 | LLM对应的tokenizer与聊天模板 | `POST /llm/tokenize` |
| 健康检查 | 模型加载与GPU配置 | `GET /health` |

## 安装与启动

模型依赖见 [requirements.txt](requirements.txt)。在项目根目录创建独立环境并下载模型：

```bash
python -m venv .venv-models
source .venv-models/bin/activate
pip install -r deploy/requirements.txt
python deploy/download_models.py models
```

下载需要 `curl`，有 `aria2c` 时自动使用多连接下载。[models.lock.json](models.lock.json) 固定模型仓库版本、文件大小和SHA256；下载脚本跳过校验通过的文件并支持断点续传。

项目根目录的 `.env.models` 可配置GPU、监听地址、端口及LLM上游：

```dotenv
MODEL_GPU=2
MODEL_HOST=127.0.0.1
MODEL_PORT=18082
LLM_UPSTREAM=http://127.0.0.1:8018
```

`MODEL_GPU` 可填GPU序号或UUID；选中的GPU在进程内显示为 `cuda:0`。远程应用访问时，将 `MODEL_HOST` 设置为服务器可访问的监听地址。`MODEL_ROOT` 可覆盖默认的项目 `models/` 目录。

```bash
python deploy/serve.py
```

该命令前台运行，使用一个worker加载模型，按 `Ctrl+C` 停止。重启前停止旧进程，避免重复加载。Qwen3.5-27B上游服务需另外启动，并提供兼容的生成与token计数接口；其上下文窗口须与应用 `configs/models.yaml` 中的8192设置一致。

应用根目录 `.env` 配置各模型地址与名称，可参考 `.env.example`。LLM的base URL应指向网关的 `/llm`，例如 `http://127.0.0.1:18082/llm`；意图、向量与重排使用 `http://127.0.0.1:18082`。

## 请求协议

意图、向量、重排分别使用以下JSON请求：

```json
{"question":"保险期间是多久？","context":{}}
{"texts":["保险期间是多久？"],"purpose":"query"}
{"query":"保险期间是多久？","documents":["本合同的保险期间为终身。"]}
```

意图返回 `route/task/topics/execution_status`。`route` 包含 `insurance`、`assistant_info`、`chitchat`、`out_of_scope` 和 `unknown`；未知或分类失败时应用继续规划。意图采用提示式分类，业务标签仅作查询建议。

向量返回 `vectors/model/dimensions`，重排返回 `results[{index,score}]`。每批最多8条；输入最多6000字符，含模板后最多4096 tokens，超限拒绝。查询指令只在服务器编码查询时添加，文档编码不添加查询指令。

LLM网关只转发模型列表、非流式生成和token计数。应用规划、回答、复核的输出上限分别为1536、2048、768 tokens，关闭思考；单次模型调用时限为60秒，整轮请求时限为90秒。token计数须使用与生成相同的模型、聊天模板、思考开关和生成前缀；接口不可用、响应不合约或窗口不足时明确失败。

## 文件职责

- `download_models.py`：下载模型并核对锁定清单。
- `model_server.py`：实现意图、向量与重排接口。
- `llm_gateway.py`：转发独立LLM服务。
- `serve.py`：读取环境配置、选择GPU并启动服务。
- `insurance_qa/models/`：应用侧模型适配器。

应用端 `python main.py doctor` 检查配置与本地数据；模型服务的 `/health` 提供加载状态。
