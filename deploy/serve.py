"""服务器运行：python deploy/serve.py；GPU和端口由.env.models配置。"""

import os
from pathlib import Path

from dotenv import load_dotenv


def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env.models")
    os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=os.getenv("MODEL_GPU", "2"),
                      MODEL_ROOT=os.getenv("MODEL_ROOT", str(root / "models")), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    # 在导入torch/transformers前选择GPU；始终只有一个worker加载模型。
    import uvicorn
    from model_server import app

    uvicorn.run(app, host=os.getenv("MODEL_HOST", "127.0.0.1"), port=int(os.getenv("MODEL_PORT", "18082")))


if __name__ == "__main__":
    main()
