#!/bin/bash
# 切换到正确的环境并运行推理

echo "========================================"
echo "切换到 cosmos-transfer1 环境并运行推理"
echo "========================================"
echo ""

# 激活 cosmos-transfer1 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-transfer1

# 验证环境
echo "当前环境: $CONDA_DEFAULT_ENV"
echo ""

# 检查 transformer_engine
echo "检查 transformer_engine..."
python -c "import transformer_engine; print(f'✓ transformer_engine 版本: {transformer_engine.__version__}')" 2>&1

if [ $? -ne 0 ]; then
    echo "❌ transformer_engine 未安装或版本不对"
    echo ""
    echo "请先安装正确版本："
    echo "  conda activate cosmos-transfer1"
    echo "  pip install https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/v1.1.0/transformer_engine-1.13.0+cu128.torch271-cp312-cp312-linux_x86_64.whl"
    exit 1
fi

echo ""
echo "========================================"
echo "环境验证通过，开始运行推理..."
echo "========================================"
echo ""

# 切换到工作目录
cd /home/invs/Path-planning20251027/cosmos-transfer1/cosmos-transfer1

# 运行推理
PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
    cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir checkpoints \
    --video_save_folder outputs/carla_basic \
    --controlnet_specs examples/cosmos/client/cosmos_data_config.toml \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus 1
