#!/bin/bash
# 验证数据文件路径是否正确

echo "========================================"
echo "验证 Cosmos 数据文件路径"
echo "========================================"
echo ""

cd /home/invs/Path-planning20251027/cosmos-transfer1/cosmos-transfer1

echo "当前目录: $(pwd)"
echo ""

# 从配置文件中提取路径
echo "检查配置文件中的路径..."
echo ""

CONFIG_FILES=(
    "examples/cosmos/client/cosmos_data_config.toml"
    "examples/cosmos/client/cosmos_data_rain.toml"
    "examples/cosmos/client/cosmos_data_night.toml"
)

for config in "${CONFIG_FILES[@]}"; do
    echo "=== $config ==="
    
    if [ ! -f "$config" ]; then
        echo "  ✗ 配置文件不存在"
        continue
    fi
    
    # 提取 input_video_path
    video_path=$(grep "^input_video_path" "$config" | sed 's/.*= *"\(.*\)".*/\1/')
    if [ -n "$video_path" ]; then
        if [ -f "$video_path" ]; then
            size=$(du -h "$video_path" | cut -f1)
            echo "  ✓ RGB视频: $video_path ($size)"
        else
            echo "  ✗ RGB视频缺失: $video_path"
        fi
    fi
    
    # 提取控制输入路径
    control_paths=$(grep "^input_control" "$config" | sed 's/.*= *"\(.*\)".*/\1/')
    for path in $control_paths; do
        control_type=$(basename "$path" .mp4)
        if [ -f "$path" ]; then
            size=$(du -h "$path" | cut -f1)
            echo "  ✓ 控制信号: $path ($size)"
        else
            echo "  ✗ 控制信号缺失: $path"
        fi
    done
    echo ""
done

echo "========================================"
echo "✓ 验证完成"
echo ""
echo "如果所有文件都显示 ✓，说明路径配置正确！"
echo ""
echo "现在可以运行推理命令："
echo "  cd /home/invs/Path-planning20251027/cosmos-transfer1/cosmos-transfer1"
echo "  PYTHONPATH=\$(pwd) torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \\"
echo "    cosmos_transfer1/diffusion/inference/transfer.py \\"
echo "    --checkpoint_dir checkpoints \\"
echo "    --video_save_folder outputs/carla_test \\"
echo "    --controlnet_specs examples/cosmos/client/cosmos_data_config.toml \\"
echo "    --offload_text_encoder_model \\"
echo "    --offload_guardrail_models \\"
echo "    --num_gpus 1"
echo "========================================"
