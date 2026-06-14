# 🚀 使用你师兄采集的CARLA数据运行Cosmos-Transfer1

## 📁 数据准备情况

你师兄已经采集好的数据在 `cosmos/` 目录下：

```
cosmos/
├── rgb.mp4                     # RGB原始视频（输入）
├── depth.mp4                   # 深度图视频（控制信号）
├── semantic_segmentation.mp4   # 语义分割视频（控制信号）
├── rgb_edges.mp4               # 边缘检测视频（控制信号）
├── rgb_masked.mp4              # 遮罩RGB
└── instance_segmentation.mp4   # 实例分割
```

## 🎯 使用方式

### 方式一：本地推理（直接在服务器上运行）

如果你有 GPU 服务器并已安装 Cosmos-Transfer1：

```bash
cd /home/invs/Path-planning20251027/cosmos-transfer1/cosmos-transfer1

# 1. 激活环境（根据你的安装方式）
conda activate cosmos-transfer1  # 或你的环境名

# 2. 运行推理 - 基础场景
PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
  cosmos_transfer1/diffusion/inference/transfer.py \
  --checkpoint_dir checkpoints \
  --video_save_folder outputs/carla_basic \
  --controlnet_specs examples/cosmos/client/cosmos_data_config.toml \
  --offload_text_encoder_model \
  --offload_guardrail_models \
  --num_gpus 1

# 3. 运行推理 - 雨天场景
PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
  cosmos_transfer1/diffusion/inference/transfer.py \
  --checkpoint_dir checkpoints \
  --video_save_folder outputs/carla_rain \
  --controlnet_specs examples/cosmos/client/cosmos_data_rain.toml \
  --offload_text_encoder_model \
  --offload_guardrail_models \
  --num_gpus 1

# 4. 运行推理 - 夜晚场景
PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
  cosmos_transfer1/diffusion/inference/transfer.py \
  --checkpoint_dir checkpoints \
  --video_save_folder outputs/carla_night \
  --controlnet_specs examples/cosmos/client/cosmos_data_night.toml \
  --offload_text_encoder_model \
  --offload_guardrail_models \
  --num_gpus 1
```

**输出位置**：`outputs/carla_basic/`, `outputs/carla_rain/`, `outputs/carla_night/`

### 方式二：通过 REST API（使用远程服务器）

如果有部署好的 Cosmos 服务器：

```bash
cd /home/invs/Path-planning20251027/cosmos-transfer1/cosmos-transfer1/examples/cosmos/client

# 1. 激活客户端环境
conda activate carla-cosmos-client  # 或你的环境名

# 2. 发送请求 - 基础场景
python cosmos_client.py http://SERVER_IP:PORT \
  cosmos_data_config.toml \
  --output outputs/result_basic.mp4

# 3. 发送请求 - 雨天场景
python cosmos_client.py http://SERVER_IP:PORT \
  cosmos_data_rain.toml \
  --output outputs/result_rain.mp4

# 4. 发送请求 - 夜晚场景
python cosmos_client.py http://SERVER_IP:PORT \
  cosmos_data_night.toml \
  --output outputs/result_night.mp4
```

## ⚙️ 配置说明

已创建的配置文件：

1. **cosmos_data_config.toml** - 基础配置，保持原场景风格
2. **cosmos_data_rain.toml** - 雨天场景变换
3. **cosmos_data_night.toml** - 夜晚场景变换

### 关键参数调整

在 `.toml` 文件中可以调整：

```toml
# 控制场景变化程度
sigma_max = 78        # 0-20: 微调, 40-60: 中等, 78: 大变化, ≥80: 全新生成

# 控制信号强度
[edge]
control_weight = 0.5  # 边缘约束 (0.0-1.0)

[seg]
control_weight = 0.9  # 语义约束 (0.0-1.0)

[depth]
control_weight = 0.3  # 深度约束 (0.0-1.0)

# 生成质量
num_steps = 35        # 扩散步数，越多越细腻但越慢 (20-50)
guidance = 7.0        # CFG强度，越大越符合prompt (5.0-10.0)
seed = 1024           # 随机种子，固定可复现

# 修改场景描述
prompt = "你的场景描述..."
```

## 📊 控制信号组合策略

| 场景需求 | 推荐组合 | 权重建议 |
|---------|---------|---------|
| 保持几何结构 | edge + depth | edge: 0.5-0.7, depth: 0.3-0.5 |
| 保持语义布局 | seg + edge | seg: 0.8-0.9, edge: 0.4-0.6 |
| 天气变化 | seg + depth + edge | seg: 0.9, depth: 0.3, edge: 0.5 |
| 光照变化（日夜） | seg + depth | seg: 0.9, depth: 0.4-0.6 |
| 风格迁移 | edge + vis | edge: 0.5, vis: 0.3-0.5 |

## 🎨 创建自定义场景

复制并修改配置文件：

```bash
cp cosmos_data_config.toml cosmos_data_foggy.toml
# 编辑 cosmos_data_foggy.toml，修改 prompt 为雾天描述
```

示例 prompt（雾天）：
```toml
prompt = "Captured from a camera mounted on a vehicle's roof, we see an urban intersection shrouded in thick fog. Visibility is significantly reduced, with buildings and distant objects fading into a gray haze. Street lights create soft, diffused glows through the mist. Vehicle headlights pierce the fog with beams of light, while taillights appear as red halos. The wet road surface reflects the limited light sources. Lane markings are barely visible through the dense fog. The atmosphere is eerie and quiet, with muted colors and soft edges on all objects. cinematic, photorealistic, ultra high quality, ultra high resolution, high fidelity, high definition, expert cinematography, realism, structure of the scene maintained, vehicles maintained, foggy weather."
```

## 🔍 故障排查

### 1. 缺少文件错误
```
FileNotFoundError: cosmos/rgb.mp4
```
**解决**：确保当前目录在 `examples/cosmos/client/`

### 2. CUDA内存不足
**解决**：
- 减少 `num_steps`（35 → 20）
- 使用 `--offload_text_encoder_model`
- 使用 `--offload_guardrail_models`

### 3. 生成结果不理想
**解决**：
- 调整 `sigma_max`（降低保留更多原始特征）
- 调整控制权重（提高关键控制的权重）
- 修改 prompt 更详细描述
- 尝试不同 seed

## 📝 批量处理脚本

创建批量生成不同场景的脚本：

```bash
#!/bin/bash
cd /home/invs/Path-planning20251027/cosmos-transfer1/cosmos-transfer1

CONFIGS=(
  "examples/cosmos/client/cosmos_data_config.toml"
  "examples/cosmos/client/cosmos_data_rain.toml"
  "examples/cosmos/client/cosmos_data_night.toml"
)

NAMES=("basic" "rain" "night")

for i in "${!CONFIGS[@]}"; do
  echo "生成场景: ${NAMES[$i]}"
  PYTHONPATH=$(pwd) torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
    cosmos_transfer1/diffusion/inference/transfer.py \
    --checkpoint_dir checkpoints \
    --video_save_folder "outputs/carla_${NAMES[$i]}" \
    --controlnet_specs "${CONFIGS[$i]}" \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus 1
done

echo "所有场景生成完成！"
```

保存为 `batch_generate.sh`，运行：
```bash
chmod +x batch_generate.sh
./batch_generate.sh
```
