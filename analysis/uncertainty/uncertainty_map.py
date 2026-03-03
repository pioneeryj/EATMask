# aleatoric uncertainty
import torch
import torch.nn.functional as F

# epistemic uncertainty
# epistemic uncertainty by monte carlo dropout
import torch
import torch.nn as nn
import torch.nn.functional as F
import encoder3D
from decoder3D import LightDecoder

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")



class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = True


class STUNet(nn.Module):
    def __init__(self, input_channels, num_classes, depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None, enable_deep_supervision=True, dropout_ratio=0.5):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.final_nonlin = lambda x: x
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.dropout_ratio=0.5
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool = len(pool_op_kernel_sizes)

        assert num_pool == len(dims) - 1

        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(
            BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in
              range(depth[0] - 1)])
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool + 1):
            stage = nn.Sequential(BasicResBlock(dims[d - 1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d],
                                                stride=self.pool_op_kernel_sizes[d - 1], use_1x1conv=True),
                                  *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
                                    for _ in range(depth[d] - 1)])
            self.conv_blocks_context.append(stage)

        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1 - u], dims[-2 - u], pool_op_kernel_sizes[-1 - u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(BasicResBlock(dims[-2 - u] * 2, dims[-2 - u], self.conv_kernel_sizes[-2 - u],
                                                self.conv_pad_sizes[-2 - u], use_1x1conv=True),
                                  *[BasicResBlock(dims[-2 - u], dims[-2 - u], self.conv_kernel_sizes[-2 - u],
                                                  self.conv_pad_sizes[-2 - u]) for _ in range(depth[-2 - u] - 1)])
            self.conv_blocks_localization.append(stage)

        # outputs
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(dims[-2 - ds], num_classes, kernel_size=1))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)

    def forward(self, x):
        skips = []
        seg_outputs = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self.decoder.deep_supervision:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]



class BasicResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False, dropout_ratio=0.5):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_ratio)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.dropout(y)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)

class Upsample_Layer_nearest(nn.Module):
    def __init__(self, input_channels, output_channels, pool_op_kernel_size, mode='nearest', dropout_ratio=0.5):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        self.dropout = nn.Dropout3d(p=dropout_ratio)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        x = self.dropout(x)
        return x

pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1,1,1]]
conv_kernel_sizes =  [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

# STUNet_B
model = STUNet(
    input_channels=1,
    num_classes=1,
    depth=[1, 1, 1, 1, 1, 1],
    dims=[32, 64, 128, 256, 512, 512],
    pool_op_kernel_sizes=pool_op_kernel_sizes,
    conv_kernel_sizes=conv_kernel_sizes,
    enable_deep_supervision=True,
    dropout_ratio=0.5
)

# Pretrained weights 로드
pretrained_path = "path/to/pretrained_weights.pth"  # pretrained weights 경로
pretrained_weights = torch.load(pretrained_path, map_location=torch.device('cpu'))

# 모델에 pretrained weights 로드 (strict=False로 일부 매칭 안 되는 부분 무시)
model.load_state_dict(pretrained_weights, strict=False)

# 모델을 GPU로 이동
model = model.to(device)


# Monte Carlo Dropout으로 Logits 샘플 생성
def generate_logits_with_mc_dropout(model, input_data, num_samples):
    """
    model: PyTorch model with Dropout layers
    input_data: Input tensor of shape (B, C_in, H, W)
    num_samples: Number of Monte Carlo samples
    Returns: List of logits tensors, each of shape (B, 1, H, W)
    """
    model.train()  # Dropout 활성화
    logits_list = []
    with torch.no_grad():  # Gradient 계산 비활성화
        for _ in range(num_samples):
            logits = model(input_data)  # Forward pass
            logits_list.append(logits)
    return logits_list

# Epistemic Uncertainty 계산
def calculate_epistemic_uncertainty(logits_list):
    """
    logits_list: List of T tensors, each of shape (B, 1, H, W)
    Returns: Epistemic uncertainty map of shape (B, H, W)
    """
    # Stack logits along a new dimension (T samples)
    logits_stack = torch.stack(logits_list, dim=0)  # Shape: (T, B, 1, H, W)

    # Apply Sigmoid to convert logits to probabilities
    probs_stack = torch.sigmoid(logits_stack)  # Shape: (T, B, 1, H, W)

    # Mean and variance of probabilities across T samples
    mean_probs = probs_stack.mean(dim=0)  # Shape: (B, 1, H, W)
    var_probs = probs_stack.var(dim=0)  # Shape: (B, 1, H, W)

    # Squeeze to remove the single channel dimension
    epistemic_uncertainty = var_probs.squeeze(1)  # Shape: (B, H, W)
    return epistemic_uncertainty

# Example usage
if __name__ == "__main__":
    # 입력 및 모델 정의
    B, C_in, H, W = 2, 3, 64, 64  # Batch size, input channels, height, width
    T = 5  # Monte Carlo 샘플 수

    # 임의의 입력 데이터
    input_data = torch.randn(B, C_in, H, W)  # Shape: (B, C_in, H, W)

    # 모델 생성
    model = SimpleModel()

    # Monte Carlo Dropout으로 logits 샘플 생성
    logits_list = generate_logits_with_mc_dropout(model, input_data, T)

    # Epistemic Uncertainty 계산
    epistemic_uncertainty_map = calculate_epistemic_uncertainty(logits_list)

    # 결과 확인
    print("Epistemic Uncertainty Map Shape:", epistemic_uncertainty_map.shape)  # (B, H, W)

    # Epistemic Uncertainty 시각화 (예: 첫 번째 배치의 첫 번째 샘플)
    import matplotlib.pyplot as plt
    plt.imshow(epistemic_uncertainty_map[0].cpu().numpy(), cmap="hot")
    plt.colorbar()
    plt.title("Epistemic Uncertainty Map")
    plt.show()


# aleatoric uncertainty
# 1. Softmax를 사용하여 클래스 확률 계산
def calculate_softmax(logits):
    """
    logits: Tensor of shape (B, C, H, W)
    B: Batch size, C: Number of classes, H: Height, W: Width
    Returns: Softmax probabilities (B, C, H, W)
    """
    return F.softmax(logits, dim=1)

# 2. 픽셀 단위 조건부 엔트로피 계산
def calculate_conditional_entropy(probabilities):
    """
    probabilities: Tensor of shape (B, C, H, W)
    Returns: Conditional entropy map of shape (B, H, W)
    """
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-8
    log_probs = torch.log(probabilities + epsilon)
    entropy = -torch.sum(probabilities * log_probs, dim=1)  # Sum over classes
    return entropy  # Shape: (B, H, W)

# 3. Aleatoric Uncertainty 계산 (Monte Carlo 샘플링 기반)
def calculate_aleatoric_uncertainty(logits_list):
    """
    logits_list: List of T tensors, each of shape (B, C, H, W)
    T: Number of Monte Carlo samples (or ensemble models)
    Returns: Aleatoric uncertainty map of shape (B, H, W)
    """
    T = len(logits_list)
    assert T > 0, "Logits list must contain at least one sample."

    # 1. Softmax probabilities for each sample
    prob_samples = [calculate_softmax(logits) for logits in logits_list]  # List of (B, C, H, W)

    # 2. Conditional entropy for each sample
    entropy_samples = [calculate_conditional_entropy(probs) for probs in prob_samples]  # List of (B, H, W)

    # 3. Average conditional entropy across samples
    aleatoric_uncertainty_map = torch.stack(entropy_samples, dim=0).mean(dim=0)  # Shape: (B, H, W)
    return aleatoric_uncertainty_map

# Example usage
if __name__ == "__main__":
    # Random logits for demonstration purposes
    B, C, H, W = 2, 4, 64, 64  # Batch size, number of classes, height, width
    T = 5  # Number of Monte Carlo samples

    # Generate random logits (e.g., from an ensemble of models or MC dropout)
    logits_list = [torch.randn(B, C, H, W) for _ in range(T)]

    # Calculate Aleatoric Uncertainty Map
    aleatoric_uncertainty_map = calculate_aleatoric_uncertainty(logits_list)

    # Print shape of the resulting map
    print("Aleatoric Uncertainty Map Shape:", aleatoric_uncertainty_map.shape)  # (B, H, W)