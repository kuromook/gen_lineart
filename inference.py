import torch
from torchvision import transforms
from PIL import Image
import os
from trainer import AttnResUNet  # trainer.pyからモデルを読み込み

def load_model(checkpoint_path, device='cuda'):
    model = AttnResUNet(in_ch=1, out_ch=1, base_ch=64, use_self_attn=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    print(f"✅ Loaded checkpoint from {checkpoint_path}")
    return model

def preprocess_image(img_path, size=(256,256)):
    tr = transforms.Compose([
        transforms.Resize(size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    img = Image.open(img_path).convert('RGB')
    return tr(img).unsqueeze(0)  # 1x1xHxW tensor

def postprocess_and_save(tensor, save_path):
    # tensor: 1x1xHxW in [0,1]
    img = tensor.squeeze(0).squeeze(0).detach().cpu()  # HxW
    img = transforms.ToPILImage()(img)
    img.save(save_path)
    print(f"💾 Saved output to {save_path}")

@torch.no_grad()
def inference(model, input_path, output_path, device='cuda'):
    img = preprocess_image(input_path)
    img = img.to(device)

    with torch.amp.autocast(device_type='cuda'):
        output = model(img)

    postprocess_and_save(output, output_path)

if __name__ == "__main__":
    # ======== 設定 =========
    checkpoint_path = "checkpoints/model_epoch_30.pth"  # 学習済みモデル
    input_dir = "test/rough"   # テスト用の下絵
    output_dir = "results"             # 出力先
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint_path, device)

    # ======== 推論 =========
    rough_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for fname in rough_files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        inference(model, in_path, out_path, device)

