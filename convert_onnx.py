import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def _compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """Быстрый SSIM для проверки качества"""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = window_size // 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=pad)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=pad)

    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=pad) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=pad) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=pad) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return (num / den).mean().item()

def progressive_half_conversion(
        model: nn.Module,
        test_input: torch.Tensor,
        device: torch.device,
        ssim_threshold: float = 0.995
) -> nn.Module:
    """
    Послойно пробует перевести слои в FP16.
    Если SSIM падает ниже порога → откатывает в FP32.
    ИСПРАВЛЕНО: корректное замыкание хуков, безопасный rollback, оригинальный skip_types.
    """
    model.eval()
    test_input = test_input.to(device)

    # 1. Эталон в FP32
    with torch.no_grad():
        ref_out = model(test_input).float()

    # ОРИГИНАЛЬНЫЙ список исключений (вернул, чтобы не терять модули)
    skip_types = (
        nn.ModuleList, nn.Sequential, nn.Identity, nn.Dropout,
        nn.GELU, nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Softmax,
        nn.AdaptiveAvgPool2d, nn.PixelShuffle, nn.Unfold, nn.AvgPool2d, nn.MaxPool2d
    )

    target_modules = [
        (n, m) for n, m in model.named_modules()
        if list(m.parameters()) and not isinstance(m, skip_types)
    ]
    print(f"🔍 Найдено {len(target_modules)} параметризованных модулей. Запуск...")

    kept, reverted = 0, 0

    for name, module in tqdm(target_modules, desc="Проверка слоёв"):
        orig_dtype = next(module.parameters()).dtype
        target_dtype = torch.float16

        # Сохраняем полное состояние модуля для гарантированного отката
        orig_state = {k: v.clone() for k, v in module.state_dict().items()}

        # 2. Переводим параметры в half
        module.to(target_dtype)

        # 3. Хуки с ПРАВИЛЬНЫМ захватом переменной через default-argument
        def pre_hook(m, inp, td=target_dtype):
            def cast(x):
                return x.to(td) if isinstance(x, torch.Tensor) else x
            if isinstance(inp, torch.Tensor):
                return cast(inp)
            elif isinstance(inp, (tuple, list)):
                return type(inp)(cast(x) for x in inp)
            elif isinstance(inp, dict):
                return {k: cast(v) for k, v in inp.items()}
            return inp

        def post_hook(m, inp, out):
            def cast(x):
                return x.to(torch.float32) if isinstance(x, torch.Tensor) else x
            if isinstance(out, torch.Tensor):
                return cast(out)
            elif isinstance(out, (tuple, list)):
                return type(out)(cast(x) for x in out)
            elif isinstance(out, dict):
                return {k: cast(v) for k, v in out.items()}
            return out

        h_pre = module.register_forward_pre_hook(pre_hook)
        h_post = module.register_forward_hook(post_hook)

        def rollback():
            """Полный откат: тип + веса + буферы + удаление хуков"""
            module.to(orig_dtype)
            module.load_state_dict(orig_state, strict=False)
            h_pre.remove()
            h_post.remove()

        try:
            with torch.no_grad():
                new_out = model(test_input).float()

            if torch.isnan(new_out).any() or torch.isinf(new_out).any():
                raise RuntimeError("NaN/Inf detected")

            ssim_val = _compute_ssim(ref_out, new_out)

            if ssim_val >= ssim_threshold:
                print(f"✅ [FP16 OK] {name:50s} | SSIM: {ssim_val:.5f}")
                kept += 1
                # Хуки остаются навсегда → модель готова к запуску и ONNX-экспорту
            else:
                print(f"❌ [Откат]   {name:50s} | SSIM: {ssim_val:.5f} < {ssim_threshold}")
                rollback()
                reverted += 1

        except Exception as e:
            print(f"⚠️  [Ошибка]  {name:50s} | {str(e)[:100]}")
            rollback()
            reverted += 1

    print(f"\n📊 Итог: {kept} слоёв переведено в FP16, {reverted} оставлено в FP32")
    return model

print('torch.cuda.is_available()', torch.cuda.is_available())
print('torch.backends.cudnn.enabled', torch.backends.cudnn.enabled)

class MODEL(nn.Module):
    def __init__(self, generator):
        super(MODEL, self).__init__()
        self.generator = generator

    def forward(self, input):
        input = input.permute(0, 3, 1, 2)
        output = self.generator(input)
        output = output.permute(0, 2, 3, 1)
        return output

def convert_onnx(pt_model, fin_path, device, test_input):
    model = MODEL(pt_model)
    model.to(device)
    model.eval()

    # 3. Запустите прогрессивную конвертацию
    model = progressive_half_conversion(
        model,
        test_input=test_input,
        device=device,
        ssim_threshold=0.995,  # Можно поднять до 0.998 для строгости или опустить до 0.99
    )

    dummy_tensor_1 = torch.randn(1, 512, 512, 3, device=device, requires_grad=True)
    inputs = ['x:0']
    outputs = ['Identity:0']
    # dynamic_axes = {'input': {0: 'batches',
    #                           1: 'height',
    #                           2: 'width'},
    #                 'output': {0: 'batches',
    #                            1: 'height',
    #                            2: 'width'}}

    print('export start')
    torch.onnx.export(model, dummy_tensor_1, fin_path,
                      export_params=True, do_constant_folding=True,
                      # dynamic_axes=dynamic_axes,
                      input_names=inputs, output_names=outputs, opset_version=19,  # 14
                      verbose=False)
    print('export done')