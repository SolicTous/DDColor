from torchvision.transforms import ToTensor, Grayscale


def color_enhacne_blend(x, factor=1.2):
    """
    Color enhancement by increasing saturation.
    
    Args:
        x: Input tensor image (C, H, W), range [0, 1]
        factor: Saturation enhancement factor. 
                factor=1.0 means no change,
                factor>1.0 increases saturation,
                factor<1.0 decreases saturation.
    
    Returns:
        Enhanced tensor image
    """
    x_g = Grayscale(3)(x)
    # Correct formula: interpolate between grayscale and original
    # out = grayscale + (original - grayscale) * factor
    out = x_g + (x - x_g) * factor
    out = out.clamp(0, 1)
    return out