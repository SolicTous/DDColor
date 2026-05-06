import onnx

model = onnx.load(r"D:\Models\Diffusers\SimianLuoDreamshaper_v7\onnx_alter_3\model.onnx", load_external_data=True)
onnx.save(model, r"D:\Models\Diffusers\SimianLuoDreamshaper_v7\onnx_alter_3\model.onnx", save_as_external_data=False)
