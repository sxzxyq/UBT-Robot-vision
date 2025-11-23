import sys
import os

# 打印初始的 sys.path
print("--- Initial sys.path ---")
import pprint
pprint.pprint(sys.path)
print("-" * 20)

# 关键：手动添加路径，就像我们用 PYTHONPATH 做的那样
image_pipeline_path = '/home/nvidia/Workspace/imagepipeline_conda/ImagePipeline'
if image_pipeline_path not in sys.path:
    sys.path.insert(0, image_pipeline_path)

# 打印修改后的 sys.path
print("\n--- Modified sys.path ---")
pprint.pprint(sys.path)
print("-" * 20)

# 现在，进行最终的导入测试
print("\nAttempting to import 'from imagepipeline.model_adapters import OpenAIAdapter'...")

try:
    from imagepipeline.model_adapters import OpenAIAdapter
    print("\n\033[92mSUCCESS! OpenAIAdapter was imported successfully.\033[0m")
    print(f"Type of imported object: {type(OpenAIAdapter)}")
except Exception as e:
    print(f"\n\033[91mFAILURE! The import failed.\033[0m")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Details: {e}")

    # 如果导入失败，进行更深层次的诊断
    print("\n--- Deep Diagnosis ---")
    try:
        print("Trying to import just 'imagepipeline'...")
        import imagepipeline
        print("SUCCESS: 'import imagepipeline' worked.")
        print(f"Found imagepipeline at: {imagepipeline.__file__}")

        print("\nTrying to import 'imagepipeline.model_adapters'...")
        import imagepipeline.model_adapters
        print("SUCCESS: 'import imagepipeline.model_adapters' worked.")
        print(f"Found model_adapters at: {imagepipeline.model_adapters.__file__}")

    except Exception as e_deep:
        print(f"FAILURE during deep diagnosis: {e_deep}")