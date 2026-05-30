#!/usr/bin/env python3
"""
使用视觉模型识别苹果图片内容（花/果实/叶子）
支持CLIP或torchvision预训练模型
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

# 尝试导入CLIP
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# 尝试使用torchvision
try:
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

def load_clip_model():
    """加载CLIP模型"""
    if not CLIP_AVAILABLE:
        return None, None, None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        print(f"使用CLIP模型 (ViT-B/32)，设备: {device}")
        return model, preprocess, device
    except Exception as e:
        print(f"加载CLIP模型失败: {e}")
        return None, None, None

def load_torchvision_model():
    """加载torchvision预训练模型作为备选"""
    if not TORCHVISION_AVAILABLE:
        return None, None, None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 使用ResNet50作为特征提取器
        model = models.resnet50(pretrained=True)
        model.eval()
        model = model.to(device)
        
        # 预处理
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print(f"使用ResNet50模型，设备: {device}")
        return model, preprocess, device
    except Exception as e:
        print(f"加载torchvision模型失败: {e}")
        return None, None, None

def classify_with_clip(model, preprocess, img_path, device):
    """使用CLIP模型分类"""
    try:
        text_descriptions = [
            "apple flower, pink or white apple blossom",
            "apple fruit, red or green apple",
            "apple leaf, green apple leaf"
        ]
        
        image = Image.open(img_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = clip.tokenize(text_descriptions).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            probs = similarity[0].cpu().numpy()
            class_idx = probs.argmax()
            
            class_names = ['flower', 'fruit', 'leaf']
            predicted_class = class_names[class_idx]
            confidence = probs[class_idx]
            
            return predicted_class, confidence, probs
            
    except Exception as e:
        return None, 0.0, None

def classify_with_torchvision(model, preprocess, img_path, device):
    """使用torchvision模型分类（基于颜色和纹理特征）"""
    try:
        image = Image.open(img_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model(image_tensor)
            # 这里我们需要一个简单的分类器，暂时使用启发式方法
            # 实际上应该训练一个分类器，但这里我们结合图像分析
            pass
        
        # 回退到简单的图像分析
        import cv2
        import numpy as np
        
        img = cv2.imread(str(img_path))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height, width = img.shape[:2]
        
        # 红色区域
        red_mask = cv2.inRange(img_hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_ratio = np.sum(red_mask > 0) / (width * height)
        
        # 粉色/白色区域
        pink_mask = cv2.inRange(img_hsv, np.array([140, 30, 150]), np.array([170, 100, 255]))
        white_mask = cv2.inRange(img_hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        pink_white_ratio = np.sum(cv2.bitwise_or(pink_mask, white_mask) > 0) / (width * height)
        
        # 绿色区域
        green_mask = cv2.inRange(img_hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
        green_ratio = np.sum(green_mask > 0) / (width * height)
        
        brightness = np.mean(img_hsv[:, :, 2])
        
        # 简单判断
        if (pink_white_ratio > 0.2) or (brightness > 160 and pink_white_ratio > 0.1):
            return 'flower', 0.7, None
        elif red_ratio > 0.15:
            return 'fruit', 0.7, None
        elif green_ratio > 0.3:
            return 'leaf', 0.7, None
        else:
            return 'fruit', 0.5, None
            
    except Exception as e:
        return None, 0.0, None

def fix_json_annotation(json_path, detected_type):
    """修正JSON文件的标注"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'images' not in data or len(data['images']) == 0:
            return False
        
        img_info = data['images'][0]
        current_type = img_info.get('content_type', '')
        
        if detected_type and detected_type != current_type:
            img_info['content_type'] = detected_type
            img_info['subcategory'] = detected_type
            
            if 'annotations' in data and len(data['annotations']) > 0:
                data['annotations'][0]['content_type'] = detected_type
            
            if 'categories' in data and len(data['categories']) > 0:
                category = data['categories'][0]
                category['subcategory'] = detected_type
                if detected_type == 'flower':
                    category['name'] = 'apple_flower'
                elif detected_type == 'fruit':
                    category['name'] = 'apple_fruit'
                elif detected_type == 'leaf':
                    category['name'] = 'apple_leaf'
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        
        return False
        
    except Exception as e:
        return False

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    apple_dir = dataset_root / 'train' / 'apple'
    json_dir = apple_dir / 'json'
    
    # 用户指定的文件（已修正）
    user_specified = {f"apple_{i:06d}" for i in range(13)}
    
    # 加载模型
    model, preprocess, device = load_clip_model()
    use_clip = model is not None
    
    if not use_clip:
        print("CLIP不可用，尝试使用torchvision模型...")
        model, preprocess, device = load_torchvision_model()
        if model is None:
            print("错误：无法加载任何模型")
            return
    
    print("="*70)
    print("使用视觉模型验证和修正 train/apple 文件夹中的标注")
    print("="*70)
    print(f"JSON文件目录: {json_dir}")
    print()
    
    json_files = sorted(list(json_dir.glob('*.json')))
    total_files = len(json_files)
    
    print(f"找到 {total_files} 个JSON文件")
    print("开始使用模型识别和修正...")
    print()
    
    fixed_count = 0
    error_count = 0
    unchanged_count = 0
    low_confidence_count = 0
    
    stats = {
        'flower_to_fruit': 0,
        'flower_to_leaf': 0,
        'fruit_to_flower': 0,
        'fruit_to_leaf': 0,
        'leaf_to_flower': 0,
        'leaf_to_fruit': 0,
    }
    
    for json_path in tqdm(json_files, desc="处理进度"):
        img_stem = json_path.stem
        
        # 跳过用户已指定的文件（已修正）
        if img_stem in user_specified:
            unchanged_count += 1
            continue
        
        img_files = list(apple_dir.glob(f"{img_stem}.*"))
        img_files = [f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        if not img_files:
            error_count += 1
            continue
        
        img_path = img_files[0]
        
        # 读取当前JSON标注
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            current_type = data['images'][0].get('content_type', 'unknown')
        except:
            error_count += 1
            continue
        
        # 使用模型识别
        if use_clip:
            result = classify_with_clip(model, preprocess, img_path, device)
        else:
            result = classify_with_torchvision(model, preprocess, img_path, device)
        
        if result[0] is None:
            error_count += 1
            continue
        
        detected_type, confidence, probs = result
        
        if confidence < 0.4:
            low_confidence_count += 1
        
        # 如果检测到的类型与当前标注不同，修正
        if detected_type != current_type:
            change_key = f"{current_type}_to_{detected_type}"
            if change_key in stats:
                stats[change_key] += 1
            
            if fix_json_annotation(json_path, detected_type):
                fixed_count += 1
                if fixed_count <= 30:  # 显示前30个修正
                    print(f"  修正: {json_path.name} ({current_type} -> {detected_type}, 置信度: {confidence:.2f})")
            else:
                error_count += 1
        else:
            unchanged_count += 1
    
    print()
    print("="*70)
    print("处理完成！")
    print("="*70)
    print(f"总文件数: {total_files}")
    print(f"修正文件数: {fixed_count}")
    print(f"未变更文件数: {unchanged_count}")
    print(f"错误文件数: {error_count}")
    print(f"低置信度文件数: {low_confidence_count} (可能需要人工审查)")
    print()
    print("修正统计：")
    for change_type, count in stats.items():
        if count > 0:
            print(f"  - {change_type}: {count}")
    print("="*70)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
        sys.exit(1)
