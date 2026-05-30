#!/usr/bin/env python3
"""
使用CLIP模型识别苹果图片内容（花/果实/叶子）
然后验证和修正JSON文件标注
"""

import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("警告: CLIP未安装，尝试安装...")
    print("请运行: pip install git+https://github.com/openai/CLIP.git")

def load_clip_model():
    """加载CLIP模型"""
    if not CLIP_AVAILABLE:
        return None, None
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        print(f"CLIP模型已加载，使用设备: {device}")
        return model, preprocess
    except Exception as e:
        print(f"加载CLIP模型失败: {e}")
        return None, None

def classify_with_clip(model, preprocess, img_path, device):
    """
    使用CLIP模型分类图片
    返回: 'flower', 'fruit', 或 'leaf'
    """
    try:
        # 定义类别文本
        text_descriptions = [
            "apple flower, pink or white apple blossom",
            "apple fruit, red or green apple",
            "apple leaf, green apple leaf"
        ]
        
        # 加载和预处理图片
        image = Image.open(img_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = clip.tokenize(text_descriptions).to(device)
        
        # 获取特征
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            
            # 计算相似度
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # 获取最高相似度的类别
            probs = similarity[0].cpu().numpy()
            class_idx = probs.argmax()
            
            class_names = ['flower', 'fruit', 'leaf']
            predicted_class = class_names[class_idx]
            confidence = probs[class_idx]
            
            return predicted_class, confidence, probs
            
    except Exception as e:
        print(f"  错误分类 {img_path.name}: {e}")
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
            # 更新所有相关字段
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
        print(f"  错误修正 {json_path.name}: {e}")
        return False

def main():
    dataset_root = Path('/home/yuhanlin/APP/AFSA/task1_fruit_classification/dataset')
    apple_dir = dataset_root / 'train' / 'apple'
    json_dir = apple_dir / 'json'
    
    if not json_dir.exists():
        print(f"错误：{json_dir} 不存在")
        return
    
    # 加载CLIP模型
    model, preprocess = load_clip_model()
    if model is None:
        print("无法加载CLIP模型，退出")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*70)
    print("使用CLIP模型验证和修正 train/apple 文件夹中的标注")
    print("="*70)
    print(f"JSON文件目录: {json_dir}")
    print()
    
    json_files = sorted(list(json_dir.glob('*.json')))
    total_files = len(json_files)
    
    print(f"找到 {total_files} 个JSON文件")
    print("开始使用CLIP模型识别和修正...")
    print()
    
    # 用户指定的正确标注（apple_000000 到 apple_000012 都是花）
    user_corrected = {}
    for i in range(13):
        img_name = f"apple_{i:06d}"
        user_corrected[img_name] = 'flower'
    
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
        
        # 如果用户已经指定了正确标注，使用用户的标注
        if img_stem in user_corrected:
            detected_type = user_corrected[img_stem]
            confidence = 1.0  # 用户确认的，置信度设为1.0
        else:
            # 使用CLIP模型识别
            result = classify_with_clip(model, preprocess, img_path, device)
            if result[0] is None:
                error_count += 1
                continue
            
            detected_type, confidence, probs = result
            
            # 如果置信度太低，可能需要人工审查
            if confidence < 0.4:
                low_confidence_count += 1
        
        # 如果检测到的类型与当前标注不同，修正
        if detected_type != current_type:
            change_key = f"{current_type}_to_{detected_type}"
            if change_key in stats:
                stats[change_key] += 1
            
            if fix_json_annotation(json_path, detected_type):
                fixed_count += 1
                if fixed_count <= 20:  # 显示前20个修正
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
    main()
