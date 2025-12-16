import os

def find_test_image():
    # 定义可能的位置
    search_paths = [
        # 当前目录及子目录
        '.',
        './',
        os.getcwd(),
        
        # 上级目录
        '..',
        '../',
        os.path.dirname(os.getcwd()),
        
        # 常见的数据目录
        './data',
        './images',
        './test_images',
        './img',
        './input',
        
        # 特定路径（根据您的项目结构）
        'D:/REPOS/Ai-project/classify_pytorch',
        'D:/REPOS/Ai-project/classify_pytorch/',
        'D:\\REPOS\\Ai-project\\classify_pytorch',
    ]
    
    # 可能的文件名变体
    possible_filenames = [
        'test.jpg',
        'test.png',
        'test.jpeg',
        'test.bmp',
        'test.tif',
        'test.tiff',
        'Test.jpg',
        'TEST.JPG',
        'sample.jpg',
        'example.jpg',
        'digit.jpg',
        'number.jpg',
    ]
    
    found_files = []
    
    # 搜索所有组合
    for base_path in search_paths:
        # 检查路径是否存在
        if not os.path.exists(base_path):
            continue
            
        # 检查每个文件名
        for filename in possible_filenames:
            full_path = os.path.join(base_path, filename)
            
            # 检查文件是否存在
            if os.path.isfile(full_path):
                file_size = os.path.getsize(full_path)
                
                # 检查文件是否可读
                try:
                    import cv2
                    img = cv2.imread(full_path)
                    readable = img is not None
                except:
                    readable = False
                
                found_files.append({
                    'path': full_path,
                    'size': file_size,
                    'readable': readable,
                    'exists': True
                })
    
    # 也搜索所有jpg文件（即使不是test.jpg）
    for base_path in search_paths:
        if os.path.exists(base_path):
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        full_path = os.path.join(root, file)
                        
                        # 检查是否已包含
                        already_found = any(f['path'] == full_path for f in found_files)
                        if not already_found:
                            found_files.append({
                                'path': full_path,
                                'size': os.path.getsize(full_path),
                                'readable': None,  # 稍后检查
                                'exists': True
                            })
    
    return found_files

# 运行搜索
found_files = find_test_image()

if found_files:
    print(f"\n找到 {len(found_files)} 个可能的图像文件:")
    print("-" * 80)
    print(f"{'序号':<5} {'路径':<60} {'大小(字节)':<12} {'可读':<8}")
    print("-" * 80)
    
    for i, file_info in enumerate(found_files, 1):
        # 检查可读性
        if file_info['readable'] is None:
            import cv2
            img = cv2.imread(file_info['path'])
            file_info['readable'] = img is not None
        
        readable_str = "是" if file_info['readable'] else "否"
        print(f"{i:<5} {file_info['path']:<60} {file_info['size']:<12} {readable_str:<8}")
else:
    print("未找到任何图像文件！")