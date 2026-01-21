#!/usr/bin/env python3
"""
思维链合成数据质检工具 - 启动脚本
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """检查必要的依赖包"""
    try:
        import flask
        print("✓ Flask 已安装")
    except ImportError:
        print("✗ Flask 未安装，请运行: pip install -r requirements.txt")
        return False
    
    try:
        import requests
        print("✓ requests 已安装")
    except ImportError:
        print("✗ requests 未安装，请运行: pip install requests")
        return False
    
    return True

def main():
    print("=" * 60)
    print("思维链合成数据质检工具")
    print("=" * 60)
    
    # 检查依赖
    if not check_dependencies():
        print("\n请先安装必要的依赖包")
        sys.exit(1)
    
    # 启动Web应用
    try:
        from app import app
        print("\n✅ 启动成功!")
        print("🌐 访问地址: http://localhost:5001")
        print("📊 API健康检查: http://localhost:5001/api/health")
        print("⏹️  按 Ctrl+C 停止服务")
        print("-" * 60)
        
        app.run(debug=True, host='0.0.0.0', port=5001)
        
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("请检查:")
        print("1. 依赖包是否安装: pip install -r requirements.txt")
        print("2. 端口5001是否被占用")
        sys.exit(1)

if __name__ == "__main__":
    main()