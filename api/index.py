from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入主应用
from main import app

# 导出应用供Vercel使用
app = app 