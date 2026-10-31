# 使用轻量级的 Python 3.10 镜像（最稳定兼容）
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 1. 替换 Debian 系统源为阿里云（解决国内构建时 apt-get 网络超时问题）
# 兼容 Debian 11 和 Debian 12 的源文件路径
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
    sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list 2>/dev/null || true

# 2. 安装系统级编译依赖（gcc/g++ 用于编译部分 Python C 扩展，libmagic 用于文档解析）
RUN apt-get update && apt-get install -y gcc g++ libmagic-dev && rm -rf /var/lib/apt/lists/*

# 3. 先拷贝依赖清单，利用 Docker 缓存机制（代码改动不会导致重新装包）
COPY requirements.txt .

# 4. 使用清华源加速 pip 安装（因为之前已经精简过 requirements，这里不会拉取 nvidia 毒瘤包）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    rm -rf /tmp/* /var/tmp/*

# 5. 拷贝项目其余所有文件
COPY . .

# 暴露 FastAPI 端口
EXPOSE 8000

# 默认启动命令（会被 docker-compose.yml 覆盖）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]