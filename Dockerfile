FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 解决国内网络问题，配置 Debian 镜像源（可选，如果拉包慢可以加上）
# RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list

# 安装系统级依赖（有些向量化库可能需要编译工具）
RUN apt-get update && apt-get install -y gcc g++ libmagic-dev && rm -rf /var/lib/apt/lists/*

# 先拷贝依赖清单，利用 Docker 缓存机制
COPY requirements.txt .

# 使用清华源加速 pip 安装
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 拷贝项目其余文件
COPY . .

# 暴露端口
EXPOSE 8000

# 默认启动命令（会被 docker-compose 覆盖）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]