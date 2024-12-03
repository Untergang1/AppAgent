# AppAgent

图数据库：

提前安装并启用neo4j，用环境变量配置neo4j用户名和密码：
```angular2html
export NEO4J_USERNAME=xxx
export NEO4J_PASSWORD=xxx
```

代码默认使用Gemini api，提前用环境变量配置api key:
```angular2html
export GEMINI_API_KEY=xxx
```

运行：

```angular2html
python run.py --task 'turn on wifi' 
```

