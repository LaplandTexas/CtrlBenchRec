🛠️ 环境准备
克隆项目

安装依赖

📊 数据说明
本项目主要使用以下数据集：

MovieLens 1M: 经典的电影推荐数据集。

Amazon Toys & Games: 包含项目信息和用户购买记录。

注意: 模型权重文件（如 twhin-bert-base）应放置在 rec_models/ 目录下，该目录已被 Git 忽略。

🚀 快速上手
该项目环境python版本为3.10

1.终端运行以下命令进行环境配置：
pip install -r requirements

2.运行../runner/load_twhin_bert.py脚本，加载twhin-bert-base推荐模型

3.进入deepseek-api开放平台https://platform.deepseek.com/usage，获取您的api密钥

4.可运行不同阶段的相应脚本，运行时配置环境变量为您所获取的密钥，如DEEPSEEK_API_KEY=xxxxx
智能体初始化元信息构建阶段，可运行脚本../runner/user_profile_initialize.py。
智能体与推荐系统交互/融合阶段，可运行脚本../runner/epoch.py,我们提供了在sasrec模型上进行交互/融合的可运行函数。
评估阶段，可运行脚本../runner/evaluation.py,自行设置需要评估的profile。
