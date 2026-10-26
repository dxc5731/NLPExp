# NLPExp 实验演示前端

React + TypeScript + Vite + Recharts 本地应用，用于查看终端发起的真实 dev 推理与统一评分，以及执行三路线短文本实体识别。

完整安装、受控资源配置、启动、录像流程及输入限制见 [使用说明](../docs/demo/使用说明.md)。后端源码位于 [backend](../backend)，终端入口为 [scripts/run_dev_demo.py](../scripts/run_dev_demo.py)。

已有本机资源和环境配置时，在仓库根目录分别打开两个终端：

```powershell
.\scripts\start_demo_backend.cmd
```

```powershell
.\scripts\start_demo_frontend.cmd
```

浏览器访问 http://127.0.0.1:5173 。网页不会启动全量 dev 或训练。全量实验请在另一个终端手动执行：

```powershell
.\.venv-bert-crf\Scripts\python.exe -B -X utf8 scripts/run_dev_demo.py --split dev --methods rule,softmax,crf
```

首次检出建议使用 **main**：该分支包含规则、Softmax、统一评估模块等应用依赖。dxc 保留原分支文件范围；已按第一阶段增量补全的本地 dxc 工作区也可使用。模型、原始数据、本机路径配置和医疗文本没有随前端提交，不能把安装 npm 依赖等同于三模型已具备运行条件。
