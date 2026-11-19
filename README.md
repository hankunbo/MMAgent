# MMAgent

这是一个只包含源代码的 MMAgent 快照，便于代码审查、复现和后续工程协作。

## 内容范围

仓库保留 Python 实现、算法工具运行器、Agent 组件、应用入口、前端模板、接口契约 schema 和测试代码。代码中的注释用于解释局部实现。

本仓库不包含本地运行记录、数据集或病例内容、数据库及其导出物、模型权重或检查点、演示文稿、研究方案文档、环境凭据和隐私数据。运行时需要的数据、模型和服务由使用者在仓库外提供。

## 本地开发

需要 Python 3.12。开发依赖和常用检查命令如下：

```bash
make install-dev
make format-check
make lint
make test
```

由于运行时数据、数据库和模型文件不随仓库分发，完整运行需要在仓库外准备相应环境。环境变量模板见 [`.env.example`](.env.example)。

## 目录

- `Algorithm_Tool/`：算法工具与 CPU 测量实现
- `agents/`：Agent 组件与契约 schema
- `apps/`：应用入口、服务和展示模板
- `packages/`：可复用的 Python 包与 schema
- `Seg_Model/`：分割工具的代码入口；模型文件不随仓库分发
- `tests/`：测试代码
