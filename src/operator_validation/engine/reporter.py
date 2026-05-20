"""报告生成器"""
import json
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

class ReportGenerator:
    def __init__(self, results: Dict[str, Any], config: Dict[str, Any]):
        self.results = results
        self.config = config

    def to_json(self, path: str | Path) -> None:
        with open(path, 'w') as f:
            json.dump({"config": self.config, "results": self._serialize(self.results)}, f, indent=2, default=str)
        print(f"📄 JSON 报告已保存: {path}")

    def to_markdown(self, path: str | Path) -> None:
        lines = ["# 算子验证报告", "", "## 配置", f"- Reference Backend: `{self.config.get('reference', 'N/A')}`", f"- 精度阈值: rtol={self.config.get('rtol', 'N/A')}, atol={self.config.get('atol', 'N/A')}", "", "## 汇总", f"- 总测试用例: {self.results.get('total', 0)}", f"- ✅ PASS: {self.results.get('passed', 0)}", f"- ❌ FAIL: {self.results.get('failed', 0)}", f"- ⚠️ WARNING: {self.results.get('warning', 0)}", "", "## 详细结果", ""]
        by_op = self.results.get("by_operator", {})
        for op_name, summary in by_op.items():
            icon = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️"}.get(summary.get("overall", "?"), "❓")
            lines.append(f"### {icon} {op_name}")
            lines.append(f"- 状态: {summary.get('overall', 'UNKNOWN')}")
            lines.append(f"- PASS: {summary.get('pass', 0)}, FAIL: {summary.get('fail', 0)}, WARNING: {summary.get('warning', 0)}")
            lines.append("")
        with open(path, 'w') as f:
            f.write("\n".join(lines))
        print(f"📄 Markdown 报告已保存: {path}")

    def _serialize(self, results: Any) -> Any:
        if hasattr(results, 'tolist'):
            return results.tolist()
        elif isinstance(results, dict):
            return {k: self._serialize(v) for k, v in results.items()}
        elif isinstance(results, (list, tuple)):
            return [self._serialize(item) for item in results]
        elif hasattr(results, '__dict__'):
            return {k: self._serialize(v) for k, v in vars(results).items()}
        else:
            return results
