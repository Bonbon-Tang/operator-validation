"""主入口"""
from __future__ import annotations
import argparse
from pathlib import Path
from .config import FullConfig
from .engine.validator import ValidatorEngine
from .engine.reporter import ReportGenerator

def parse_args():
    parser = argparse.ArgumentParser(description="算子验证框架 - 多后端精度与性能对比")
    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--operators", "-o", nargs="+", default=None, help="要测试的算子列表")
    parser.add_argument("--shapes", "-s", nargs="+", choices=["small", "medium", "large"], default=None, help="测试规模")
    parser.add_argument("--report-json", type=str, default=None, help="JSON 报告输出路径")
    parser.add_argument("--report-md", type=str, default=None, help="Markdown 报告输出路径")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 60)
    print("🔬 算子验证框架")
    print("=" * 60)
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    config = FullConfig.from_file(config_path)
    print(f"\n📋 配置: {config_path}")
    print(f"   后端: {[b.name for b in config.backends]}")
    print(f"   算子: {[op.name for op in config.operators]}")
    print(f"   精度: rtol={config.validation.rtol}, atol={config.validation.atol}")
    engine = ValidatorEngine(config)
    results = engine.run_all(operators=args.operators, shapes=args.shapes)
    summary = engine.get_summary()
    print("\n" + "=" * 60)
    print("📊 验证总结")
    print("=" * 60)
    print(f"  总测试用例: {summary['total']}")
    print(f"  ✅ PASS:    {summary['passed']}")
    print(f"  ❌ FAIL:    {summary['failed']}")
    print(f"  ⚠️ WARNING: {summary['warning']}")
    print()
    for op_name, info in summary["by_operator"].items():
        icon = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️"}.get(info["overall"], "❓")
        print(f"  {icon} {op_name}: {info['overall']} (P={info['pass']}, F={info['fail']}, W={info['warning']})")
    report_config = {"config_path": str(config_path), "reference": config.backends[0].name if config.backends else "N/A", "rtol": config.validation.rtol, "atol": config.validation.atol, "operators_tested": args.operators, "shapes_tested": args.shapes}
    if args.report_json:
        reporter = ReportGenerator(summary, report_config)
        reporter.to_json(args.report_json)
    if args.report_md:
        reporter = ReportGenerator(summary, report_config)
        reporter.to_markdown(args.report_md)
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
