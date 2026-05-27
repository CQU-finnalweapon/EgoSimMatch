"""EgoSimMatch 流水线示例脚本。"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="EgoSimMatch Pipeline")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()

    print(f"EgoSimMatch pipeline starting... (config: {args.config})")
    # TODO: 实现完整流水线


if __name__ == "__main__":
    main()
