#!/usr/bin/env python
"""
Setup script to configure Scientific Papers MCP Server with Claude Code

Run this script to automatically:
1. Check Python version and dependencies
2. Index documents
3. Configure Claude Code MCP settings
"""

import sys
import json
import os
from pathlib import Path
import subprocess

def check_python_version():
    """Check Python version (need 3.10+)"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"[FAIL] Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    required = [
        "fastmcp",
        "chromadb",
        "sentence_transformers",
        "rank_bm25",
        "langchain",
    ]

    try:
        import pkg_resources
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = [pkg for pkg in required if pkg.lower() not in installed]

        if missing:
            print(f"[FAIL] Missing packages: {missing}")
            print("Install with: pip install -e .")
            return False

        print(f"[OK] All dependencies installed")
        return True
    except Exception as e:
        print(f"[WARN] Could not check dependencies: {e}")
        return True


def check_documents():
    """Check if documents directory exists"""
    print("\nChecking documents directory...")
    from src.config import config

    if not config.documents_path.exists():
        print(f"[FAIL] Documents path not found: {config.documents_path}")
        return False

    md_files = list(config.documents_path.glob("*.md"))
    print(f"[OK] Found {len(md_files)} markdown files")
    return True


def index_documents():
    """Index all documents"""
    print("\nIndexing documents...")
    try:
        from src.server import index_all_documents
        from src.config import config

        result = index_all_documents(config.documents_path)

        if result["status"] == "success":
            print(f"[OK] Indexed {result['indexed_files']} files")
            return True
        else:
            print(f"[FAIL] {result['message']}")
            return False
    except Exception as e:
        print(f"[FAIL] Indexing error: {e}")
        return False


def setup_claude_code():
    """Setup Claude Code MCP configuration"""
    print("\nSetting up Claude Code configuration...")

    # Determine Claude Code config path
    if sys.platform == "win32":
        claude_config_path = Path.home() / "AppData/Local/Claude Code/.claude/claude.json"
    elif sys.platform == "darwin":
        claude_config_path = Path.home() / "Library/Application Support/Claude Code/.claude/claude.json"
    else:  # Linux
        claude_config_path = Path.home() / ".config/Claude Code/.claude/claude.json"

    # Check if config exists
    if not claude_config_path.exists():
        print(f"[INFO] Claude Code config not found at {claude_config_path}")
        print("Please refer to SETUP_CLAUDE_CODE.md for manual configuration")
        return False

    try:
        # Read existing config
        with open(claude_config_path, "r") as f:
            config_data = json.load(f)

        # Add our MCP server
        if "mcpServers" not in config_data:
            config_data["mcpServers"] = {}

        project_path = Path(__file__).parent
        config_data["mcpServers"]["scientific-papers"] = {
            "command": "python",
            "args": ["-m", "src.server"],
            "cwd": str(project_path),
            "env": {
                "PYTHONPATH": str(project_path)
            }
        }

        # Write back
        with open(claude_config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        print(f"[OK] Updated {claude_config_path}")
        print("[INFO] Restart Claude Code for changes to take effect")
        return True

    except Exception as e:
        print(f"[WARN] Could not update Claude Code config: {e}")
        print("Please refer to SETUP_CLAUDE_CODE.md for manual configuration")
        return False


def main():
    """Run setup"""
    print("=" * 60)
    print("Scientific Papers MCP Server - Setup")
    print("=" * 60)

    steps = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Documents", check_documents),
        ("Indexing", index_documents),
        ("Claude Code Config", setup_claude_code),
    ]

    results = {}
    for name, func in steps:
        try:
            results[name] = func()
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Setup Summary:")
    print("=" * 60)

    all_ok = True
    for name, passed in results.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")
        if not passed:
            all_ok = False

    print("=" * 60)

    if all_ok:
        print("\n✓ Setup completed successfully!")
        print("\nYour MCP server is ready to use:")
        print("1. Restart Claude Code")
        print("2. Try searching: 'search: glacier albedo'")
        print("\nSee README.md for more information")
    else:
        print("\n✗ Setup completed with errors.")
        print("See SETUP_CLAUDE_CODE.md for troubleshooting")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
