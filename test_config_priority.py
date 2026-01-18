#!/usr/bin/env python3
"""Test script to verify STT_INPUT_ADDRESS configuration priority."""
import os
from pathlib import Path
from src.stt.config import STTConfig


def test_default():
    """Test default value (no env var, no override)."""
    # Clear env var if it exists
    os.environ.pop("STT_INPUT_ADDRESS", None)
    
    config = STTConfig()
    assert config.input_address == "tcp://localhost:20499", \
        f"Expected 'tcp://localhost:20499', got '{config.input_address}'"
    print("✓ Test 1 passed: Default value works")


def test_env_variable():
    """Test environment variable (env var set, no override)."""
    os.environ["STT_INPUT_ADDRESS"] = "tcp://*:5555"
    
    config = STTConfig()
    assert config.input_address == "tcp://*:5555", \
        f"Expected 'tcp://*:5555', got '{config.input_address}'"
    print("✓ Test 2 passed: Environment variable works")


def test_override():
    """Test CLI override (env var set, but CLI overrides)."""
    os.environ["STT_INPUT_ADDRESS"] = "tcp://*:5555"
    
    config = STTConfig(input_address="tcp://*:9999")
    assert config.input_address == "tcp://*:9999", \
        f"Expected 'tcp://*:9999', got '{config.input_address}'"
    print("✓ Test 3 passed: CLI override works")


def test_priority_order():
    """Test complete priority: override > env variable > default."""
    # Test 1: default
    os.environ.pop("STT_INPUT_ADDRESS", None)
    config = STTConfig()
    assert config.input_address == "tcp://localhost:20499"
    
    # Test 2: env variable overrides default
    os.environ["STT_INPUT_ADDRESS"] = "tcp://*:7777"
    config = STTConfig()
    assert config.input_address == "tcp://*:7777"
    
    # Test 3: CLI override beats env variable
    config = STTConfig(input_address="tcp://*:8888")
    assert config.input_address == "tcp://*:8888"
    
    print("✓ Test 4 passed: Priority order is correct")


def main():
    print("Testing STT_INPUT_ADDRESS configuration priority...")
    print("=" * 60)
    
    try:
        test_default()
        test_env_variable()
        test_override()
        test_priority_order()
        
        print("=" * 60)
        print("✅ All tests passed!")
        print("\nPriority order confirmed:")
        print("  1. CLI flag (--input-address)")
        print("  2. Environment variable (STT_INPUT_ADDRESS)")
        print("  3. Default (tcp://localhost:20499)")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up
        os.environ.pop("STT_INPUT_ADDRESS", None)
    
    return 0


if __name__ == "__main__":
    exit(main())
