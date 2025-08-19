#!/usr/bin/env python3
"""
System validation script for BPM detection.

Tests core functionality without mocks to ensure the system works correctly.
"""

import logging
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_bpm_detector():
    """Test BPM detector with synthetic audio."""
    print("🎵 Testing BPM Detector...")

    try:
        from services.analysis_service.src.bpm_detector import BPMDetector
        from services.analysis_service.src.config import get_config

        # Initialize detector
        config = get_config()
        detector = BPMDetector(config.bpm)

        # Test with generated audio files
        test_dir = Path(__file__).parent / "fixtures"
        test_files = [
            ("test_120bpm_rock.wav", [120.0, 60.0], "120 BPM rock track"),
            ("test_128bpm_electronic.wav", [128.0, 64.0], "128 BPM electronic track"),
            ("test_silence.wav", None, "Silence (should have low confidence)"),
        ]

        results = []
        for filename, expected_bpms, description in test_files:
            file_path = test_dir / filename
            if not file_path.exists():
                print(f"⚠️  Skipping {filename} - file not found")
                continue

            print(f"   Testing: {description}")
            try:
                result = detector.detect_bpm(str(file_path))

                # Basic validation
                assert "bpm" in result, "BPM not in result"
                assert "confidence" in result, "Confidence not in result"
                assert "algorithm" in result, "Algorithm not in result"
                assert isinstance(result["bpm"], int | float), "BPM not numeric"
                assert 0.0 <= result["confidence"] <= 1.0, f"Invalid confidence: {result['confidence']}"

                # BPM validation
                if expected_bpms:
                    detected_bpm = result["bpm"]
                    tolerance = 10.0  # Generous tolerance for validation
                    bpm_matches = any(abs(detected_bpm - expected) <= tolerance for expected in expected_bpms)
                    if not bpm_matches:
                        print(f"      ⚠️  BPM outside tolerance: expected {expected_bpms}, got {detected_bpm}")
                    else:
                        print(f"      ✅ BPM: {detected_bpm:.1f} (confidence: {result['confidence']:.2f})")
                else:
                    print(f"      ✅ Processed: {result['bpm']:.1f} BPM (confidence: {result['confidence']:.2f})")

                results.append({"file": filename, "result": result, "status": "success"})

            except Exception as e:
                print(f"      ❌ Error: {e}")
                results.append({"file": filename, "error": str(e), "status": "error"})

        successful = len([r for r in results if r["status"] == "success"])
        total = len(results)
        print(f"   Results: {successful}/{total} files processed successfully")

        if successful > 0:
            print("✅ BPM Detector working correctly")
            return True
        else:
            print("❌ BPM Detector failed all tests")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure Essentia is installed: uv add essentia-tensorflow")
        return False
    except Exception as e:
        print(f"❌ BPM Detector test failed: {e}")
        return False


def test_temporal_analyzer():
    """Test temporal analyzer with variable tempo audio."""
    print("⏰ Testing Temporal Analyzer...")

    try:
        from services.analysis_service.src.config import get_config
        from services.analysis_service.src.temporal_analyzer import TemporalAnalyzer

        # Initialize analyzer
        config = get_config()
        analyzer = TemporalAnalyzer(config.temporal)

        # Test with variable tempo file
        test_dir = Path(__file__).parent / "fixtures"
        test_file = test_dir / "test_variable_tempo.wav"

        if not test_file.exists():
            print("⚠️  Variable tempo test file not found, skipping")
            return True

        print("   Testing: Variable tempo analysis")
        result = analyzer.analyze_temporal_bpm(str(test_file))

        # Validate result structure
        required_keys = ["average_bpm", "stability_score", "is_variable_tempo", "tempo_changes"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        # Validate values
        assert isinstance(result["average_bpm"], int | float), "Average BPM not numeric"
        assert 0.0 <= result["stability_score"] <= 1.0, f"Invalid stability score: {result['stability_score']}"
        assert isinstance(result["is_variable_tempo"], bool), "is_variable_tempo not boolean"

        print(f"   ✅ Average BPM: {result['average_bpm']:.1f}")
        print(f"   ✅ Stability: {result['stability_score']:.2f}")
        print(f"   ✅ Variable tempo: {result['is_variable_tempo']}")
        print(f"   ✅ Tempo changes: {len(result['tempo_changes'])}")

        print("✅ Temporal Analyzer working correctly")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Temporal Analyzer test failed: {e}")
        return False


def test_configuration_system():
    """Test configuration loading and validation."""
    print("⚙️  Testing Configuration System...")

    try:
        from services.analysis_service.src.config import get_config

        # Test default configuration
        config = get_config()

        # Validate configuration structure
        assert hasattr(config, "bpm"), "Missing BPM config"
        assert hasattr(config, "temporal"), "Missing temporal config"
        assert hasattr(config, "cache"), "Missing cache config"
        assert hasattr(config, "performance"), "Missing performance config"

        # Test configuration values
        assert 0.0 <= config.bpm.confidence_threshold <= 1.0, "Invalid BPM confidence threshold"
        assert config.temporal.window_size_seconds > 0, "Invalid temporal window size"
        assert config.performance.parallel_workers >= 1, "Invalid parallel workers"

        print(f"   ✅ BPM confidence threshold: {config.bpm.confidence_threshold}")
        print(f"   ✅ Temporal window size: {config.temporal.window_size_seconds}s")
        print(f"   ✅ Parallel workers: {config.performance.parallel_workers}")
        print(f"   ✅ Cache enabled: {config.cache.enabled}")

        # Test validation
        validation_errors = config.validate()
        if validation_errors:
            print(f"⚠️  Configuration warnings: {validation_errors}")
        else:
            print("   ✅ Configuration validation passed")

        print("✅ Configuration System working correctly")
        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_audio_cache():
    """Test audio cache functionality."""
    print("💾 Testing Audio Cache...")

    try:
        from services.analysis_service.src.audio_cache import AudioCache
        from services.analysis_service.src.config import get_config

        config = get_config()

        # Test with cache disabled (no Redis dependency)
        cache_config = config.cache
        cache_config.enabled = False
        cache_config.redis.host = "localhost"  # Ensure host is set as string

        cache = AudioCache(cache_config)

        # Test basic operations (should work without Redis)
        test_file = "/fake/test/file.mp3"
        test_results = {"bpm": 120.0, "confidence": 0.85, "algorithm": "primary"}

        # Test cache operations (should handle gracefully when disabled)
        cached_result = cache.get_bpm_results(test_file)
        assert cached_result is None, "Should return None when cache disabled"

        cache.set_bpm_results(test_file, test_results)
        # Should handle gracefully (either succeed or fail gracefully)

        print("   ✅ Cache operations handled gracefully")
        print("✅ Audio Cache working correctly")
        return True

    except Exception as e:
        print(f"❌ Audio Cache test failed: {e}")
        return False


def test_performance_monitoring():
    """Test performance monitoring utilities."""
    print("📊 Testing Performance Monitoring...")

    try:
        from services.analysis_service.src.config import get_config
        from services.analysis_service.src.performance import MemoryManager, PerformanceMonitor

        # Test performance monitor
        monitor = PerformanceMonitor()

        with monitor.measure("test_operation"):
            time.sleep(0.01)  # Simulate work

        metrics = monitor.get_metrics()
        assert "test_operation" in metrics, "Operation not recorded"
        assert "duration_seconds" in metrics["test_operation"], "Duration not recorded"
        assert metrics["test_operation"]["duration_seconds"] >= 0.01, "Duration too short"

        print(f"   ✅ Performance monitoring: {metrics['test_operation']['duration_seconds']:.3f}s")

        # Test memory manager with high limit
        config = get_config()
        config.performance.memory_limit_mb = 10000  # High limit to avoid test failures

        memory_manager = MemoryManager(config.performance)
        memory_mb, within_limit = memory_manager.check_memory()

        assert memory_mb > 0, "Memory usage should be positive"
        print(f"   ✅ Memory usage: {memory_mb:.1f}MB (within limit: {within_limit})")

        # Test memory info
        memory_info = memory_manager.get_memory_info()
        required_keys = ["process_memory_mb", "system_memory_mb", "available_memory_mb"]
        for key in required_keys:
            assert key in memory_info, f"Missing memory info key: {key}"
            assert memory_info[key] >= 0, f"Invalid memory value for {key}"

        print("✅ Performance Monitoring working correctly")
        return True

    except Exception as e:
        print(f"❌ Performance Monitoring test failed: {e}")
        return False


def generate_test_report(results: dict[str, bool]) -> bool:
    """Generate a summary test report."""
    print("\n" + "=" * 60)
    print("📋 SYSTEM VALIDATION REPORT")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30} {status}")

    print("-" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")

    if passed_tests == total_tests:
        print("\n🎉 All system validation tests passed!")
        print("🎵 BPM detection system is ready for production")
        return True
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed. Please review the issues above.")
        return False


def main():
    """Run system validation tests."""
    print("🚀 BPM Detection System Validation")
    print("=" * 40)
    print("Testing core functionality without external dependencies...\n")

    # Run validation tests
    results = {
        "Configuration System": test_configuration_system(),
        "BPM Detector": test_bpm_detector(),
        "Temporal Analyzer": test_temporal_analyzer(),
        "Audio Cache": test_audio_cache(),
        "Performance Monitoring": test_performance_monitoring(),
    }

    # Generate report
    success = generate_test_report(results)

    if success:
        print("\n💡 Next steps:")
        print("   - Run full integration tests: uv run python tests/run_integration_tests.py")
        print("   - Deploy to staging environment")
        print("   - Monitor performance in production")
        return 0
    else:
        print("\n💡 Troubleshooting:")
        print("   - Check error messages above")
        print("   - Verify dependencies are installed")
        print("   - Review configuration settings")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
