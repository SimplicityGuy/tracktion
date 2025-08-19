#!/usr/bin/env python3
"""
Test runner for BPM detection integration tests.

Runs the complete test suite and generates a report.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_tests():
    """Run integration tests and generate report."""
    print("🎵 Running BPM Detection Integration Tests")
    print("=" * 50)

    # Change to project root
    project_root = Path(__file__).parent.parent

    # Ensure test audio files exist
    test_audio_dir = project_root / "tests" / "fixtures"
    audio_files = list(test_audio_dir.glob("test_*.wav"))

    print(f"📁 Test audio files found: {len(audio_files)}")
    for audio_file in audio_files:
        print(f"   - {audio_file.name}")

    if not audio_files:
        print("⚠️  No test audio files found. Generating them now...")
        try:
            subprocess.run([sys.executable, "tests/fixtures/generate_test_audio.py"], cwd=project_root, check=True)
            print("✅ Test audio files generated successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate test audio files: {e}")
            return False

    # Run integration tests
    print("\n🧪 Running integration tests...")
    start_time = time.time()

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/integration/test_bpm_integration.py",
                "-v",  # Verbose output
                "--tb=short",  # Short traceback format
                "--durations=10",  # Show 10 slowest tests
                "--color=yes",  # Colored output
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        duration = time.time() - start_time

        print(f"\n⏱️  Test duration: {duration:.2f} seconds")
        print(f"📊 Exit code: {result.returncode}")

        # Print test output
        if result.stdout:
            print("\n📋 Test Output:")
            print(result.stdout)

        if result.stderr:
            print("\n⚠️  Test Errors:")
            print(result.stderr)

        if result.returncode == 0:
            print("\n✅ All integration tests passed!")
            return True
        else:
            print(f"\n❌ Tests failed with exit code {result.returncode}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run tests: {e}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Please install test dependencies:")
        print("   uv add pytest pytest-asyncio")
        return False


def run_unit_tests():
    """Run unit tests for comparison."""
    print("\n🔬 Running Unit Tests for Context...")
    project_root = Path(__file__).parent.parent

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/unit/analysis_service/", "-v", "--tb=short", "--color=yes"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ Unit tests passed")
        else:
            print(f"⚠️  Some unit tests failed (exit code: {result.returncode})")

        return result.returncode == 0

    except Exception as e:
        print(f"⚠️  Could not run unit tests: {e}")
        return False


def generate_test_report():
    """Generate a test coverage report."""
    print("\n📊 Generating Test Report...")
    project_root = Path(__file__).parent.parent

    try:
        # Run tests with coverage
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "--cov=services.analysis_service.src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "-q",  # Quiet mode for coverage report
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        if "TOTAL" in result.stdout:
            print("✅ Coverage report generated")
            print(result.stdout.split("\n")[-3:-1])  # Show coverage summary
        else:
            print("⚠️  Coverage report generation failed")

    except Exception as e:
        print(f"⚠️  Could not generate coverage report: {e}")


def main():
    """Main test runner function."""
    print("🚀 BPM Detection Test Suite")
    print("=" * 30)

    # Track results
    results = []

    # Run integration tests
    integration_passed = run_tests()
    results.append(("Integration Tests", integration_passed))

    # Run unit tests for context
    unit_passed = run_unit_tests()
    results.append(("Unit Tests", unit_passed))

    # Generate coverage report if possible
    try:
        generate_test_report()
    except Exception as e:
        print(f"⚠️  Coverage report skipped: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)

    for test_type, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_type:20} {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests completed successfully!")
        print("🎵 BPM detection pipeline is ready for deployment")
        exit_code = 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        exit_code = 1

    print("\n💡 Next steps:")
    print("   - Review test coverage report in htmlcov/index.html")
    print("   - Add more test cases for edge cases if needed")
    print("   - Consider performance benchmarking")

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
