import unittest
import sys
import os

def run_project_pipeline():
    print("🚀 Starting Computer Vision Pipeline...")

    test_dir = './tests'

    if os.path.exists(test_dir):
        print("🔍 Running tests...")
        loader = unittest.TestLoader()
        suite = loader.discover(test_dir, pattern='test_*.py')

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if not result.wasSuccessful():
            print("\n❌ Pipeline Failed: Tests failed.")
            sys.exit(1)

    else:
        print("⚠️ No tests found, skipping testing phase.")

    print("\n✅ Pipeline Passed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    run_project_pipeline()