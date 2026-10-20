import unittest
import sys

if __name__ == "__main__":
    print("Starting system test suite execution...")
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    skipped = result.skipped + result.expectedFailures
    if result.wasSuccessful() and not skipped:
        print("\nAll tests passed successfully!")
        sys.exit(0)
    else:
        if skipped:
            print("\nTest suite is incomplete: skipped/expected-failure tests are forbidden.")
            for test, reason in skipped:
                print(f" - {test}: {reason}")
        else:
            print("\nSome tests failed!")
        sys.exit(1)
