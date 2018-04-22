from unittest import TestLoader, TextTestRunner

loader = TestLoader()
suite = loader.discover('test_cases', pattern='*_tests.py')

runner = TextTestRunner()
runner.run(suite)