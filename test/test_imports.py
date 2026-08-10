import importlib
import unittest


class ImportTests(unittest.TestCase):
    def test_package_api_exports(self):
        pkg = importlib.import_module("impjax_toymodels")
        self.assertTrue(hasattr(pkg, "InterfaceSpec"))
        self.assertTrue(hasattr(pkg, "SamplingVariable"))
        self.assertTrue(hasattr(pkg, "start_timing"))
        self.assertTrue(hasattr(pkg, "elapsed_timing"))
        self.assertTrue(hasattr(pkg, "__version__"))


if __name__ == "__main__":
    unittest.main()
