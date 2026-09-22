import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import runtime


class RuntimeTests(unittest.TestCase):
    def test_changed_module_refreshes_without_reloading_unchanged_code(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'research_desk_refresh_fixture.py'
            path.write_text('version = "old"\n')
            sys.path.insert(0, folder)
            self.addCleanup(sys.path.remove, folder)
            self.addCleanup(sys.modules.pop, 'research_desk_refresh_fixture', None)
            self.addCleanup(runtime._loaded.pop, 'research_desk_refresh_fixture', None)
            module = importlib.import_module('research_desk_refresh_fixture')
            revision = runtime.refresh_answer_modules(module)
            with patch('runtime.importlib.reload', wraps=importlib.reload) as reload:
                self.assertEqual(runtime.refresh_answer_modules(module), revision)
                reload.assert_not_called()
                path.write_text('version = "updated answer behavior"\n')
                os.utime(path, None)
                self.assertNotEqual(runtime.refresh_answer_modules(module), revision)
                self.assertEqual(module.version, 'updated answer behavior')
                reload.assert_called_once_with(module)


if __name__ == '__main__':
    unittest.main()
