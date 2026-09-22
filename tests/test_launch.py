from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch, call
import launch


class LauncherTests(unittest.TestCase):
    def context(self, stack):
        folder = stack.enter_context(tempfile.TemporaryDirectory())
        stack.enter_context(patch('launch.DATA', Path(folder)))
        stack.enter_context(patch('launch.os.chdir'))
        stack.enter_context(patch('launch.webbrowser.open'))
        stack.enter_context(patch('launch.shutil.which', return_value='/bin/sh'))
        stack.enter_context(patch('launch.time.sleep', side_effect=KeyboardInterrupt))

    def test_new_server_stays_in_terminal_and_ctrl_c_cleans_up_children(self):
        with ExitStack() as stack:
            self.context(stack)
            stack.enter_context(patch('launch.listeners', return_value=set()))
            stack.enter_context(patch('launch.available', return_value=False))
            stack.enter_context(patch('launch.wait_until_ready'))
            ollama, app = Mock(), Mock()
            ollama.poll.return_value = app.poll.return_value = None
            popen = stack.enter_context(patch('launch.subprocess.Popen', side_effect=[ollama, app]))
            self.assertEqual(launch.main(), 0)
            self.assertNotIn('start_new_session', popen.call_args_list[1].kwargs)
            self.assertNotIn('stdout', popen.call_args_list[1].kwargs)
            app.terminate.assert_called_once()
            ollama.terminate.assert_called_once()

    def test_existing_project_servers_are_controlled_without_restarting(self):
        with ExitStack() as stack:
            self.context(stack)
            stack.enter_context(patch('launch.listeners', side_effect=lambda port: {100 if port == 8501 else 200}))
            stack.enter_context(patch('launch.belongs_to_project', return_value=True))
            stack.enter_context(patch('launch.available', return_value=True))
            popen = stack.enter_context(patch('launch.subprocess.Popen'))
            stop = stack.enter_context(patch('launch.stop_service'))
            self.assertEqual(launch.main(), 0)
            popen.assert_not_called()
            self.assertEqual(stop.call_args_list, [call('Research Desk', 8501), call('Ollama', 11434)])

    def test_other_app_on_port_is_not_stopped(self):
        with ExitStack() as stack:
            self.context(stack)
            stack.enter_context(patch('launch.listeners', return_value={100}))
            stack.enter_context(patch('launch.belongs_to_project', return_value=False))
            stop = stack.enter_context(patch('launch.stop_service'))
            self.assertEqual(launch.main(), 1)
            stop.assert_not_called()

    def test_startup_failure_cleans_up_launched_processes(self):
        with ExitStack() as stack:
            self.context(stack)
            stack.enter_context(patch('launch.listeners', return_value=set()))
            stack.enter_context(patch('launch.available', return_value=False))
            stack.enter_context(patch('launch.wait_until_ready', side_effect=[None, RuntimeError('failed')]))
            ollama, app = Mock(), Mock()
            ollama.poll.return_value = app.poll.return_value = None
            stack.enter_context(patch('launch.subprocess.Popen', side_effect=[ollama, app]))
            self.assertEqual(launch.main(), 1)
            app.terminate.assert_called_once()
            ollama.terminate.assert_called_once()
