import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from studio_lifecycle import request_shutdown, write_session, monitor_parent


class LifecycleTests(unittest.TestCase):
    def server(self):
        return Mock(server_port=43210, token='session_token',
                    stopped=threading.Event(),
                    service=Mock(job={'state': 'idle'}, cancel_event=threading.Event()))

    def test_handshake_is_atomic_and_describes_this_process(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as temp:
            path = Path(temp)/'session.json'
            server = self.server()
            write_session(server, path)
            value = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(value['url'], 'http://127.0.0.1:43210/')
            self.assertEqual(value['token'], 'session_token')
            self.assertGreater(value['pid'], 0)
            self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_shutdown_waits_until_current_image_is_saved(self):
        server = self.server()
        server.service.job = {'state': 'running'}
        thread = request_shutdown(server)
        self.assertTrue(server.service.cancel_event.wait(2))
        self.assertFalse(server.shutdown.called)
        server.service.job = {'state': 'cancelled'}
        thread.join(3)
        server.shutdown.assert_called_once()

    def test_parent_exit_stops_owned_service(self):
        server = self.server()
        monitor_parent(server, alive=lambda: False, interval=.01)
        server.shutdown.assert_called_once()

    def test_monitor_returns_when_service_has_already_stopped(self):
        server = self.server()
        server.stopped = threading.Event()
        server.stopped.set()
        monitor_parent(server, alive=lambda: True, interval=.01)
        server.shutdown.assert_not_called()


if __name__ == '__main__':
    unittest.main()
