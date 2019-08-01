"""
Speech-related data analysis tools.
"""
import os

def _get_unittest_data_dir():
    from speechbox import __path__
    speechbox_root = os.path.dirname(__path__[0])
    return os.path.join(speechbox_root, "test", "data_common_voice")
