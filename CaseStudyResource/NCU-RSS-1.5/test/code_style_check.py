import tensorflow as tf
import pycodestyle
import glob
import os


class TestCodeFormat(tf.test.TestCase):

    def test_conformance(self):
        """Test that we conform to PEP-8."""
        current_repo_path = os.getcwd()  # path should be sth like this D:\\1111_work\\NCU-RSS-1.5

        cfg_path = os.path.join(current_repo_path, 'test', 'setup.cfg')
        style = pycodestyle.StyleGuide(config_file=cfg_path)
        filenames = glob.glob(os.path.join(current_repo_path, 'src') + '/**/*.py', recursive=True)
        result = style.check_files(filenames)

        self.assertLess(result.total_errors, 250, msg="Found code style errors (and warnings).")
