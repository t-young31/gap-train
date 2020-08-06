from functools import wraps
import os
import shutil
from tempfile import mkdtemp


def work_in_tmp_dir():
    """Execute a function in a temporary directory"""

    def func_decorator(func):

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            here = os.getcwd()

            tmpdir_path = mkdtemp()

            for filename in os.listdir(os.getcwd()):
                shutil.copy(filename, tmpdir_path)

            # Move directories and execute
            os.chdir(tmpdir_path)
            func(*args, **kwargs)
            os.chdir(here)

            # Remove the temporary dir with all files
            shutil.rmtree(tmpdir_path)

        return wrapped_function
    return func_decorator
