from functools import wraps
import os
import shutil
from tempfile import mkdtemp


def work_in_tmp_dir(kept_file_exts=None):
    """Execute a function in a temporary directory"""

    def func_decorator(func):

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            here = os.getcwd()

            tmpdir_path = mkdtemp()

            for item in os.listdir(os.getcwd()):
                if os.path.isdir(item):
                    continue

                if item.startswith('tmp'):
                    continue

                try:
                    shutil.copy(item, tmpdir_path)
                except FileNotFoundError:
                    pass

            # Move directories and execute
            os.chdir(tmpdir_path)
            out = func(*args, **kwargs)

            if kept_file_exts is not None:

                # Copy all the files back that have one of the file extensions
                # in the list kept_file_exts
                for filename in os.listdir(os.getcwd()):
                    if any(filename.endswith(ext) for ext in kept_file_exts):
                        shutil.copy(src=filename,
                                    dst=os.path.join(here, filename))

            os.chdir(here)

            # Remove the temporary dir with all files and return the output
            shutil.rmtree(tmpdir_path)

            return out

        return wrapped_function
    return func_decorator
