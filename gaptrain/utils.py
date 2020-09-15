from functools import wraps
import os
import shutil
from tempfile import mkdtemp


def work_in_tmp_dir(kept_exts=None, copied_exts=None):
    """Execute a function in a temporary directory"""

    def func_decorator(func):

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            here = os.getcwd()

            tmpdir_path = mkdtemp()
            # logger.info(f'Working in {tmpdir_path}')

            for item in os.listdir(os.getcwd()):

                if copied_exts is None:
                    continue

                if any(item.endswith(ext) for ext in copied_exts):
                    # logger.info(f'Copying {item}')
                    shutil.copy(item, tmpdir_path)

            # Move directories and execute
            os.chdir(tmpdir_path)
            out = func(*args, **kwargs)

            if kept_exts is not None:
                # logger.info(f'Copying back all files with {kept_exts} '
                #            f'file extensions')

                for filename in os.listdir(os.getcwd()):
                    if any(filename.endswith(ext) for ext in kept_exts):
                        shutil.copy(src=filename,
                                    dst=os.path.join(here, filename))

            os.chdir(here)

            # Remove the temporary dir with all files and return the output
            shutil.rmtree(tmpdir_path)

            return out

        return wrapped_function
    return func_decorator
