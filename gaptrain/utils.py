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


def unique_name(basename, exts=('xml', 'log', 'xyz')):
    """
    Return a unique filename based on not clashing with other files with the
    same basename plus extension. Append 0, 1... iteratively untill something
    unique is defined

    :param basename: (str)
    :param exts: (tuple(str))
    :return:
    """
    if any(ext.startswith('.') for ext in exts):
        raise ValueError('Extensions cannot have . prefixes')

    def any_exist():
        """Do any of the filenames with the possible extensions exist?"""
        return any(os.path.exists(f'{basename}.{ext}') for ext in exts)

    if not any_exist():
        return basename

    old_basename = basename
    i = 0
    while any_exist():
        basename = f'{old_basename}{i}'
        i += 1

    return basename
