from gaptrain.utils import work_in_tmp_dir
import os


def test_tmp_dir():

    @work_in_tmp_dir(kept_exts=['txt'])
    def func(filename):
        open(filename, 'w').close()
        return os.path.exists(filename)

    out_exists = func(filename='tmp')
    assert out_exists

    # Working it a temporary directory should mean that file is not copied back
    assert not os.path.exists('tmp')

    # If the file ends with something in the kept extensions then copy it
    func(filename='tmp.txt')
    assert os.path.exists('tmp.txt')
    os.remove('tmp.txt')

