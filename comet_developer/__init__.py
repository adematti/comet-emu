"""Init module."""

import os
from comet.PTEmu import PTEmu as comet
base_dir = os.path.join(os.path.dirname(__file__))


def download_data(download_dir):
    """
    Download the data needed for the emulators to the specified directory.

    Parameters
    ----------
    download_dir : str
        the data will be downloaded to this directory
    """
    from six.moves import urllib
    import shutil
    import gzip
    import glob

    urls = ['https://www.ice.csic.es/owncloud/s/six47i7cceNkkp9/download',
            'https://www.ice.csic.es/owncloud/s/6EYSaxPxRZ57iEc/download']
    filenames = ['tables.zip', 'models.zip']
    out_filenames = ['tables', 'models']

    # download both files
    for i, url in enumerate(urls):

        # the download path
        # filename = url.split('/')[-1]
        file_path = os.path.join(download_dir+"/data_dir", filenames[i])
        final_path = os.path.join(download_dir+"/data_dir", out_filenames[i])

        # do not re-download

        if not os.path.exists(final_path):
            if i == 0:
                print("\n As it is the first instance of the emulator, "
                      "we need to download some data, it can take a few "
                      "seconds...\n")

            print("Downloading %s...\n" % out_filenames[i])

            file_path, _ = urllib.request.urlretrieve(url=url,
                                                      filename=file_path,
                                                      reporthook=None)

            print("Download finished. Extracting files.")

            # unzip the file
            shutil.unpack_archive(
                filename=file_path, extract_dir=download_dir+"/data_dir")
            os.remove(file_path)
            print("Done.\n")
        else:
            continue


download_data(base_dir)

if __name__ == '__main__':
    comet = comet()
