import os

import utils.io as io


class GoogleImagesConstants(io.JsonSerializableClass):
    def __init__(
            self,
            raw_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/google_images')):
        self.raw_dir = raw_dir