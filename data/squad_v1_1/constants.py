import os

import utils.io as io


class SquadConstants(io.JsonSerializableClass):
    def __init__(
            self,
            raw_dir=os.path.join(
                os.getcwd(),
                'symlinks/data/squad_v1_1')):
        self.raw_dir = raw_dir
        self.train_json = os.path.join(self.raw_dir,'squad-train-v1.1.json')
        self.dev_json = os.path.join(self.raw_dir,'squad-dev-v1.1.json')