from pathlib import Path
from datetime import datetime


class SimpleLogger:
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.save_dir / 'train.log'

    def log(self, msg):
        t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{t}] {msg}'
        print(line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
