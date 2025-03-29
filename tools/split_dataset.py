import os
import random
import shutil
import argparse
from glob import glob
from tqdm import tqdm
import threading
import queue

WORKERS = 8

def process_file(args):
    """
    Process a single file (move or copy)
    
    Args:
        args (tuple): (file_path, dest_path, copy_flag)
    
    Returns:
        int: 0 for success, 1 for failure
    """
    file_path, dest_path, copy = args
    try:
        if copy:
            shutil.copy2(file_path, dest_path)
        else:
            shutil.move(file_path, dest_path)
        return 0
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return 1

class ThreadedFileProcessor:
    def __init__(self, num_threads):
        self.task_queue = queue.Queue()
        self.results = []
        self.lock = threading.Lock()
        self.completed = 0
        self.total = 0
        self.pbar = None

    def worker(self):
        while True:
            task = self.task_queue.get()
            if task is None:  # Termination signal
                self.task_queue.task_done()
                break
                
            result = process_file(task)
            
            with self.lock:
                self.results.append(result)
                self.completed += 1
                if self.pbar:
                    self.pbar.update(1)
            
            self.task_queue.task_done()

    def process_tasks(self, tasks, num_threads):
        self.total = len(tasks)
        self.pbar = tqdm(total=self.total)
        
        # Start worker threads
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=self.worker)
            t.daemon = True
            t.start()
            threads.append(t)
        
        # Add tasks to the queue
        for task in tasks:
            self.task_queue.put(task)
        
        # Add termination signals
        for _ in range(num_threads):
            self.task_queue.put(None)
        
        # Wait for all tasks to complete
        self.task_queue.join()
        
        self.pbar.close()
        return self.results

def split_dataset(source_dir, dest_dir, train_ratio=0.9, valid_ratio=0.05, test_ratio=0.05, 
                  copy=False, file_types=('*.mid', '*.midi'), workers=WORKERS):
    """
    Split files from source directory into train/valid/test subdirectories
    
    Args:
        source_dir (str): Source directory
        dest_dir (str): Destination directory
        train_ratio (float): Training set ratio, default 0.9
        valid_ratio (float): Validation set ratio, default 0.05
        test_ratio (float): Test set ratio, default 0.05
        copy (bool): True to copy files, False to move files
        file_types (tuple): File types to process
        workers (int): Number of parallel workers
    """
    # Ensure target directories exist
    train_dir = os.path.join(dest_dir, 'train')
    valid_dir = os.path.join(dest_dir, 'valid')
    test_dir = os.path.join(dest_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all files matching the specified types
    all_files = []
    for file_type in file_types:
        all_files.extend(glob(os.path.join(source_dir, '**', file_type), recursive=True))
    
    # Calculate split points
    total_files = len(all_files)
    train_end = int(total_files * train_ratio)
    valid_end = train_end + int(total_files * valid_ratio)
    
    # Split files without shuffling
    train_files = all_files[:train_end]
    valid_files = all_files[train_end:valid_end]
    test_files = all_files[valid_end:]
    
    print(f"Total files: {total_files}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(valid_files)}")
    print(f"Test files: {len(test_files)}")
    
    # Prepare tasks for sequential processing
    tasks = []
    
    # Process training files
    print("Preparing training files...")
    for file_path in train_files:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(train_dir, file_name)
        
        # If target file exists, add suffix to avoid duplicate names
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(file_name)
            file_name = f"{base}_{random.randint(1000, 9999)}{ext}"
            dest_path = os.path.join(train_dir, file_name)
        
        tasks.append((file_path, dest_path, copy))
    
    # Process validation files
    print("Preparing validation files...")
    for file_path in valid_files:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(valid_dir, file_name)
        
        # If target file exists, add suffix to avoid duplicate names
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(file_name)
            file_name = f"{base}_{random.randint(1000, 9999)}{ext}"
            dest_path = os.path.join(valid_dir, file_name)
        
        tasks.append((file_path, dest_path, copy))
    
    # Process test files
    print("Preparing test files...")
    for file_path in test_files:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(test_dir, file_name)
        
        # If target file exists, add suffix to avoid duplicate names
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(file_name)
            file_name = f"{base}_{random.randint(1000, 9999)}{ext}"
            dest_path = os.path.join(test_dir, file_name)
        
        tasks.append((file_path, dest_path, copy))
    
    # Process files using multiple threads
    print(f"Processing {len(tasks)} files with {workers} threads...")
    processor = ThreadedFileProcessor(workers)
    results = processor.process_tasks(tasks, workers)
    
    failures = sum(results)
    success_rate = (len(tasks) - failures) / len(tasks) * 100
    print(f"Processed {len(tasks) - failures} files successfully ({success_rate:.2f}%)")
    if failures > 0:
        print(f"Failed to process {failures} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split MIDI files into train/valid/test subdirectories")
    parser.add_argument("source_dir", help="Source directory containing MIDI files")
    parser.add_argument("--dest_dir", default=None, help="Destination directory, defaults to source directory")
    parser.add_argument("--train", type=float, default=0.9, help="Training set ratio, default 0.9")
    parser.add_argument("--valid", type=float, default=0.05, help="Validation set ratio, default 0.05")
    parser.add_argument("--test", type=float, default=0.05, help="Test set ratio, default 0.05")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them")
    parser.add_argument("--workers", type=int, default=WORKERS, help=f"Number of parallel workers, default {WORKERS}")
    
    args = parser.parse_args()
    
    # Validate that ratios sum to 1.0
    total_ratio = args.train + args.valid + args.test
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Warning: The sum of ratios ({total_ratio}) is not equal to 1.0")
        print("Normalizing ratios...")
        args.train /= total_ratio
        args.valid /= total_ratio  
        args.test /= total_ratio
    
    dest_dir = args.dest_dir if args.dest_dir else args.source_dir
    split_dataset(args.source_dir, dest_dir, args.train, args.valid, args.test, args.copy, workers=args.workers)
    
    print("Dataset splitting completed!")
