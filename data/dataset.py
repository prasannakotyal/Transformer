"""
Data loader for Gutenberg corpus.

Handles character-level tokenization and batching.
"""

import os
import torch
from typing import Tuple


class GutenbergDataset:
    """
    Character-level dataset for Project Gutenberg books.
    
    Supports:
    - Character-level tokenization
    - Train/val/test splits
    - Batching with context windows
    """
    
    def __init__(
        self,
        data_path: str,
        context_length: int = 256,
        train_split: float = 0.9
        device: str = 'cuda'
    ):
        """
        Args:
            data_path: Path to text file or directory
            context_length: Maximum sequence length
            train_split: Fraction of data for training
            device: 'cuda' or 'cpu'
        """
        self.context_length = context_length
        self.device = device
        
        # Load text
        self.text = self._load_data(data_path)
        
        # Build vocabulary
        self.vocab = sorted(list(set(self.text)))
        self.vocab_size = len(self.vocab)
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # Tokenize
        self.data = torch.tensor(
            [self.char_to_idx[char] for char in self.text],
            dtype=torch.long
        )
        
        # Split
        split_idx = int(len(self.data) * train_split)
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]
        
        print(f"Loaded {len(self.text):, characters")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Train: {len(self.train_data):, Val: {len(self.val_data):}")
    
    def _load_data(self, data_path: str) -> str:
        """
        Load text from file or directory.
        """
        if os.path.isfile(data_path):
            # Single file
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return text
        elif os.path.isdir(data_path):
            # Directory - concatenate all files
            all_text = []
            for filename in sorted(os.listdir(data_path)):
                if filename.endswith('.txt'):
                    filepath = os.path.join(data_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    all_text.append(text)
                    print(f"Loaded: {filename}")
            return ''.join(all_text)
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")
    
    def encode(self, text: str) -> torch.Tensor:
        """Convert string to token indices."""
        return torch.tensor(
            [self.char_to_idx[char] for char in text],
            dtype=torch.long,
            device=self.device
        )
    
    def decode(self, indices: torch.Tensor) -> str:
        """Convert token indices to string."""
        chars = [self.idx_to_char[idx.item()] for idx in indices.flatten()]
        return ''.join(chars)
    
    def get_batch(self, split: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a random batch of data.
        
        Args:
            split: 'train' or 'val'
            batch_size: Number of sequences per batch
            
        Returns:
            x: Input sequences (batch, context_length)
            y: Target sequences (batch, context_length)
        """
        data = self.train_data if split == 'train' else self.val_data
        
        # Random start positions
        max_start = len(data) - self.context_length - 1
        start_indices = torch.randint(0, max_start, (batch_size,))
        
        # Create batches
        x_batch = torch.stack([
            data[start:start + self.context_length]
            for start in start_indices
        ])
        y_batch = torch.stack([
            data[start + 1:start + self.context_length + 1]
            for start in start_indices
        ])
        
        return x_batch.to(self.device), y_batch.to(self.device)


def download_gutenberg_sample(
    output_dir: str = 'data/gutenberg_subset/',
    num_books: int = 10
) -> str:
    """
    Download sample of Project Gutenberg books for experiments.
    
    Args:
        output_dir: Directory to save books
        num_books: Number of books to download
        
    Returns:
        Path to combined text file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Selected books (diverse, manageable size)
    BOOKS = [
        # Classic Literature
        ('1342-0.txt', 'Pride and Prejudice - Jane Austen'),
        ('2701-0.txt', 'Moby Dick - Herman Melville'),
        ('11-0.txt', 'Alice in Wonderland - Lewis Carroll'),
        
        # Fantasy/Adventure
        ('84-0.txt', 'Frankenstein - Mary Shelley'),
        ('35-0.txt', 'The Time Machine - H.G. Wells'),
        
        # Philosophy
        ('600-0.txt', 'The Prince - Niccolo Machiavelli'),
        
        # Mystery
        ('1661-0.txt', 'The Adventures of Sherlock Holmes - Arthur Conan Doyle'),
        
        # Science Fiction
        ('23684-0.txt', 'A Princess of Mars - Edgar Rice Burroughs'),
    ][:num_books]
    
    import requests
    
    all_text = []
    base_url = 'https://www.gutenberg.org/files/'
    
    for book_id, book_title in BOOKS:
        print(f"Downloading: {book_title}")
        
        try:
            # Get metadata to find correct folder
            meta_url = f"{base_url}{book_id.split('-')[0]}/{book_id}"
            meta_response = requests.get(meta_url)
            
            # Parse metadata to find text file
            for line in meta_response.text.split('\n'):
                if '.txt' in line.lower() and 'plain text' in line.lower():
                    # Extract actual text filename
                    text_filename = line.split('/')[-1].split('.')[0] + '.txt'
                    break
            
            text_url = f"{base_url}{book_id.split('-')[0]}/{text_filename}"
            response = requests.get(text_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save file
            filepath = os.path.join(output_dir, f"{book_title}.txt")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"  Saved: {filepath}")
            all_text.append(response.text)
            
        except Exception as e:
            print(f"  Error downloading {book_title}: {e}")
    
    combined_file = os.path.join(output_dir, 'all_books.txt')
    with open(combined_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join([t for t in all_text if t.strip()]))
    
    print(f"\nCombined dataset saved to: {combined_file}")
    print(f"Total characters: {sum(len(t) for t in all_text):,}")
    
    return combined_file
