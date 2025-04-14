from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from .tokenizer import H4Tokenizer

class ASRDataset(Dataset):
    def __init__(
            self,
            partition: Literal['train-clean-100', 'dev-clean', 'test-clean'],
            config: dict,
            tokenizer: H4Tokenizer,
            isTrainPartition: bool,
            global_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        # Store basic configuration
        self.config = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        # Get tokenizer ids for special tokens
        self.eos_token = self.tokenizer.eos_id
        self.sos_token = self.tokenizer.sos_id
        self.pad_token = self.tokenizer.pad_id

        # Setting up data paths
        self.fbank_dir = os.path.join(self.config['root'], self.partition, 'fbank')
        self.fbank_files = sorted([f for f in os.listdir(self.fbank_dir) if f.endswith('.npy')])

        # Take subset if specified
        subset_size = int(len(self.fbank_files) * self.config.get('subset', 1.0))
        self.fbank_files = self.fbank_files[:subset_size]
        self.length = len(self.fbank_files)

        # Handle text files for non-test partitions
        if self.partition != "test-clean":
            self.text_dir = os.path.join(self.config['root'], self.partition, 'text')
            self.text_files = sorted([f for f in os.listdir(self.text_dir) if f.endswith('.npy')])
            self.text_files = self.text_files[:subset_size]

            # Verify data alignment
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature and transcript files must match")

        # Initialize lists to store features and transcripts
        self.feats, self.transcripts_shifted, self.transcripts_golden = [], [], []

        # Initialize counters for character and token counts
        self.total_chars = 0
        self.total_tokens = 0

        # Initialize max length variables
        self.feat_max_len = 0
        self.text_max_len = 0

        # Initialize Welford's algorithm accumulators if needed for global_mvn
        if self.config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition:
                raise ValueError("global_stats must be provided for non-training partitions when using global_mvn")
            count = 0
            mean = torch.zeros(self.config['num_feats'], dtype=torch.float64)
            M2 = torch.zeros(self.config['num_feats'], dtype=torch.float64)

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            # Load features
            feat = np.load(os.path.join(self.fbank_dir, self.fbank_files[i]))
            feat = feat[:self.config['num_feats'], :]  # Truncate to num_feats
            self.feats.append(feat)
            self.feat_max_len = max(self.feat_max_len, feat.shape[1])

            # Update global statistics if needed
            if self.config['norm'] == 'global_mvn' and global_stats is None:
                feat_tensor = torch.FloatTensor(feat)  # (num_feats, time)
                batch_count = feat_tensor.shape[1]  # number of time steps
                count += batch_count

                # Update mean and M2 for all time steps at once
                delta = feat_tensor - mean.unsqueeze(1)  # (num_feats, time)
                mean += delta.mean(dim=1)  # (num_feats,)
                delta2 = feat_tensor - mean.unsqueeze(1)  # (num_feats, time)
                M2 += (delta * delta2).sum(dim=1)  # (num_feats,)

            # Handle transcripts for non-test partitions
            if self.partition != "test-clean":
                transcript = ''.join(np.load(os.path.join(self.text_dir, self.text_files[i])).tolist())
                self.total_chars += len(transcript)
                tokenized = self.tokenizer.encode(transcript)
                self.total_tokens += len(tokenized)

                # Track max length (add 1 for the sos/eos tokens)
                self.text_max_len = max(self.text_max_len, len(tokenized) + 1)
                self.transcripts_shifted.append([self.sos_token] + tokenized)
                self.transcripts_golden.append(tokenized + [self.eos_token])

        # Calculate average characters per token
        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0

        if self.partition != "test-clean":
            # Verify data alignment
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned")

        # Compute final global statistics if needed
        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                # Compute variance and standard deviation
                variance = M2 / (count - 1)
                self.global_std = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()

        # Initialize SpecAugment transforms
        self.time_mask = tat.TimeMasking(
            time_mask_param=config['specaug_conf']['time_mask_width_range'],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=config['specaug_conf']['freq_mask_width_range'],
            iid_masks=True
        )

    def get_avg_chars_per_token(self):
        return self.avg_chars_per_token

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load features
        feat = torch.FloatTensor(self.feats[idx])

        # Apply normalization
        if self.config['norm'] == 'global_mvn':
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-8)
        elif self.config['norm'] == 'none':
            pass

        # Get transcripts for non-test partitions
        shifted_transcript, golden_transcript = None, None
        if self.partition != "test-clean":
            shifted_transcript = torch.LongTensor(self.transcripts_shifted[idx])
            golden_transcript = torch.LongTensor(self.transcripts_golden[idx])

        return feat, shifted_transcript, golden_transcript

    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Collect transposed features from the batch into a list of tensors (B x T x F)
        batch_feats = [feat.transpose(0, 1) for feat, _, _ in batch]
        feat_lengths = torch.LongTensor([feat.shape[0] for feat in batch_feats])
        padded_feats = pad_sequence(batch_feats, batch_first=True, padding_value=0.0)  # Use 0.0 for float padding

        # Handle transcripts for non-test partitions
        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            batch_shifted = [shifted for _, shifted, _ in batch if shifted is not None]
            batch_golden = [golden for _, _, golden in batch if golden is not None]
            transcript_lengths = torch.LongTensor([len(transcript) for transcript in batch_shifted])
            padded_shifted = pad_sequence(batch_shifted, batch_first=True, padding_value=self.pad_token)
            padded_golden = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token)

        # Apply SpecAugment for training
        if self.config["specaug"] and self.isTrainPartition:
            padded_feats = padded_feats.permute(0, 2, 1)  # (B x F x T)
            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    padded_feats = self.freq_mask(padded_feats)
            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    padded_feats = self.time_mask(padded_feats)
            padded_feats = padded_feats.permute(0, 2, 1)  # (B x T x F)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths
